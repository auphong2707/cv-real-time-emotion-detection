import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze-and-Excitation Block (EfficientNet/MobileNetV3)
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

# Mobile Inverted Bottleneck Convolution (EfficientNet)
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6, kernel_size=3, stride=1):
        super(MBConvBlock, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.stride = stride
        self.use_residual = (in_channels == out_channels and stride == 1)

        self.expand = nn.Conv2d(in_channels, hidden_dim, 1, bias=False) if expand_ratio != 1 else nn.Identity()
        self.bn0 = nn.BatchNorm2d(hidden_dim) if expand_ratio != 1 else nn.Identity()
        self.depthwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=stride, padding=kernel_size//2, groups=hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.se = SEBlock(hidden_dim)
        self.project = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.expand(x)
        out = F.silu(self.bn0(out))  # Swish activation
        out = self.depthwise(out)
        out = F.silu(self.bn1(out))
        out = self.se(out)
        out = self.project(out)
        out = self.bn2(out)
        if self.use_residual:
            out = out + identity
        return out

# Depthwise Separable Convolution (MobileNetV3)
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=kernel_size//2, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        out = self.depthwise(x)
        out = F.hardswish(self.bn1(out))  # h-swish activation
        out = self.pointwise(out)
        out = self.bn2(out)
        out = self.se(out)
        return out

# Complex EmotionCNN
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7, in_channels=3, image_size=224):
        super(EmotionCNN, self).__init__()
        self.image_size = image_size

        # Block 1: VGG-style (stacked 3x3 convs)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # image_size -> image_size/2
        self.dropout1 = nn.Dropout(0.1)

        # Block 2: MBConv (EfficientNet-inspired)
        self.mbconv1 = MBConvBlock(128, 256, expand_ratio=6, kernel_size=3, stride=2)  # image_size/2 -> image_size/4
        self.dropout2 = nn.Dropout(0.1)

        # Block 3: MBConv (additional for complexity)
        self.mbconv2 = MBConvBlock(256, 256, expand_ratio=6, kernel_size=3, stride=1)
        self.dropout3 = nn.Dropout(0.1)

        # Block 4: Depthwise Separable Conv (MobileNetV3-inspired)
        self.dsconv1 = DepthwiseSeparableConv(256, 512, kernel_size=3, stride=2)  # image_size/4 -> image_size/8
        self.dropout4 = nn.Dropout(0.1)

        # Block 5: Depthwise Separable Conv (additional for complexity)
        self.dsconv2 = DepthwiseSeparableConv(512, 512, kernel_size=3, stride=2)  # image_size/8 -> image_size/16
        self.dropout5 = nn.Dropout(0.1)

        # Global Average Pooling and FC layers
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 512)  # Intermediate FC layer
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout6 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1: VGG-style
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2: MBConv
        x = self.mbconv1(x)
        x = self.dropout2(x)

        # Block 3: MBConv
        x = self.mbconv2(x)
        x = self.dropout3(x)

        # Block 4: Depthwise Separable
        x = self.dsconv1(x)
        x = self.dropout4(x)

        # Block 5: Depthwise Separable
        x = self.dsconv2(x)
        x = self.dropout5(x)

        # Global pooling and FC layers
        x = self.global_pool(x)
        x = self.flatten(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout6(x)
        x = self.fc2(x)
        return x

# Example usage
if __name__ == "__main__":
    for img_size in [224, 112, 64]:
        model = EmotionCNN(num_classes=7, in_channels=3, image_size=img_size)
        x = torch.randn(1, 3, img_size, img_size)
        output = model(x)
        print(f"Image size: {img_size}x{img_size}, Output shape: {output.shape}")