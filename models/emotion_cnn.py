import torch
import torch.nn as nn

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expansion=6, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        hidden_dim = in_channels * expansion
        # Expand phase
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        ) if expansion != 1 else nn.Identity()
        # Depthwise convolution
        self.depth = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )
        # Squeeze-and-Excitation
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, se_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_channels, hidden_dim, 1),
            nn.Sigmoid()
        )
        # Projection phase
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        out = self.expand(x)
        out = self.depth(out)
        out = self.se(out) * out
        out = self.project(out)
        if self.use_res_connect:
            return x + out
        return out

class EmotionCNN(nn.Module):
    def __init__(self, num_classes, in_channels, image_size):
        super(EmotionCNN, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.image_size = image_size

        # VGG-16 up to block3_pool, dynamic in_channels
        self.vgg_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # -> (image_size/2)x(image_size/2)x64
        self.vgg_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # -> (image_size/4)x(image_size/4)x128
        self.vgg_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # -> (image_size/8)x(image_size/8)x256

        # EfficientNet-B0 blocks manually up to MBConv stage at matching resolution
        self.eff_stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(inplace=True)
        )  # -> (image_size/2)x(image_size/2)x32
        self.eff_mb1 = MBConvBlock(32, 16, kernel_size=3, stride=1, expansion=1)
        self.eff_mb2 = MBConvBlock(16, 24, kernel_size=3, stride=2, expansion=6)
        self.eff_mb3 = MBConvBlock(24, 24, kernel_size=3, stride=1, expansion=6)
        self.eff_mb4 = MBConvBlock(24, 40, kernel_size=5, stride=2, expansion=6)
        self.eff_mb5 = MBConvBlock(40, 40, kernel_size=5, stride=1, expansion=6)
        # after eff_mb5 -> (image_size/8)x(image_size/8)x40

        # Classification head
        concat_channels = 256 + 40
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> [B, concat_channels, 1, 1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(concat_channels, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # VGG branch
        v = self.vgg_block1(x)
        v = self.vgg_block2(v)
        v = self.vgg_block3(v)  # [B,256,H/8,W/8]

        # EfficientNet branch
        e = self.eff_stem(x)
        e = self.eff_mb1(e)
        e = self.eff_mb2(e)
        e = self.eff_mb3(e)
        e = self.eff_mb4(e)
        e = self.eff_mb5(e)      # [B,40,H/8,W/8]

        # Concatenate feature maps
        f = torch.cat([v, e], dim=1)  # [B,296,H/8,W/8]

        # Head
        out = self.pool(f)            # [B,296,1,1]
        out = self.classifier(out)    # [B,num_classes]
        return out
