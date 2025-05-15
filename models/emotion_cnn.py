import torch
import torch.nn as nn
import torch.nn.functional as F

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, expansion_factor=6, stride=1):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        expand_channels = in_channels * expansion_factor

        # Expansion layer (1x1 conv)
        self.expand = nn.Conv2d(in_channels, expand_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(expand_channels)

        # Depthwise convolution (3x3)
        self.depthwise = nn.Conv2d(expand_channels, expand_channels, kernel_size=kernel_size, 
                                 stride=stride, padding=kernel_size//2, groups=expand_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_channels)

        # Projection layer (1x1 conv)
        self.project = nn.Conv2d(expand_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Residual connection condition
        self.add_residual = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        identity = x

        out = self.expand(x)
        out = self.bn0(out)
        out = F.relu6(out)

        out = self.depthwise(out)
        out = self.bn1(out)
        out = F.relu6(out)

        out = self.project(out)
        out = self.bn2(out)

        if self.add_residual:
            out = out + identity

        return out

class EmotionCNN(nn.Module):
    def __init__(self, num_classes, in_channels, image_size):
        super(EmotionCNN, self).__init__()
        
        # Block 1: Initial convolutional layers
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            MBConvBlock(128, 256, stride=1),
            MBConvBlock(256, 256, stride=1),
            MBConvBlock(256, 256, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            MBConvBlock(256, 512, stride=1),
            MBConvBlock(512, 512, stride=1),
            MBConvBlock(512, 512, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 5
        self.block5 = nn.Sequential(
            MBConvBlock(512, 512, stride=1),
            MBConvBlock(512, 512, stride=1),
            MBConvBlock(512, 512, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate the flattened size based on image_size
        spatial_size = image_size // 32  # After 5 max pooling layers
        flattened_size = spatial_size * spatial_size * 512
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.fc(x)
        return x
