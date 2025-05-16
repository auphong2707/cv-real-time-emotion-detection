import torch
import torch.nn as nn
from torchvision import models

class EmotionCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3, image_size: tuple = (224, 224)):
        super(EmotionCNN, self).__init__()
        
        # 1) VGG-16 trunk up to block3_pool (features[:17])
        vgg = models.vgg16(pretrained=False)
        self.vgg_trunk = nn.Sequential(*list(vgg.features)[:17])  # up to 28x28x256
        
        # 2) 1x1 convolution to project 256 -> 40 channels
        # instead of a single 256→40:
        self.proj = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 40, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )

        
        # 3) EfficientNet-B0 tail from Block 4 onward
        eff = models.efficientnet_b0(pretrained=False)
        # According to torchvision's feature list:
        # features[0] = stem, [1]=MBConv1, [2]=MBConv2, [3]=MBConv3 (output 28x28)
        # so Block4 starts at index 4
        eff_blocks = list(eff.features)[4:]
        self.eff_tail = nn.Sequential(*eff_blocks)
        self.avgpool = eff.avgpool  # AdaptiveAvgPool2d
        
        # 4) Classifier: override to match num_classes
        # eff.classifier = [Dropout, Linear(in_features, 1000)]
        dropout_p = eff.classifier[0].p
        in_feats = eff.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_feats, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # VGG trunk
        x = self.vgg_trunk(x)          # -> [B, 256, 28, 28]
        # projection
        x = self.proj(x)               # -> [B, 40, 28, 28]
        # EfficientNet tail
        x = self.eff_tail(x)           # -> [B, 1280, 7, 7]
        x = self.avgpool(x)            # -> [B, 1280, 1, 1]
        x = torch.flatten(x, 1)        # -> [B, 1280]
        # classifier
        x = self.classifier(x)         # -> [B, num_classes]
        return x
