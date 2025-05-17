import torch
import torch.nn as nn
from torchvision import models

class EmotionCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3, image_size: tuple = (224, 224)):
        super(EmotionCNN, self).__init__()
        
        # VGG-16 trunk
        vgg = models.vgg16(pretrained=True)
        vgg_trunk = list(vgg.features)[:17]  # up to 28x28x256

        # Projection block: 256 → 40
        proj = [
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 40, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        ]

        # EfficientNet-B0 tail
        eff = models.efficientnet_b0(pretrained=False)
        eff_tail = list(eff.features)[4:]  # from Block 4 onward

        # Final head: avgpool + flatten + classifier
        dropout_p = eff.classifier[0].p
        in_feats = eff.classifier[1].in_features
        head = [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Dropout(dropout_p),
            nn.Linear(in_feats, num_classes)
        ]

        # Unified model chain
        self.model = nn.Sequential(
            *vgg_trunk,
            *proj,
            *eff_tail,
            *head
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
