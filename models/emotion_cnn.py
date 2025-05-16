import torch
import torch.nn as nn
import torchvision.models as models

class EmotionCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3, image_size: tuple = (224, 224)):
        super().__init__()
        # --- 1) Load pretrained backbones ---
        vgg = models.vgg16(pretrained=False)
        eff = models.efficientnet_b0(pretrained=False)

        # --- 2) (Optional) Adapt first conv if in_channels != 3 ---
        if in_channels != 3:
            # VGG stem
            orig = vgg.features[0]
            vgg.features[0] = nn.Conv2d(in_channels,
                                        orig.out_channels,
                                        kernel_size=orig.kernel_size,
                                        stride=orig.stride,
                                        padding=orig.padding)
            # EfficientNet stem
            # features[0] is a ConvBNAct block: conv is at [0]
            orig = eff.features[0][0]
            eff.features[0][0] = nn.Conv2d(in_channels,
                                           orig.out_channels,
                                           kernel_size=orig.kernel_size,
                                           stride=orig.stride,
                                           padding=orig.padding,
                                           bias=False)

        # --- 3) Truncate each to the 28×28 output stage ---
        # VGG16: stop after block3_pool (features indices 0–16 inclusive)
        self.vgg_trunc = nn.Sequential(*list(vgg.features.children())[:17])
        # EfficientNet-B0: keep stem + first 5 MBConv blocks
        stem = eff.features[0]
        blocks = eff.features[1]
        self.eff_trunc = nn.Sequential(stem, *list(blocks[:5]))

        # --- 4) Head: global‐avg‐pool + FC layers ---
        # After concat we have 256 (VGG) + 40 (Eff) = 296 channels
        cat_channels = 256 + 40
        self.pool = nn.AdaptiveAvgPool2d(1)   # → (N, 296, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                     # → (N, 296)
            nn.Linear(cat_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_channels, H, W)
        x1 = self.vgg_trunc(x)    # → (N,256,28,28)
        x2 = self.eff_trunc(x)    # → (N, 40,28,28)
        x = torch.cat([x1, x2], dim=1)      # → (N,296,28,28)
        x = self.pool(x)                     # → (N,296,1,1)
        logits = self.classifier(x)          # → (N,num_classes)
        return logits
