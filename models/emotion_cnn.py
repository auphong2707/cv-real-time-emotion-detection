import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        reduced = int(in_channels * se_ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced, 1),
            nn.SiLU(),
            nn.Conv2d(reduced, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale

class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, expand_ratio, se_ratio=0.25, drop_connect_rate=0.0):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        mid_ch = in_ch * expand_ratio

        self.expand = nn.Identity() if expand_ratio == 1 else nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU()
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size, stride, kernel_size // 2,
                      groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU()
        )

        self.se = SEBlock(mid_ch, se_ratio)

        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        self.drop_connect_rate = drop_connect_rate

    def stochastic_depth(self, x, p, training):
        if not training or p == 0.0:
            return x
        keep_prob = 1 - p
        shape = [x.size(0), 1, 1, 1]
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x / keep_prob * binary_tensor

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        if self.use_residual:
            x = self.stochastic_depth(x, self.drop_connect_rate, self.training)
            x += identity
        return x

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=8, in_channels=3, image_size=(224, 224), drop_rate=0.2):
        super().__init__()

        # Slightly more complex structure than EfficientNetB0
        blocks_cfg = [
            # expand, out_ch, num_blocks, kernel, stride
            (1,  16, 1, 3, 1),
            (6,  24, 2, 3, 2),
            (6,  40, 3, 5, 2),  # ↑ slightly deeper
            (6,  80, 3, 3, 2),
            (6, 112, 4, 5, 1),  # ↑ deeper
            (6, 192, 4, 5, 2),
            (6, 320, 2, 3, 1),  # ↑ deeper tail
        ]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        in_ch = 32
        block_id = 0
        total_blocks = sum(cfg[2] for cfg in blocks_cfg)
        blocks = []
        for expand, out_ch, num_blocks, k, s in blocks_cfg:
            for i in range(num_blocks):
                stride = s if i == 0 else 1
                drop_connect = drop_rate * block_id / total_blocks
                blocks.append(MBConv(in_ch, out_ch, k, stride, expand, drop_connect_rate=drop_connect))
                in_ch = out_ch
                block_id += 1

        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
