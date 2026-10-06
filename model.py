"""CIFAR-sized ResNet-18/34, implemented without torchvision models."""

import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 residual: bool = True):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        if self.residual:
            out = out + self.shortcut(x)
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, depth: int = 18, residual: bool = True):
        super().__init__()
        if depth not in (18, 34):
            raise ValueError('depth must be 18 or 34')
        blocks = (2, 2, 2, 2) if depth == 18 else (3, 4, 6, 3)
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        channels = 64
        stages = []
        for out_channels, count, stride in zip((64, 128, 256, 512), blocks, (1, 2, 2, 2)):
            layers = [BasicBlock(channels, out_channels, stride, residual)]
            layers.extend(BasicBlock(out_channels, out_channels, residual=residual)
                          for _ in range(count - 1))
            stages.append(nn.Sequential(*layers))
            channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.stages(self.stem(x)))
        return self.fc(torch.flatten(x, 1))
