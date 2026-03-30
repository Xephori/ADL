from __future__ import annotations
from typing import List
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Two conv layers with a skip connection.

    Input → Conv → BN → ReLU → Conv → BN → (+skip) → ReLU → Output
    If input channels != output channels, a 1x1 conv adjusts the skip path.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Skip connection: if channel sizes differ, use 1x1 conv to match
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class CNNBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        block_channels: List[int] | None = None,
    ) -> None:
        super().__init__()

        if block_channels is None:
            block_channels = [32, 64, 128, 256]

        layers: List[nn.Module] = []
        ch_in = in_channels
        for ch_out in block_channels:
            layers.append(ResidualBlock(ch_in, ch_out))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            ch_in = ch_out

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = block_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reshape = False
        if x.dim() == 5:
            batch_size, seq_len = x.shape[0], x.shape[1]
            x = x.view(batch_size * seq_len, *x.shape[2:])
            reshape = True

        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        if reshape:
            x = x.view(batch_size, seq_len, -1)
        return x
