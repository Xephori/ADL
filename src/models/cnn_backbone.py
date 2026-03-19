from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

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
            layers.extend([
                nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
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
