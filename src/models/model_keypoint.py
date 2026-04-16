#KeypointLSTM

from __future__ import annotations
import torch
import torch.nn as nn


class KeypointLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 63,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * 2  # bidirectional

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, _) = self.lstm(x)
        forward_h = h_n[-2]   
        backward_h = h_n[-1]  
        h = torch.cat([forward_h, backward_h], dim=1)  
        return self.classifier(h)
