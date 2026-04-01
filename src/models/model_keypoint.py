"""
KeypointLSTM — classifies sign language words from sequences of hand keypoints.

Input:  [batch, num_frames, 63]   (63 = 21 landmarks × x,y,z)
Output: [batch, num_classes]

Architecture:
  BiLSTM (hidden=128, layers=2, dropout=0.3)
  → take last hidden state from both directions → concat → [batch, 256]
  → Dropout(0.3)
  → Linear(256, num_classes)

~500K parameters — much smaller than CNN models, much less overfitting risk
on the 111-video WLASL dataset.
"""
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
        # x: [batch, num_frames, 63]
        out, (h_n, _) = self.lstm(x)
        # h_n: [num_layers * 2, batch, hidden_size]
        # Take the last layer's forward and backward hidden states
        forward_h = h_n[-2]   # [batch, hidden_size]
        backward_h = h_n[-1]  # [batch, hidden_size]
        h = torch.cat([forward_h, backward_h], dim=1)  # [batch, hidden_size*2]
        return self.classifier(h)
