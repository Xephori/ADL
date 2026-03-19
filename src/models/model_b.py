#Model B: CNN + Bidirectional LSTM

from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
from src.models.cnn_backbone import CNNBackbone

class ModelB(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        block_channels: List[int] | None = None,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        lstm_bidirectional: bool = True,
        lstm_dropout: float = 0.3,
        classifier_dropout: float = 0.4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.cnn = CNNBackbone(block_channels=block_channels)
        feature_dim = self.cnn.feature_dim

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=lstm_bidirectional,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
        )

        direction_factor = 2 if lstm_bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(lstm_hidden_size * direction_factor, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frame_features = self.cnn(x)       
        lstm_out, (h_n, _) = self.lstm(frame_features) 
        if self.lstm.bidirectional:
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  
        else:
            hidden = h_n[-1] 
        logits = self.classifier(hidden)
        return logits
