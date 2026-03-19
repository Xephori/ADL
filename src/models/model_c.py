#Model C: CNN + Bidirectional LSTM + Temporal Attention
from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from src.models.attention import TemporalAttention
from src.models.cnn_backbone import CNNBackbone

class ModelC(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        block_channels: List[int] | None = None,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        lstm_bidirectional: bool = True,
        lstm_dropout: float = 0.3,
        attention_hidden_dim: int = 128,
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
        lstm_output_dim = lstm_hidden_size * direction_factor

        self.attention = TemporalAttention(
            input_dim=lstm_output_dim,
            attention_dim=attention_hidden_dim,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(lstm_output_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        frame_features = self.cnn(x)             
        lstm_out, _ = self.lstm(frame_features)   

        context, attn_weights = self.attention(lstm_out) 
        logits = self.classifier(context)             
        
        if return_attention:
            return {"logits": logits, "attention_weights": attn_weights}
        return logits
