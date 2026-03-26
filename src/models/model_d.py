#Model D: Pretrained ResNet18 + Bidirectional LSTM + Temporal Attention
from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torchvision.models as models
from src.models.attention import TemporalAttention

class PretrainedBackbone(nn.Module):
    """ResNet18 pretrained on ImageNet, used as a frame-level feature extractor."""
    def __init__(self, freeze_early: bool = True) -> None:
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the final classification layer — we only want the features
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512  # ResNet18 outputs 512-dim features

        # Freeze early layers (they already know edges, textures, shapes)
        # Fine-tune layer3 + layer4 for our task
        if freeze_early:
            for name, param in self.features.named_parameters():
                # Freeze everything except layer3 and layer4
                if "5" not in name and "6" not in name and "7" not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reshape = False
        if x.dim() == 5:
            batch_size, seq_len = x.shape[0], x.shape[1]
            x = x.view(batch_size * seq_len, *x.shape[2:])
            reshape = True

        x = self.features(x)
        x = x.view(x.size(0), -1)

        if reshape:
            x = x.view(batch_size, seq_len, -1)
        return x

class ModelD(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        lstm_bidirectional: bool = True,
        lstm_dropout: float = 0.3,
        attention_hidden_dim: int = 128,
        classifier_dropout: float = 0.4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.cnn = PretrainedBackbone(freeze_early=True)
        feature_dim = self.cnn.feature_dim  # 512

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
