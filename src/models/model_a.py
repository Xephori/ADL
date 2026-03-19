#Model A : CNN + Average Pooling (Baseline).
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
from src.models.cnn_backbone import CNNBackbone

class ModelA(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        block_channels: List[int] | None = None,
        classifier_dropout: float = 0.4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.cnn = CNNBackbone(block_channels=block_channels)
        feature_dim = self.cnn.feature_dim

        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frame_features = self.cnn(x)         
        video_feature = frame_features.mean(dim=1) 
        logits = self.classifier(video_feature)     
        return logits
