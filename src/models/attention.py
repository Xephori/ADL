#Temporal attention module 
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int = 128) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, attention_dim)
        self.score_layer = nn.Linear(attention_dim, 1)

    def forward(
        self, lstm_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        energy = torch.tanh(self.projection(lstm_outputs))
        scores = self.score_layer(energy).squeeze(-1)
        attention_weights = F.softmax(scores, dim=1)

        context = torch.bmm(
            attention_weights.unsqueeze(1),  
            lstm_outputs,                
        ).squeeze(1)

        return context, attention_weights
