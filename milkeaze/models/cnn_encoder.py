"""Per-window CNN encoder.

Consumes the raw (channel, time) block for one window on the unified grid and returns
a learned embedding. This captures waveform morphology / transient shapes that the
hand features miss (e.g. the swallow click envelope).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: list[int], kernel_size: int,
                 embedding_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        c_prev = in_channels
        for c in channels:
            layers += [
                nn.Conv1d(c_prev, c, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ]
            c_prev = c
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(c_prev, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_channels, time) -> (batch, embedding_dim)."""
        h = self.conv(x)
        h = self.pool(h).squeeze(-1)
        return self.proj(h)
