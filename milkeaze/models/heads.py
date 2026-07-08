"""Output heads on top of the per-window LSTM states.

- ClassificationHead : per window -> {silence, suck, swallow}
- VolumeHead         : per window -> volume (mL) transferred in that window.
                       Session total = sum over windows (see inference/session.py).
- SequenceLabelingHead : PROTOTYPE only, disabled by default. Intended to localize
                         exact event boundaries at sub-window resolution once we have
                         per-event labels. Kept here so the interface is stable.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, num_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)  # logits (batch, time, num_classes)


class VolumeHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, 1),
            nn.Softplus(),  # volume per window is non-negative
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h).squeeze(-1)  # (batch, time)


class SequenceLabelingHead(nn.Module):
    """PROTOTYPE — not wired into training yet (needs per-event labels)."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError(
            "SequenceLabelingHead is a prototype; enable once sub-window event "
            "boundaries are available in the training data."
        )
