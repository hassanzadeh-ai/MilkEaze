"""Full MilkEaze model.

Per window:
    raw frames --CNN--> embedding  ┐
                                    ├─ concat -> fusion -> LSTM timestep
    hand-feature vector ───────────┘

The LSTM runs over the sequence of windows in a session (window-level timesteps),
modeling temporal context across the feed. Two heads read the LSTM states:
classification (suck/swallow/silence) and volume regression (mL per window).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .cnn_encoder import CNNEncoder
from .heads import ClassificationHead, VolumeHead


@dataclass
class ModelDims:
    cnn_in_channels: int
    cnn_channels: list[int]
    cnn_kernel_size: int
    cnn_embedding_dim: int
    hand_feature_dim: int
    fusion_dim: int
    lstm_hidden: int
    lstm_layers: int
    bidirectional: bool
    dropout: float
    num_classes: int

    @classmethod
    def from_config(cls, model_cfg: dict, hand_feature_dim: int) -> "ModelDims":
        m = model_cfg
        return cls(
            cnn_in_channels=int(m["cnn"]["in_channels"]),
            cnn_channels=list(m["cnn"]["channels"]),
            cnn_kernel_size=int(m["cnn"]["kernel_size"]),
            cnn_embedding_dim=int(m["cnn"]["embedding_dim"]),
            hand_feature_dim=hand_feature_dim,
            fusion_dim=int(m["fusion_dim"]),
            lstm_hidden=int(m["lstm"]["hidden_size"]),
            lstm_layers=int(m["lstm"]["num_layers"]),
            bidirectional=bool(m["lstm"]["bidirectional"]),
            dropout=float(m["lstm"]["dropout"]),
            num_classes=int(m["heads"]["num_classes"]),
        )


class MilkEazeNet(nn.Module):
    def __init__(self, dims: ModelDims):
        super().__init__()
        self.dims = dims
        self.cnn = CNNEncoder(
            in_channels=dims.cnn_in_channels,
            channels=dims.cnn_channels,
            kernel_size=dims.cnn_kernel_size,
            embedding_dim=dims.cnn_embedding_dim,
        )
        self.fusion = nn.Sequential(
            nn.Linear(dims.cnn_embedding_dim + dims.hand_feature_dim, dims.fusion_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(dims.fusion_dim),
        )
        self.lstm = nn.LSTM(
            input_size=dims.fusion_dim,
            hidden_size=dims.lstm_hidden,
            num_layers=dims.lstm_layers,
            batch_first=True,
            bidirectional=dims.bidirectional,
            dropout=dims.dropout if dims.lstm_layers > 1 else 0.0,
        )
        head_in = dims.lstm_hidden * (2 if dims.bidirectional else 1)
        self.cls_head = ClassificationHead(head_in, dims.num_classes)
        self.vol_head = VolumeHead(head_in)

    def forward(self, frames: torch.Tensor, hand: torch.Tensor,
                lengths: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """
        frames : (batch, seq, channels, time) raw window frames
        hand   : (batch, seq, hand_feature_dim)
        returns: {"logits": (b, seq, num_classes), "volume": (b, seq)}
        """
        b, s, c, t = frames.shape
        emb = self.cnn(frames.reshape(b * s, c, t)).reshape(b, s, -1)
        fused = self.fusion(torch.cat([emb, hand], dim=-1))

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                fused, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=s)
        else:
            out, _ = self.lstm(fused)

        return {"logits": self.cls_head(out), "volume": self.vol_head(out)}
