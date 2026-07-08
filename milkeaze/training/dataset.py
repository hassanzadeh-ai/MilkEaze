"""Torch dataset + collation for variable-length sessions.

A sample is a whole session (a sequence of windows). Sessions have different lengths,
so ``collate_sessions`` pads to the batch max and returns lengths + masks. Class labels
may be absent (real data without events.csv): a label of -100 marks "ignore" so the
classification loss skips those windows while the volume loss still trains.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100  # matches nn.CrossEntropyLoss default ignore_index


@dataclass
class SessionSample:
    frames: np.ndarray          # (seq, C, T)
    hand: np.ndarray            # (seq, F)
    labels: np.ndarray | None   # (seq,) or None if no per-event labels
    volume: np.ndarray | None   # (seq,) mL per window, or None if no scale


class SessionDataset(Dataset):
    def __init__(self, samples: list[SessionSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SessionSample:
        return self.samples[idx]


def collate_sessions(batch: list[SessionSample]) -> dict[str, torch.Tensor]:
    b = len(batch)
    max_seq = max(s.frames.shape[0] for s in batch)
    C, T = batch[0].frames.shape[1], batch[0].frames.shape[2]
    F = batch[0].hand.shape[1]

    frames = torch.zeros(b, max_seq, C, T, dtype=torch.float32)
    hand = torch.zeros(b, max_seq, F, dtype=torch.float32)
    labels = torch.full((b, max_seq), IGNORE_INDEX, dtype=torch.long)
    volume = torch.zeros(b, max_seq, dtype=torch.float32)
    vol_mask = torch.zeros(b, max_seq, dtype=torch.float32)
    lengths = torch.zeros(b, dtype=torch.long)

    for i, s in enumerate(batch):
        n = s.frames.shape[0]
        lengths[i] = n
        frames[i, :n] = torch.from_numpy(s.frames)
        hand[i, :n] = torch.from_numpy(s.hand)
        if s.labels is not None:
            labels[i, :n] = torch.from_numpy(s.labels)
        if s.volume is not None:
            volume[i, :n] = torch.from_numpy(s.volume)
            vol_mask[i, :n] = 1.0

    return {
        "frames": frames, "hand": hand, "labels": labels,
        "volume": volume, "vol_mask": vol_mask, "lengths": lengths,
    }
