"""Run a trained model over one session's windows and aggregate."""
from __future__ import annotations

import numpy as np
import torch

from ..models import MilkEazeNet
from .session import SessionResult, aggregate_session


@torch.no_grad()
def predict_session(model: MilkEazeNet, frames: np.ndarray, hand: np.ndarray,
                    device: str = "cpu", min_swallow_prob: float = 0.5,
                    overlap_correction: float = 1.0) -> SessionResult:
    """
    frames             : (seq, C, T)
    hand               : (seq, F)
    overlap_correction : hop / window length; see :func:`aggregate_session`. Pass
                         ``ProcessedSession.volume_overlap_correction``.
    """
    model.eval()
    f = torch.from_numpy(frames).float().unsqueeze(0).to(device)   # (1, seq, C, T)
    h = torch.from_numpy(hand).float().unsqueeze(0).to(device)      # (1, seq, F)
    out = model(f, h)
    logits = out["logits"].squeeze(0).cpu().numpy()
    volume = out["volume"].squeeze(0).cpu().numpy()
    return aggregate_session(logits, volume, min_swallow_prob=min_swallow_prob,
                             overlap_correction=overlap_correction)
