"""Session aggregation.

Turns per-window model outputs into the numbers a parent actually sees:
total sucks, total swallows, and cumulative volume over the feed. This is the
"over a 15-20 minute session, accumulate then report" behavior from the product spec.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SessionResult:
    suck_count: int
    swallow_count: int
    total_volume_ml: float
    per_window_class: np.ndarray   # (seq,)
    per_window_volume_ml: np.ndarray  # (seq,)

    def summary(self) -> str:
        return (f"~{self.total_volume_ml:.0f} mL, {self.swallow_count} swallows, "
                f"{self.suck_count} sucks")


def aggregate_session(class_logits: np.ndarray, volume_pred: np.ndarray,
                      min_swallow_prob: float = 0.5) -> SessionResult:
    """
    class_logits : (seq, num_classes)
    volume_pred  : (seq,) mL per window
    """
    probs = _softmax(class_logits, axis=-1)
    pred_class = probs.argmax(axis=-1)

    suck = int(np.sum(pred_class == 1))
    swallow = int(np.sum((pred_class == 2) & (probs[:, 2] >= min_swallow_prob)))

    # only count volume on windows the model believes carry transfer (swallow-ish)
    transfer_mask = (pred_class == 2).astype(np.float32)
    total_volume = float(np.sum(volume_pred * transfer_mask)) if transfer_mask.any() \
        else float(np.sum(volume_pred))

    return SessionResult(
        suck_count=suck, swallow_count=swallow, total_volume_ml=total_volume,
        per_window_class=pred_class, per_window_volume_ml=np.asarray(volume_pred),
    )


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)
