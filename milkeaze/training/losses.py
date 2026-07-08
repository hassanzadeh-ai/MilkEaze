"""Combined loss: classification (masked) + volume regression (masked).

Both losses are masked so that:
  - windows without a class label (IGNORE_INDEX) don't contribute to classification;
  - padded / scale-less windows don't contribute to the volume MSE.

This is what lets a real session with a scale but no events.csv still train the volume
head while contributing nothing spurious to the classifier.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .dataset import IGNORE_INDEX


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits: (b, seq, C); labels: (b, seq) with IGNORE_INDEX where unlabeled
    b, s, c = logits.shape
    loss = F.cross_entropy(
        logits.reshape(b * s, c), labels.reshape(b * s),
        ignore_index=IGNORE_INDEX, reduction="mean",
    )
    # cross_entropy returns nan if every target is ignored; guard it.
    if torch.isnan(loss):
        return logits.new_zeros(())
    return loss


def volume_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # pred/target/mask: (b, seq)
    denom = mask.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask).sum() / denom


def combined_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
                  w_cls: float = 1.0, w_vol: float = 1.0,
                  train_classifier: bool = True) -> tuple[torch.Tensor, dict[str, float]]:
    vol = volume_loss(outputs["volume"], batch["volume"], batch["vol_mask"])
    if train_classifier:
        cls = classification_loss(outputs["logits"], batch["labels"])
    else:
        cls = outputs["logits"].new_zeros(())
    total = w_cls * cls + w_vol * vol
    return total, {"loss": float(total), "cls": float(cls), "vol": float(vol)}
