"""Validation utilities.

Two things matter for MilkEaze's clinical credibility:
  1. session-independent splits (no window from a session leaks across train/val);
  2. volume accuracy against the scale, reported as error + Bland-Altman stats.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def session_independent_split(session_ids: list[str], val_fraction: float = 0.2,
                              seed: int = 0) -> tuple[list[str], list[str]]:
    """Split at the *session* level so no session appears in both train and val."""
    rng = np.random.default_rng(seed)
    ids = list(dict.fromkeys(session_ids))  # unique, order-preserving
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_fraction)))
    val = set(ids[:n_val])
    return [i for i in ids if i not in val], [i for i in ids if i in val]


@dataclass
class VolumeAgreement:
    mae_ml: float
    bias_ml: float          # mean(pred - truth)
    loa_lower_ml: float     # Bland-Altman limits of agreement
    loa_upper_ml: float
    n: int


def volume_agreement(pred_ml: np.ndarray, true_ml: np.ndarray) -> VolumeAgreement:
    """Per-session predicted vs scale-measured volume agreement (Bland-Altman)."""
    pred_ml = np.asarray(pred_ml, dtype=np.float64)
    true_ml = np.asarray(true_ml, dtype=np.float64)
    diff = pred_ml - true_ml
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    return VolumeAgreement(
        mae_ml=float(np.mean(np.abs(diff))),
        bias_ml=bias,
        loa_lower_ml=bias - 1.96 * sd,
        loa_upper_ml=bias + 1.96 * sd,
        n=int(diff.size),
    )


def classification_metrics(pred: np.ndarray, true: np.ndarray, num_classes: int = 3) -> dict:
    """Per-class sensitivity/specificity + confusion matrix. Only meaningful once the
    classifier is trained on real per-event labels."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true, pred):
        if t < 0:
            continue
        cm[t, p] += 1
    metrics = {"confusion": cm.tolist(), "per_class": []}
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = float(tp / (tp + fn)) if (tp + fn) else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) else 0.0
        metrics["per_class"].append({"class": c, "sensitivity": sens, "specificity": spec})
    return metrics
