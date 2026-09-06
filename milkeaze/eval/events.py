"""Event-level agreement between a predicted event list and a reference one.

The product claim is a *count* — how many sucks, how much milk — so the metric that
decides whether a model works has to compare event lists, not label vectors. A window
classifier can score 99% accuracy on this rig and still be worthless, because at 42 cpm
almost every 2 s window contains a suck; :mod:`milkeaze.eval.windows` quantifies that
trap directly.

Matching is greedy nearest-first and strictly one-to-one: the closest surviving pair
inside the tolerance is committed, then both endpoints are retired. That is
deterministic, symmetric in the two lists, and it never lets one reference event absorb
two predictions — which matters, because absorbing them is exactly how a doubling bug
hides. A detector emitting every cycle twice scores recall 1.0 and precision 0.5 here,
instead of looking perfect.

Everything in this module is numpy-only and never imports torch, so it runs against
detector output and baselines on machines where the training stack is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EventMatch:
    """Outcome of matching a predicted event list against a reference list."""

    n_pred: int
    n_true: int
    n_matched: int
    tolerance_ms: float
    bias_ms: float          # mean(pred - true) over matched pairs; sign = predicted late
    mad_ms: float           # median absolute timing deviation, outlier-resistant
    rmse_ms: float
    pairs: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 2), np.int64))

    @property
    def precision(self) -> float:
        return self.n_matched / self.n_pred if self.n_pred else 0.0

    @property
    def recall(self) -> float:
        return self.n_matched / self.n_true if self.n_true else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def count_error(self) -> int:
        """Signed miscount, which is what a clinician actually sees."""
        return self.n_pred - self.n_true

    @property
    def count_error_pct(self) -> float:
        return 100.0 * self.count_error / self.n_true if self.n_true else float("nan")

    def summary(self) -> str:
        return (f"n_pred={self.n_pred} n_true={self.n_true} matched={self.n_matched} "
                f"P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f} "
                f"count_err={self.count_error:+d} ({self.count_error_pct:+.1f}%) "
                f"bias={self.bias_ms:+.1f} ms MAD={self.mad_ms:.1f} ms")


def _candidate_pairs(pred: np.ndarray, true: np.ndarray, tolerance_ms: float):
    """Every (|dt|, pred_idx, true_idx, dt) pair within tolerance, cheaply.

    Both inputs are sorted, so the admissible reference events for a prediction form a
    contiguous slice; searchsorted finds it without the full cross product.
    """
    lo = np.searchsorted(true, pred - tolerance_ms, side="left")
    hi = np.searchsorted(true, pred + tolerance_ms, side="right")
    out = []
    for i in range(pred.size):
        for j in range(lo[i], hi[i]):
            dt = float(pred[i] - true[j])
            out.append((abs(dt), i, j, dt))
    return out


def match_events(pred_t_ms, true_t_ms, tolerance_ms: float) -> EventMatch:
    """Greedily match predicted event times to reference times, one to one.

    ``tolerance_ms`` is how far apart two events may sit and still be the same event.
    Pick it from the cycle period rather than by taste — see :func:`tolerance_from_period`.
    """
    if tolerance_ms <= 0:
        raise ValueError(f"tolerance_ms must be positive, got {tolerance_ms}")

    pred = np.sort(np.asarray(pred_t_ms, dtype=np.float64).ravel())
    true = np.sort(np.asarray(true_t_ms, dtype=np.float64).ravel())

    candidates = sorted(_candidate_pairs(pred, true, tolerance_ms))
    pred_taken = np.zeros(pred.size, dtype=bool)
    true_taken = np.zeros(true.size, dtype=bool)
    pairs: list[tuple[int, int]] = []
    deltas: list[float] = []
    for _, i, j, dt in candidates:
        if pred_taken[i] or true_taken[j]:
            continue
        pred_taken[i] = true_taken[j] = True
        pairs.append((i, j))
        deltas.append(dt)

    d = np.asarray(deltas, dtype=np.float64)
    return EventMatch(
        n_pred=int(pred.size),
        n_true=int(true.size),
        n_matched=int(d.size),
        tolerance_ms=float(tolerance_ms),
        bias_ms=float(np.mean(d)) if d.size else float("nan"),
        mad_ms=float(np.median(np.abs(d - np.median(d)))) if d.size else float("nan"),
        rmse_ms=float(np.sqrt(np.mean(d ** 2))) if d.size else float("nan"),
        pairs=np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
    )


def tolerance_from_period(true_t_ms, fraction: float = 0.25,
                          floor_ms: float = 50.0) -> float:
    """A matching tolerance derived from the reference cycle period.

    A quarter period is the useful default: wide enough to absorb the ~7 ms cross-board
    alignment scatter measured on the Jul 18 batch, narrow enough that a prediction
    cannot match the neighbouring cycle. The floor keeps very fast reference streams
    from producing a tolerance finer than the alignment error itself.
    """
    t = np.sort(np.asarray(true_t_ms, dtype=np.float64).ravel())
    if t.size < 2:
        return floor_ms
    return max(floor_ms, fraction * float(np.median(np.diff(t))))


def rate_cpm(t_ms) -> float:
    """Event rate in cycles per minute, from the median inter-event interval.

    Median rather than count/duration so a single dropped cycle shifts the estimate by
    nothing instead of by its share of the run.
    """
    t = np.sort(np.asarray(t_ms, dtype=np.float64).ravel())
    if t.size < 2:
        return 0.0
    period_ms = float(np.median(np.diff(t)))
    return 60_000.0 / period_ms if period_ms > 0 else 0.0
