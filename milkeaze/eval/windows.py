"""Whether a window-level score means anything on a given dataset.

Window classification accuracy is the number everyone reaches for first, and on the
pump rig it is close to meaningless. Windows are 2 s; at 42 cpm events land every
1.43 s, so nearly every window contains at least one suck and a model that answers
"suck" unconditionally scores ~99%. The Jul 18 batch labels out at 493 suck windows of
498 for exactly that reason.

This module makes that explicit instead of leaving it to be discovered after a training
run: :func:`label_granularity` reports how many events share a window and how short a
window would have to be to isolate one, and :func:`skill_score` rescales any accuracy
against the majority-class rate so "99%" is reported as the zero skill it is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..data.windowing import Window

IGNORE_INDEX = -100  # matches training.dataset


@dataclass
class LabelGranularity:
    """How events distribute across windows on one session."""

    n_windows: int
    n_events: int
    events_per_window_mean: float
    events_per_window_max: int
    occupancy: float                 # fraction of windows holding >= 1 event
    median_period_ms: float
    window_s: float
    max_window_s_for_single_event: float

    @property
    def resolves_single_events(self) -> bool:
        """True when a window is short enough to hold at most one event."""
        return self.window_s <= self.max_window_s_for_single_event

    def summary(self) -> str:
        verdict = "resolves" if self.resolves_single_events else "CANNOT resolve"
        return (f"{self.n_events} events over {self.n_windows} windows; "
                f"{self.events_per_window_mean:.2f} events/window "
                f"(max {self.events_per_window_max}), occupancy {self.occupancy:.1%}; "
                f"{self.window_s:.2f} s windows {verdict} single events "
                f"(need <= {self.max_window_s_for_single_event:.2f} s)")


def label_granularity(events_t_ms, windows: Sequence[Window]) -> LabelGranularity:
    """Count how many events fall in each window, and whether that is one or many.

    A window can hold at most one event only if it is shorter than the inter-event
    period, so the median period is the ceiling on a window length that still lets a
    per-window label mean "this event".
    """
    t = np.sort(np.asarray(events_t_ms, dtype=np.float64).ravel())
    starts = np.array([w.t_start_ms for w in windows], dtype=np.float64)
    ends = np.array([w.t_end_ms for w in windows], dtype=np.float64)

    lo = np.searchsorted(t, starts, side="left")
    hi = np.searchsorted(t, ends, side="right")
    per_window = (hi - lo).astype(np.int64)

    period_ms = float(np.median(np.diff(t))) if t.size >= 2 else float("nan")
    window_s = float(np.median(ends - starts)) / 1000.0 if windows else 0.0

    return LabelGranularity(
        n_windows=len(windows),
        n_events=int(t.size),
        events_per_window_mean=float(per_window.mean()) if per_window.size else 0.0,
        events_per_window_max=int(per_window.max()) if per_window.size else 0,
        occupancy=float((per_window > 0).mean()) if per_window.size else 0.0,
        median_period_ms=period_ms,
        window_s=window_s,
        max_window_s_for_single_event=period_ms / 1000.0 if np.isfinite(period_ms) else 0.0,
    )


def majority_class_rate(labels, ignore_index: int = IGNORE_INDEX) -> float:
    """Accuracy of always answering with the most common label.

    This is the floor any classifier has to clear to have learned anything at all.
    """
    y = np.asarray(labels).ravel()
    y = y[y != ignore_index]
    if y.size == 0:
        return float("nan")
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / y.size)


def skill_score(accuracy: float, baseline_accuracy: float) -> float:
    """Accuracy rescaled so the majority-class baseline is 0 and perfect is 1.

    Negative means the model is worse than answering with the most common class.
    Undefined when the baseline is already perfect, which itself says the labels carry
    no usable variation.
    """
    denom = 1.0 - baseline_accuracy
    if denom <= 0:
        return float("nan")
    return (accuracy - baseline_accuracy) / denom


def class_balance(labels, num_classes: int, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    """Per-class share of the labelled windows, for reporting alongside any accuracy."""
    y = np.asarray(labels).ravel()
    y = y[y != ignore_index]
    if y.size == 0:
        return np.zeros(num_classes, dtype=np.float64)
    return np.bincount(y, minlength=num_classes)[:num_classes] / y.size
