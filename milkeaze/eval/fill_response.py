"""Strain response as a function of fill state, measured within a single capture.

Vacuum is constant for the whole of one run while fill falls monotonically, so a single
capture is a fill sweep at fixed vacuum and the vacuum/fill confound does not apply to a
*within*-run slope. That gives a way to test Steve's claim — a water-backed dome responds
roughly twice as strongly as an air-backed one — against the Jul 18 batch instead of
waiting for the deconfounding captures.

Stated as a prediction the data can refute: if the dome's gain halves between full and
empty, per-window strain amplitude should fall by about 50% of its full-reservoir value
over the run, on every capture, largely independent of vacuum level. Reporting it per
channel also answers "which channels carry the separation", though under index rather
than name while the channel map is unverified.

The obvious confounder in the other direction is that amplitude may drift for reasons
that merely correlate with elapsed time — the ring settling on the dome, thermal drift.
:func:`fill_response` therefore reports the rank correlation alongside the slope, and
:func:`split_half_consistency` checks the slope is present in both halves of a run rather
than being one monotone drift fitted end to end.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .confound import spearman


def window_amplitudes(frames: np.ndarray, channels: slice = slice(0, 8)) -> np.ndarray:
    """Per-window, per-channel RMS amplitude about the window's own mean.

    ``frames`` is the pipeline's ``(seq, channels, time)`` stack; the default slice takes
    the 8 strain channels. Centring per window removes any residual baseline the drift
    filter left, so this measures the size of the cycle rather than where it sits.
    """
    block = np.asarray(frames, dtype=np.float64)[:, channels, :]
    centred = block - block.mean(axis=2, keepdims=True)
    return np.sqrt((centred ** 2).mean(axis=2))


@dataclass
class FillResponse:
    """Linear fit of amplitude against fill fraction over one capture."""

    slope_per_fraction: float   # d(amplitude) / d(fill fraction)
    intercept: float            # fitted amplitude at fill = 0, i.e. run's emptiest
    rank_correlation: float
    n_windows: int

    @property
    def amplitude_full(self) -> float:
        return self.intercept + self.slope_per_fraction

    @property
    def amplitude_empty(self) -> float:
        return self.intercept

    @property
    def pct_change_full_to_empty(self) -> float:
        """Amplitude change from full to empty, as a percent of the full value.

        About −50% is what a gain that halves with an air-backed dome would produce.
        """
        full = self.amplitude_full
        if not np.isfinite(full) or full == 0:
            return float("nan")
        return 100.0 * (self.amplitude_empty - full) / full

    def summary(self) -> str:
        return (f"slope={self.slope_per_fraction:+.4g}/fraction "
                f"({self.pct_change_full_to_empty:+.1f}% full->empty), "
                f"rho={self.rank_correlation:+.3f}, n={self.n_windows}")


def fill_response(amplitude, fill_fraction) -> FillResponse:
    """Fit per-window amplitude against fill fraction for one channel."""
    a = np.asarray(amplitude, dtype=np.float64).ravel()
    f = np.asarray(fill_fraction, dtype=np.float64).ravel()
    if a.size != f.size:
        raise ValueError(f"length mismatch: {a.size} amplitudes, {f.size} fill values")
    if a.size < 3:
        raise ValueError(f"need at least 3 windows to fit a fill response, got {a.size}")

    if np.ptp(f) <= 0:
        return FillResponse(float("nan"), float(a.mean()), float("nan"), int(a.size))

    slope, intercept = np.polyfit(f, a, 1)
    return FillResponse(
        slope_per_fraction=float(slope),
        intercept=float(intercept),
        rank_correlation=spearman(f, a),
        n_windows=int(a.size),
    )


def per_channel_fill_response(amplitudes: np.ndarray, fill_fraction) -> list[FillResponse]:
    """One :class:`FillResponse` per channel of a ``(seq, channels)`` amplitude matrix."""
    amps = np.asarray(amplitudes, dtype=np.float64)
    if amps.ndim != 2:
        raise ValueError(f"amplitudes must be (n_windows, n_channels), got {amps.shape}")
    return [fill_response(amps[:, c], fill_fraction) for c in range(amps.shape[1])]


def split_half_consistency(amplitude, fill_fraction) -> tuple[FillResponse, FillResponse]:
    """Fit the response separately on each half of the run.

    A real gain-versus-fill effect appears in both halves. A slope that only exists end
    to end is a monotone drift wearing the fill covariate's clothes, since fill and
    elapsed time are themselves near-perfectly correlated within a run.
    """
    a = np.asarray(amplitude, dtype=np.float64).ravel()
    f = np.asarray(fill_fraction, dtype=np.float64).ravel()
    mid = a.size // 2
    return fill_response(a[:mid], f[:mid]), fill_response(a[mid:], f[mid:])
