"""Multi-rate resampling onto a single uniform grid.

Strain, IMU and mic arrive at three different (and jittery) rates. The model wants
one rectangular tensor, so every stream is linearly interpolated onto a shared
``target_grid_hz`` grid expressed in milliseconds since session start.

The mic is the exception: at 8 kHz, linear interpolation down to 200 Hz would alias
the acoustic band into nonsense. :func:`mic_envelope` instead reduces it with a
short-time RMS envelope, which is what the CNN channel is meant to carry. The
native-rate samples are kept separately for the acoustic hand features.
"""
from __future__ import annotations

import numpy as np


def make_grid(t0_ms: float, t1_ms: float, grid_hz: float) -> np.ndarray:
    """Uniform time grid in ms, inclusive of ``t0_ms``, not exceeding ``t1_ms``."""
    if not np.isfinite([t0_ms, t1_ms]).all():
        raise ValueError(f"non-finite grid bounds: t0={t0_ms}, t1={t1_ms}")
    if t1_ms <= t0_ms:
        raise ValueError(f"empty time span: t0={t0_ms} >= t1={t1_ms}")
    if grid_hz <= 0:
        raise ValueError(f"grid_hz must be positive, got {grid_hz}")

    step_ms = 1000.0 / grid_hz
    n = int(np.floor((t1_ms - t0_ms) / step_ms)) + 1
    return t0_ms + np.arange(n, dtype=np.float64) * step_ms


def resample_linear(t_ms: np.ndarray, values: np.ndarray, grid_ms: np.ndarray) -> np.ndarray:
    """Linearly interpolate ``values`` (n, c) sampled at ``t_ms`` onto ``grid_ms``.

    Timestamps must be strictly increasing; duplicated timestamps are collapsed to
    their first occurrence rather than silently producing infinite slopes.
    """
    t_ms = np.asarray(t_ms, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if t_ms.shape[0] != values.shape[0]:
        raise ValueError(f"timestamp/value length mismatch: {t_ms.shape[0]} vs {values.shape[0]}")
    if t_ms.shape[0] < 2:
        raise ValueError("need at least 2 samples to resample")

    if np.any(np.diff(t_ms) <= 0):
        keep = np.concatenate([[True], np.diff(t_ms) > 0])
        t_ms, values = t_ms[keep], values[keep]

    out = np.empty((grid_ms.shape[0], values.shape[1]), dtype=np.float32)
    for c in range(values.shape[1]):
        out[:, c] = np.interp(grid_ms, t_ms, values[:, c])
    return out


def mic_envelope(t_ms: np.ndarray, mic: np.ndarray, grid_ms: np.ndarray,
                 grid_hz: float) -> np.ndarray:
    """Short-time RMS envelope of the mic, evaluated on ``grid_ms``.

    Each grid point summarises one grid period of native-rate samples, so the
    envelope is an energy measure rather than an aliased waveform. Grid points with
    no underlying mic samples inherit the nearest populated value.
    """
    t_ms = np.asarray(t_ms, dtype=np.float64)
    mic = np.asarray(mic, dtype=np.float64)
    if mic.ndim == 1:
        mic = mic[:, None]

    half_win_ms = 0.5 * (1000.0 / grid_hz)
    # bin native samples by grid cell; edges straddle each grid point
    edges = np.concatenate([grid_ms - half_win_ms, [grid_ms[-1] + half_win_ms]])
    idx = np.searchsorted(edges, t_ms, side="right") - 1
    valid = (idx >= 0) & (idx < grid_ms.shape[0])
    idx, sel = idx[valid], np.flatnonzero(valid)

    n_grid = grid_ms.shape[0]
    out = np.zeros((n_grid, mic.shape[1]), dtype=np.float32)
    counts = np.bincount(idx, minlength=n_grid).astype(np.float64)
    populated = counts > 0
    for c in range(mic.shape[1]):
        energy = np.bincount(idx, weights=mic[sel, c] ** 2, minlength=n_grid)
        rms = np.zeros(n_grid, dtype=np.float64)
        rms[populated] = np.sqrt(energy[populated] / counts[populated])
        if not populated.all() and populated.any():
            # carry the nearest populated cell across gaps instead of injecting silence
            src = np.flatnonzero(populated)
            rms = np.interp(np.arange(n_grid), src, rms[src])
        out[:, c] = rms
    return out
