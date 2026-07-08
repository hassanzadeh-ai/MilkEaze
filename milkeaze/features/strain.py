"""Strain features (per channel: 4 bend + 4 stretch).

Captures the mechanical rhythm of the suck cycle: statistics, dynamics, and rhythm
(dominant frequency = suck rate). Bend (Radial) and stretch (Arc) are kept separate.
"""
from __future__ import annotations

import numpy as np


def _dominant_freq(x: np.ndarray, fs: float) -> float:
    x = x - np.mean(x)
    if np.allclose(x, 0):
        return 0.0
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    if len(spec) <= 1:
        return 0.0
    k = int(np.argmax(spec[1:]) + 1)
    return float(freqs[k])


def _zero_cross_rate(x: np.ndarray) -> float:
    x = x - np.mean(x)
    return float(np.mean(np.abs(np.diff(np.sign(x))) > 0))


def strain_channel_features(ch: np.ndarray, fs: float) -> list[float]:
    """~9 features for one strain channel over one window."""
    mean = float(np.mean(ch))
    std = float(np.std(ch))
    rng = float(np.max(ch) - np.min(ch))
    rms = float(np.sqrt(np.mean(ch ** 2)))
    slope = float(np.polyfit(np.arange(len(ch)), ch, 1)[0]) if len(ch) > 1 else 0.0
    p2p = float(np.max(ch) - np.min(ch))
    zcr = _zero_cross_rate(ch)
    dom = _dominant_freq(ch, fs)      # suck rate proxy
    energy = float(np.sum(ch ** 2))
    return [mean, std, rng, rms, slope, p2p, zcr, dom, energy]


def strain_features(strain_block: np.ndarray, fs: float) -> np.ndarray:
    """``strain_block`` is (n_time, 8). Returns a flat feature vector."""
    feats: list[float] = []
    for c in range(strain_block.shape[1]):
        feats.extend(strain_channel_features(strain_block[:, c], fs))
    return np.asarray(feats, dtype=np.float32)
