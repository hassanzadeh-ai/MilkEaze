"""Synthetic session generator.

Produces sequences of per-window tensors matching what the real pipeline emits:
  frames : (seq, in_channels, win_len) on the unified grid
  hand   : (seq, hand_feature_dim)
  labels : (seq,) in {0 silence, 1 suck, 2 swallow}
  volume : (seq,) mL transferred per window (non-negative; concentrated on swallows)

The feeding rhythm is modeled as a Markov-ish alternation of suck bursts punctuated by
occasional swallows, which is the coarse structure we expect during a feed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features.assemble import FEATURE_DIM


@dataclass
class SyntheticSession:
    frames: np.ndarray   # (seq, C, T)
    hand: np.ndarray     # (seq, FEATURE_DIM)
    labels: np.ndarray   # (seq,)
    volume: np.ndarray   # (seq,)  mL per window

    @property
    def session_volume_ml(self) -> float:
        return float(self.volume.sum())


# class-conditional signatures used to make the synthetic problem learnable
_SUCK_RATE_HZ = 1.2      # ~1-2 sucks/sec
_SWALLOW_ML_MEAN = 1.1   # mL per swallow window
_SWALLOW_ML_STD = 0.4


def _class_frames(cls: int, in_channels: int, win_len: int, grid_hz: float,
                  rng: np.random.Generator) -> np.ndarray:
    t = np.arange(win_len) / grid_hz
    x = rng.normal(0, 0.05, size=(in_channels, win_len)).astype(np.float32)

    # channels 0..7 strain, 8..9 mic-envelope, 10..15 imu
    if cls == 1:  # suck: rhythmic oscillation on strain channels
        for c in range(8):
            phase = rng.uniform(0, 2 * np.pi)
            x[c] += 0.6 * np.sin(2 * np.pi * _SUCK_RATE_HZ * t + phase).astype(np.float32)
        x[8:10] += 0.1 * np.abs(np.sin(2 * np.pi * _SUCK_RATE_HZ * t)).astype(np.float32)
    elif cls == 2:  # swallow: transient acoustic burst + strain bump
        center = rng.integers(win_len // 4, 3 * win_len // 4)
        env = np.exp(-0.5 * ((np.arange(win_len) - center) / (0.05 * grid_hz)) ** 2)
        x[8:10] += (0.8 * env).astype(np.float32)
        x[:8] += (0.3 * env).astype(np.float32)
    # cls == 0 silence: noise only
    return x


def _class_hand(cls: int, rng: np.random.Generator) -> np.ndarray:
    base = rng.normal(0, 1.0, size=FEATURE_DIM).astype(np.float32)
    base += float(cls) * 0.5  # class-dependent mean shift so it's learnable
    return base


def generate_session(n_windows: int = 120, in_channels: int = 16, win_len: int = 400,
                     grid_hz: float = 200.0, seed: int | None = None) -> SyntheticSession:
    rng = np.random.default_rng(seed)
    frames = np.empty((n_windows, in_channels, win_len), dtype=np.float32)
    hand = np.empty((n_windows, FEATURE_DIM), dtype=np.float32)
    labels = np.empty(n_windows, dtype=np.int64)
    volume = np.zeros(n_windows, dtype=np.float32)

    state = 0
    for i in range(n_windows):
        # coarse rhythm: mostly sucking, periodic swallows, some silence
        r = rng.random()
        if state == 1 and r < 0.25:
            cls = 2  # swallow follows a suck burst
        elif r < 0.15:
            cls = 0
        else:
            cls = 1
        state = cls

        frames[i] = _class_frames(cls, in_channels, win_len, grid_hz, rng)
        hand[i] = _class_hand(cls, rng)
        labels[i] = cls
        if cls == 2:
            volume[i] = max(0.0, rng.normal(_SWALLOW_ML_MEAN, _SWALLOW_ML_STD))
    return SyntheticSession(frames=frames, hand=hand, labels=labels, volume=volume)


def generate_dataset(n_sessions: int = 64, seed: int = 0, **kwargs) -> list[SyntheticSession]:
    rng = np.random.default_rng(seed)
    return [generate_session(seed=int(rng.integers(0, 2 ** 31)), **kwargs)
            for _ in range(n_sessions)]
