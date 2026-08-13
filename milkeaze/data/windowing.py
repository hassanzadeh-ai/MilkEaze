"""Fixed-length windowing of the unified grid.

One window is one LSTM timestep. Windows are cut at a fixed length with fixed
overlap, both from ``configs/pipeline.yaml``; a trailing partial window is dropped
rather than zero-padded, so every timestep the model sees covers a real, complete
span of signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import PipelineConfig


@dataclass
class Window:
    index: int
    start_sample: int
    t_start_ms: float
    t_end_ms: float
    frames: np.ndarray  # (win_len, channels) view onto the unified grid


def make_windows(grid_ms: np.ndarray, frames: np.ndarray,
                 pipeline: PipelineConfig) -> list[Window]:
    """Cut ``frames`` (n, channels) into overlapping fixed-length windows."""
    grid_ms = np.asarray(grid_ms, dtype=np.float64)
    if frames.shape[0] != grid_ms.shape[0]:
        raise ValueError(f"grid/frame length mismatch: {grid_ms.shape[0]} vs {frames.shape[0]}")

    grid_hz = pipeline.target_grid_hz
    win_len = int(round(pipeline.window_s * grid_hz))
    hop = int(round(pipeline.hop_s * grid_hz))
    if win_len <= 0 or hop <= 0:
        raise ValueError(f"invalid windowing: window_s={pipeline.window_s}, overlap={pipeline.overlap}")

    windows: list[Window] = []
    for start in range(0, frames.shape[0] - win_len + 1, hop):
        stop = start + win_len
        windows.append(Window(
            index=len(windows),
            start_sample=start,
            t_start_ms=float(grid_ms[start]),
            t_end_ms=float(grid_ms[stop - 1]),
            frames=frames[start:stop],
        ))
    return windows
