"""Assemble the per-window hand-feature vector from strain + acoustic + IMU.

The full per-window representation fed to the LSTM is this hand-feature vector fused
with the CNN embedding (concatenated) — see models/milkeaze_net.py.
"""
from __future__ import annotations

import numpy as np

from .acoustic import acoustic_features
from .imu import imu_features
from .strain import strain_features


def window_features(strain_block: np.ndarray, mic_block: np.ndarray, imu_block: np.ndarray,
                    strain_fs: float, mic_fs: float, imu_fs: float) -> np.ndarray:
    """Return the concatenated hand-feature vector for one window."""
    parts = [
        strain_features(strain_block, strain_fs),
        acoustic_features(mic_block, mic_fs),
        imu_features(imu_block, imu_fs),
    ]
    return np.concatenate(parts).astype(np.float32)


def _infer_feature_dim() -> int:
    s = np.zeros((40, 8), dtype=np.float32)
    m = np.zeros((1600, 2), dtype=np.float32)
    i = np.zeros((40, 6), dtype=np.float32)
    return int(window_features(s, m, i, 19.0, 8000.0, 416.0).shape[0])


FEATURE_DIM: int = _infer_feature_dim()
