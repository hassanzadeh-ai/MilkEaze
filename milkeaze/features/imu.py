"""IMU features (accel xyz + gyro xyz).

Primarily used to gate out gross movement (baby repositioning, handling) rather than
to feed feeding events directly: motion energy, jerk, tilt, dominant motion frequency.
"""
from __future__ import annotations

import numpy as np


def imu_features(imu_block: np.ndarray, fs: float) -> np.ndarray:
    """``imu_block`` is (n_time, 6): acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z."""
    acc = imu_block[:, :3]
    gyr = imu_block[:, 3:6]

    feats: list[float] = []
    for c in range(imu_block.shape[1]):
        feats.append(float(np.mean(imu_block[:, c])))
        feats.append(float(np.std(imu_block[:, c])))

    acc_mag = np.sqrt(np.sum(acc ** 2, axis=1))
    gyr_mag = np.sqrt(np.sum(gyr ** 2, axis=1))
    feats.append(float(np.mean(acc_mag)))
    feats.append(float(np.mean(gyr_mag)))

    motion_energy = float(np.mean(acc_mag ** 2))
    jerk = float(np.mean(np.abs(np.diff(acc_mag)))) if len(acc_mag) > 1 else 0.0
    feats.append(motion_energy)
    feats.append(jerk)

    # crude tilt from mean accel direction
    mean_acc = np.mean(acc, axis=0)
    tilt = float(np.arctan2(np.linalg.norm(mean_acc[:2]), abs(mean_acc[2]) + 1e-8))
    feats.append(tilt)

    return np.asarray(feats, dtype=np.float32)
