"""Session loading and layout detection.

:func:`load_session` is the single entry point above this module. It sniffs the
directory, dispatches to the matching reader, and returns a :class:`RawSession` in
which every stream shares one millisecond timebase starting at zero — regardless of
whether the source used a shared ``t_ms`` column or two free-running device clocks
stitched to a host monotonic clock.

``RawSession.units`` records, per stream, whether the values are still ADC counts or
already physical. The production rig board converts IMU on-device, so this is the
flag the calibration layer branches on rather than the layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import SensorConfig
from ..utils.logging import get_logger
from .calibration import COUNTS

log = get_logger(__name__)


class SessionLayout(str, Enum):
    FLAT = "flat"   # dev contract: strain.csv / imu.csv / mic.csv / scale.csv + meta.json
    RIG = "rig"     # production dual-board capture: <stem>_sensor_*.csv + <stem>_rig_*.csv


@dataclass
class RawSession:
    session_id: str
    strain_t_ms: np.ndarray
    imu_t_ms: np.ndarray
    mic_t_ms: np.ndarray
    scale_t_ms: np.ndarray
    strain: np.ndarray            # (n, 8)
    imu: np.ndarray               # (n, 6)
    mic: np.ndarray               # (n, n_mic) — 1 if the mic is mono
    scale_g: np.ndarray
    has_scale: bool
    measured_rates_hz: dict[str, float]
    layout: SessionLayout = SessionLayout.FLAT
    units: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def unit_of(self, stream: str) -> str:
        return self.units.get(stream, COUNTS)


#: Backward timestamp steps larger than this are treated as corruption, not jitter.
MAX_BACKSTEP_MS = 25.0


def repair_monotonic(t_ms: np.ndarray, stream: str,
                     max_backstep_ms: float = MAX_BACKSTEP_MS) -> tuple[np.ndarray, dict[str, Any]]:
    """Force a timestamp stream to be non-decreasing, within a bounded tolerance.

    The sensor board timestamps audio per 32-sample block and IMU per sample, and both
    arrive over Wi-Fi, so transport jitter can make one block appear to start before
    the previous one ended. Every backward step observed on the 20260718 batch is of
    that kind: IMU backsteps are under 1.1 ms against a 2.4 ms sample interval, and
    100% of audio backsteps sit exactly on a block boundary.

    Clamping to a running maximum removes the inversion while leaving sample order
    intact; the resulting ties are collapsed during resampling. A backward step larger
    than ``max_backstep_ms`` is not jitter and raises instead, because at that point
    the stream ordering itself is untrustworthy.
    """
    t_ms = np.asarray(t_ms, dtype=np.float64)
    stats: dict[str, Any] = {"n_backsteps": 0, "worst_backstep_ms": 0.0}
    if t_ms.shape[0] < 2:
        return t_ms, stats

    dt = np.diff(t_ms)
    backward = dt < 0
    if not backward.any():
        return t_ms, stats

    worst = float(-dt[backward].min())
    stats["n_backsteps"] = int(backward.sum())
    stats["worst_backstep_ms"] = worst
    if worst > max_backstep_ms:
        raise ValueError(
            f"{stream}: timestamps jump backwards by {worst:.1f} ms "
            f"(limit {max_backstep_ms:.0f} ms); the stream ordering is not trustworthy"
        )

    log.warning(
        "%s: repaired %d non-monotonic timestamp(s), worst %.2f ms backwards",
        stream, stats["n_backsteps"], worst,
    )
    return np.maximum.accumulate(t_ms), stats


def effective_rate_hz(t_ms: np.ndarray) -> float:
    """Median-based sample rate; robust to the odd dropout, unlike n/duration."""
    if t_ms.shape[0] < 2:
        return float("nan")
    dt = float(np.median(np.diff(t_ms)))
    return 1000.0 / dt if dt > 0 else float("nan")


def detect_layout(session_dir: str | Path) -> SessionLayout:
    """Identify the on-disk layout of ``session_dir``."""
    path = Path(session_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"session directory not found: {path}")
    if (path / "strain.csv").exists() and (path / "meta.json").exists():
        return SessionLayout.FLAT
    if any(path.glob("*_session.json")):
        return SessionLayout.RIG
    raise FileNotFoundError(
        f"{path}: no recognised session layout "
        "(expected strain.csv + meta.json, or a *_session.json dual-board capture)"
    )


def _read_csv(path: Path, required: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing required stream file: {path}")
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing column(s) {missing}; found {list(df.columns)}")
    return df


def _load_flat(session_dir: Path, sensors: SensorConfig) -> RawSession:
    import json

    meta = json.loads((session_dir / "meta.json").read_text(encoding="utf-8"))

    strain_names = sensors.strain_channel_names()
    imu_names = sensors.imu_channel_names()
    mic_names = sensors.mic_channel_names()

    strain_df = _read_csv(session_dir / "strain.csv", ["t_ms", *strain_names])
    imu_df = _read_csv(session_dir / "imu.csv", ["t_ms", *imu_names])
    mic_df = _read_csv(session_dir / "mic.csv", ["t_ms", *mic_names])

    scale_col = sensors.raw["scale"]["channel"]
    scale_path = session_dir / "scale.csv"
    if scale_path.exists():
        scale_df = _read_csv(scale_path, ["t_ms", scale_col])
        scale_t = scale_df["t_ms"].to_numpy(dtype=np.float64)
        scale_g = scale_df[scale_col].to_numpy(dtype=np.float32)
        has_scale = True
    else:
        scale_t = np.empty(0, dtype=np.float64)
        scale_g = np.empty(0, dtype=np.float32)
        has_scale = False
        log.warning("session %s: no scale.csv — no volume targets available", session_dir.name)

    strain_t = strain_df["t_ms"].to_numpy(dtype=np.float64)
    imu_t = imu_df["t_ms"].to_numpy(dtype=np.float64)
    mic_t = mic_df["t_ms"].to_numpy(dtype=np.float64)

    return RawSession(
        session_id=str(meta.get("session_id", session_dir.name)),
        strain_t_ms=strain_t,
        imu_t_ms=imu_t,
        mic_t_ms=mic_t,
        scale_t_ms=scale_t,
        strain=strain_df[strain_names].to_numpy(dtype=np.float32),
        imu=imu_df[imu_names].to_numpy(dtype=np.float32),
        mic=mic_df[mic_names].to_numpy(dtype=np.float32),
        scale_g=scale_g,
        has_scale=has_scale,
        measured_rates_hz={
            "strain": effective_rate_hz(strain_t),
            "imu": effective_rate_hz(imu_t),
            "mic": effective_rate_hz(mic_t),
        },
        layout=SessionLayout.FLAT,
        units={"strain": COUNTS, "imu": COUNTS, "mic": COUNTS},
        meta=meta,
    )


def load_session(session_dir: str | Path, sensors: SensorConfig | None = None,
                 stem: str | None = None) -> RawSession:
    """Load one session from disk, auto-detecting its layout.

    ``stem`` selects a single capture when a rig-layout directory holds several
    (as ``new_dataset/20260718/`` does); it is ignored for the flat layout.
    """
    sensors = sensors or SensorConfig.load()
    path = Path(session_dir)
    layout = detect_layout(path)

    if layout is SessionLayout.FLAT:
        return _load_flat(path, sensors)

    from .rig_session import load_rig_session  # deferred: rig reader imports this module

    return load_rig_session(path, sensors, stem=stem)
