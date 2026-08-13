"""Raw-data layer: everything between files on disk and rectangular model tensors.

The pipeline treats this package as the only place that knows about on-disk reality —
file layouts, ADC counts, per-board clocks, sample-rate jitter. Everything above it
works in physical units on a single uniform time grid.

Two on-disk layouts are supported and auto-detected by :func:`ingestion.load_session`:

``flat``
    The dev/test contract written by ``milkeaze.synthetic.write_session`` —
    ``strain.csv`` / ``imu.csv`` / ``mic.csv`` / ``scale.csv`` + ``meta.json``,
    all sharing one ``t_ms`` column.

``rig``
    The production dual-board capture (e.g. ``new_dataset/20260718/``) — a sensor
    board and a rig board, each with its own device clock, joined to the host
    monotonic clock through per-board sync files. See :mod:`milkeaze.data.rig_session`.
"""
from __future__ import annotations

from .contract import ContractError, validate_session
from .ingestion import (
    RawSession, SessionLayout, detect_layout, load_session, repair_monotonic,
)
from .labels import events_to_window_labels, load_events
from .pressure_events import DetectorConfig, DetectionResult, detect_suck_events, write_events
from .quality import QualityReport, check_saturation, check_timestamps, check_window
from .resampling import make_grid, mic_envelope, resample_linear
from .rig_session import (
    RigCapture, discover_stems, load_pressure, load_temperature, open_capture, pooled_skew_ppm,
)
from .schema import SidecarReport, validate_dataset, validate_sidecars
from .windowing import Window, make_windows

__all__ = [
    "ContractError",
    "DetectionResult",
    "DetectorConfig",
    "QualityReport",
    "RawSession",
    "RigCapture",
    "SessionLayout",
    "SidecarReport",
    "Window",
    "check_saturation",
    "check_timestamps",
    "check_window",
    "detect_layout",
    "detect_suck_events",
    "discover_stems",
    "events_to_window_labels",
    "load_events",
    "load_pressure",
    "load_session",
    "load_temperature",
    "make_grid",
    "make_windows",
    "mic_envelope",
    "open_capture",
    "pooled_skew_ppm",
    "repair_monotonic",
    "resample_linear",
    "validate_dataset",
    "validate_session",
    "validate_sidecars",
    "write_events",
]
