"""End-to-end session processing.

    load -> validate -> calibrate -> resample to unified grid -> drift removal
    -> window -> quality -> per-window hand features + CNN frames
    -> (optional) per-window volume target from the scale
    -> (optional) per-window class labels from events.csv

Produces a :class:`ProcessedSession` ready for training or inference. This is the
single place the raw-data reality (rates, calibration, alignment) is turned into the
rectangular per-window tensors the model consumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PipelineConfig, SensorConfig
from .data.calibration import (
    ema_drift_removal, imu_counts_to_physical, mic_pcm_to_pa, strain_counts_to_resistance,
)
from .data.contract import validate_session
from .data.ingestion import RawSession, load_session
from .data.labels import events_to_window_labels, load_events
from .data.quality import check_window
from .data.resampling import make_grid, mic_envelope, resample_linear
from .data.windowing import Window, make_windows
from .features.assemble import window_features
from .utils.logging import get_logger

log = get_logger(__name__)

MILK_DENSITY_G_PER_ML = 1.03  # breast milk ~1.03 g/mL; used to convert scale grams -> mL


@dataclass
class ProcessedSession:
    session_id: str
    frames: np.ndarray            # (seq, 16, T) unified-grid CNN input
    hand: np.ndarray              # (seq, F) hand features
    windows: list[Window]
    labels: np.ndarray | None     # (seq,) class targets or None
    volume: np.ndarray | None     # (seq,) mL/window target or None
    grid_hz: float


def _unified_frames(session: RawSession, sensors: SensorConfig, pipeline: PipelineConfig):
    grid_hz = pipeline.target_grid_hz
    t0 = max(session.strain_t_ms[0], session.imu_t_ms[0], session.mic_t_ms[0])
    t1 = min(session.strain_t_ms[-1], session.imu_t_ms[-1], session.mic_t_ms[-1])
    grid = make_grid(t0, t1, grid_hz)

    strain_phys = strain_counts_to_resistance(session.strain, sensors)
    imu_phys = imu_counts_to_physical(session.imu, sensors)
    mic_pa = mic_pcm_to_pa(session.mic, sensors)

    strain_grid = resample_linear(session.strain_t_ms, strain_phys, grid)      # (n, 8)
    imu_grid = resample_linear(session.imu_t_ms, imu_phys, grid)                # (n, 6)
    mic_env = mic_envelope(session.mic_t_ms, mic_pa, grid, grid_hz)             # (n, 2)

    # drift removal on strain only (slow mechanical baseline)
    drift = pipeline.raw["calibration"]["drift"]
    seed = int(float(drift["baseline_window_s"]) * grid_hz)
    strain_grid = ema_drift_removal(strain_grid, float(drift["alpha"]), seed)

    frames = np.concatenate([strain_grid, mic_env, imu_grid], axis=1)  # (n, 16)
    return grid, frames, mic_pa


def _window_volume_targets(session: RawSession, windows: list[Window]) -> np.ndarray | None:
    if not session.has_scale:
        return None
    vol = np.zeros(len(windows), dtype=np.float32)
    for i, w in enumerate(windows):
        g0 = np.interp(w.t_start_ms, session.scale_t_ms, session.scale_g)
        g1 = np.interp(w.t_end_ms, session.scale_t_ms, session.scale_g)
        grams = max(0.0, float(g1 - g0))
        vol[i] = grams / MILK_DENSITY_G_PER_ML
    return vol


def process_session(session_dir: str | Path,
                    sensors: SensorConfig | None = None,
                    pipeline: PipelineConfig | None = None,
                    strict: bool = True) -> ProcessedSession:
    sensors = sensors or SensorConfig.load()
    pipeline = pipeline or PipelineConfig.load()

    raw = load_session(session_dir, sensors)
    validate_session(raw, sensors)  # fail loud on contract violations

    grid, frames, mic_pa = _unified_frames(raw, sensors, pipeline)
    windows = make_windows(grid, frames, pipeline)
    if not windows:
        raise ValueError(f"session {raw.session_id}: no windows produced (too short?)")

    mic_fs = raw.measured_rates_hz.get("mic", sensors.sample_rates["mic"])
    strain_fs = pipeline.target_grid_hz
    imu_fs = pipeline.target_grid_hz

    frame_stack = np.empty((len(windows), frames.shape[1], windows[0].frames.shape[0]),
                           dtype=np.float32)
    hand_stack = []
    kept: list[Window] = []
    for w in windows:
        q = check_window(w.frames, pipeline)
        if not q.ok and strict:
            log.debug("dropping window %d: %s", w.index, q.reasons)
            continue
        # native-rate mic slice for this window's acoustic features
        m0, m1 = w.t_start_ms, w.t_end_ms
        sel = (raw.mic_t_ms >= m0) & (raw.mic_t_ms <= m1)
        mic_block = mic_pa[sel] if sel.any() else np.zeros((16, mic_pa.shape[1]), np.float32)

        strain_block = w.frames[:, :8]
        imu_block = w.frames[:, 10:16]
        feats = window_features(strain_block, mic_block, imu_block, strain_fs, mic_fs, imu_fs)

        frame_stack[len(kept)] = w.frames.T  # -> (channels, time)
        hand_stack.append(feats)
        kept.append(w)

    frame_stack = frame_stack[: len(kept)]
    hand = np.stack(hand_stack, axis=0) if hand_stack else np.zeros((0,), np.float32)

    volume = _window_volume_targets(raw, kept)

    labels = None
    events = load_events(session_dir)
    if events is not None:
        labels = events_to_window_labels(events, kept, sensors)

    log.info("processed session %s: %d windows kept", raw.session_id, len(kept))
    return ProcessedSession(
        session_id=raw.session_id, frames=frame_stack, hand=hand, windows=kept,
        labels=labels, volume=volume, grid_hz=pipeline.target_grid_hz,
    )
