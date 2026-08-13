"""Fail-loud validation of a loaded session against the data contract.

This is the boundary where a malformed capture is supposed to stop the run. Anything
that would silently produce a *plausible but wrong* tensor raises
:class:`ContractError`; anything merely unusual is logged and allowed through. The
distinction matters because most of these faults are invisible downstream — a session
whose mic stream ends 40 s early still trains, it just trains on the wrong thing.

Rate checks are a warning, not an error, on purpose: the effective strain rate has
already moved from 19 Hz to ~82 Hz between hardware revisions, and refusing to load
a session because the board got faster is not useful.
"""
from __future__ import annotations

import numpy as np

from ..config import SensorConfig
from ..utils.logging import get_logger
from .ingestion import RawSession

log = get_logger(__name__)

# fraction by which a measured rate may differ from the configured nominal before we warn
RATE_TOLERANCE = 0.25
# streams must overlap by at least this long for cross-modal windowing to be meaningful
MIN_OVERLAP_S = 5.0


class ContractError(ValueError):
    """A loaded session violates the data contract in a way that would corrupt training."""


def _check_stream(name: str, t_ms: np.ndarray, values: np.ndarray,
                  expected_channels: int, errors: list[str]) -> None:
    if values.shape[0] == 0:
        errors.append(f"{name}: stream is empty")
        return
    if values.ndim != 2 or values.shape[1] != expected_channels:
        errors.append(
            f"{name}: expected {expected_channels} channels, got shape {values.shape}"
        )
    if t_ms.shape[0] != values.shape[0]:
        errors.append(
            f"{name}: {t_ms.shape[0]} timestamps for {values.shape[0]} samples"
        )
    if t_ms.shape[0] > 1 and np.any(np.diff(t_ms) < 0):
        n_back = int((np.diff(t_ms) < 0).sum())
        errors.append(f"{name}: timestamps go backwards at {n_back} point(s)")
    if not np.isfinite(values).all():
        n_bad = int((~np.isfinite(values)).sum())
        errors.append(f"{name}: {n_bad} non-finite sample(s)")


def validate_session(raw: RawSession, sensors: SensorConfig) -> None:
    """Raise :class:`ContractError` if ``raw`` cannot be safely turned into tensors."""
    errors: list[str] = []

    n_strain = len(sensors.strain_channel_names())
    n_imu = len(sensors.imu_channel_names())
    n_mic = len(sensors.mic_channel_names())

    _check_stream("strain", raw.strain_t_ms, raw.strain, n_strain, errors)
    _check_stream("imu", raw.imu_t_ms, raw.imu, n_imu, errors)
    _check_stream("mic", raw.mic_t_ms, raw.mic, n_mic, errors)

    spans = {
        "strain": raw.strain_t_ms,
        "imu": raw.imu_t_ms,
        "mic": raw.mic_t_ms,
    }
    if all(t.shape[0] >= 2 for t in spans.values()):
        t0 = max(float(t[0]) for t in spans.values())
        t1 = min(float(t[-1]) for t in spans.values())
        overlap_s = (t1 - t0) / 1000.0
        if overlap_s < MIN_OVERLAP_S:
            detail = ", ".join(
                f"{k} [{t[0] / 1000.0:.1f}, {t[-1] / 1000.0:.1f}] s" for k, t in spans.items()
            )
            errors.append(
                f"streams overlap for only {overlap_s:.1f} s "
                f"(need >= {MIN_OVERLAP_S:.0f} s): {detail}"
            )

    if raw.has_scale:
        if raw.scale_t_ms.shape[0] != raw.scale_g.shape[0]:
            errors.append(
                f"scale: {raw.scale_t_ms.shape[0]} timestamps for {raw.scale_g.shape[0]} readings"
            )
        elif raw.scale_g.shape[0] >= 2 and not np.isfinite(raw.scale_g).all():
            errors.append("scale: non-finite readings")

    if errors:
        raise ContractError(
            f"session {raw.session_id} violates the data contract:\n  - "
            + "\n  - ".join(errors)
        )

    for stream, nominal in sensors.sample_rates.items():
        measured = raw.measured_rates_hz.get(stream)
        if measured is None or not np.isfinite(measured):
            continue
        if abs(measured - float(nominal)) > RATE_TOLERANCE * float(nominal):
            log.warning(
                "session %s: %s runs at %.1f Hz, config says %.1f Hz",
                raw.session_id, stream, measured, float(nominal),
            )

    if raw.has_scale and raw.scale_g.shape[0] >= 2:
        backflow = int((np.diff(raw.scale_g) < -0.5).sum())
        if backflow:
            log.warning(
                "session %s: %d scale decrease(s) > 0.5 g — bottle disturbed or re-tared",
                raw.session_id, backflow,
            )
