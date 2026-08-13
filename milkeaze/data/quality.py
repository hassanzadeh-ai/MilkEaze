"""Signal-quality checks.

Three checkpoints, deliberately at different stages, because each fault is only
visible in one domain:

``check_saturation``
    Runs on raw ADC counts at ingestion. ``saturation_adc_low/high`` are counts, so
    this check is meaningless once calibration has converted the stream to ohms,
    pascals and m/s² — a rail-to-rail strain channel and a legitimate 9.8 m/s²
    gravity reading are indistinguishable at that point.

``check_timestamps``
    Runs on the *native* streams, before resampling. Interpolation happily bridges a
    half-second dropout with a smooth ramp, so a gap that isn't caught here becomes
    invisible, plausible-looking data downstream.

``check_window``
    Runs per window, after resampling, on physical units. Catches NaNs, non-finite
    values and dead (flatlined) channels in exactly the tensor the model consumes.

All three return a :class:`QualityReport` rather than raising: the caller decides
whether ``reject_on_fail`` means drop the window or abort the session. ``ok`` already
folds in that config flag, so callers can branch on it directly.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import PipelineConfig
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class QualityReport:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def _finalize(reasons: list[str], stats: dict[str, Any],
              pipeline: PipelineConfig) -> QualityReport:
    reject = bool(pipeline.raw["quality"].get("reject_on_fail", True))
    return QualityReport(ok=(not reasons) or (not reject), reasons=reasons, stats=stats)


def check_timestamps(t_ms: np.ndarray, pipeline: PipelineConfig,
                     stream: str = "unknown") -> QualityReport:
    """Check a native stream's timestamps for dropouts and ordering faults."""
    t_ms = np.asarray(t_ms, dtype=np.float64)
    max_dropout_ms = float(pipeline.raw["quality"]["max_dropout_ms"])

    reasons: list[str] = []
    stats: dict[str, Any] = {"stream": stream, "n_samples": int(t_ms.shape[0])}

    if t_ms.shape[0] < 2:
        reasons.append(f"{stream}: fewer than 2 samples")
        return _finalize(reasons, stats, pipeline)

    dt = np.diff(t_ms)
    median_dt = float(np.median(dt))
    stats["median_dt_ms"] = median_dt
    stats["max_dt_ms"] = float(dt.max())
    stats["effective_hz"] = 1000.0 / median_dt if median_dt > 0 else float("nan")

    n_back = int((dt <= 0).sum())
    if n_back:
        stats["n_non_monotonic"] = n_back
        reasons.append(f"{stream}: {n_back} non-monotonic timestamps")

    gaps = dt > max_dropout_ms
    n_gaps = int(gaps.sum())
    stats["n_dropouts"] = n_gaps
    if n_gaps:
        lost_ms = float(dt[gaps].sum())
        stats["dropout_total_ms"] = lost_ms
        reasons.append(
            f"{stream}: {n_gaps} dropouts > {max_dropout_ms:.0f} ms "
            f"(worst {dt.max():.0f} ms, {lost_ms / 1000.0:.1f} s total)"
        )
        log.warning("%s: %d timestamp dropouts, worst %.0f ms", stream, n_gaps, dt.max())

    return _finalize(reasons, stats, pipeline)


def check_saturation(counts: np.ndarray, pipeline: PipelineConfig,
                     stream: str = "unknown") -> QualityReport:
    """Check a raw ADC stream (n, channels) for rail-to-rail saturation.

    Rails are per stream: the strain ADC is 24-bit signed while the mic is 16-bit PCM,
    so a single global pair of thresholds flags one of them on every sample.
    """
    counts = np.asarray(counts, dtype=np.float64)
    q = pipeline.raw["quality"]
    max_fraction = float(q["max_nan_fraction"])

    reasons: list[str] = []
    stats: dict[str, Any] = {"stream": stream}

    rails = q.get("saturation", {}).get(stream)
    if rails is None:
        log.debug("no saturation rails configured for stream '%s'; skipping", stream)
        return _finalize(reasons, stats, pipeline)
    sat_low = float(rails["low"])
    sat_high = float(rails["high"])

    if counts.size == 0:
        reasons.append(f"{stream}: empty stream")
        return _finalize(reasons, stats, pipeline)

    if counts.ndim == 1:
        counts = counts[:, None]

    railed = (counts <= sat_low) | (counts >= sat_high)
    per_channel = railed.mean(axis=0)
    stats["saturated_fraction_per_channel"] = [float(x) for x in per_channel]
    bad = np.flatnonzero(per_channel > max_fraction)
    if bad.size:
        worst = ", ".join(f"ch{int(c)}={per_channel[c]:.1%}" for c in bad)
        reasons.append(
            f"{stream}: {bad.size} channel(s) saturated outside [{sat_low:.0f}, {sat_high:.0f}] ({worst})"
        )
        log.warning("%s: saturated channels %s", stream, bad.tolist())

    return _finalize(reasons, stats, pipeline)


def check_window(frames: np.ndarray, pipeline: PipelineConfig,
                 ignore_channels: Sequence[int] | None = None) -> QualityReport:
    """Check one resampled window (win_len, channels) in physical units.

    ``ignore_channels`` exempts channels that are known-inactive by configuration —
    a zero-filled slot held open for faulty hardware is flat by design and must not
    be read as a dead channel.
    """
    max_nan_fraction = float(pipeline.raw["quality"]["max_nan_fraction"])

    reasons: list[str] = []
    stats: dict[str, Any] = {}

    if frames.size == 0:
        return _finalize(["empty window"], stats, pipeline)

    live = np.ones(frames.shape[1], dtype=bool)
    if ignore_channels is not None:
        live[list(ignore_channels)] = False
    if not live.any():
        return _finalize(["every channel is marked inactive"], stats, pipeline)
    live_frames = frames[:, live]

    nan_fraction = float((~np.isfinite(live_frames)).mean())
    stats["nan_fraction"] = nan_fraction
    if nan_fraction > max_nan_fraction:
        reasons.append(f"non-finite fraction {nan_fraction:.3f} > {max_nan_fraction:.3f}")

    # a channel with zero variance across a whole 2 s window is disconnected, not quiet
    finite_cols = np.isfinite(live_frames).all(axis=0)
    if finite_cols.any():
        spread = np.ptp(live_frames[:, finite_cols], axis=0)
        n_dead = int((spread == 0).sum())
        stats["n_flatlined_channels"] = n_dead
        if n_dead:
            reasons.append(f"{n_dead} flatlined channel(s)")

    return _finalize(reasons, stats, pipeline)
