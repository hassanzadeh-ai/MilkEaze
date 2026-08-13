"""ADC counts to physical units, plus mechanical drift removal.

Raw counts are the ground truth on the wire; every conversion factor lives in
``configs/sensors.yaml`` and mirrors the per-session sidecar. Two streams can arrive
*already* converted — the production rig board emits IMU in m/s² and dps — so each
converter takes a ``units`` argument and is a no-op when handed physical values.
That keeps a single call site in the pipeline instead of branching per layout.

A note on strain units: the two strain families are electrically different and do not
share a unit. Bend (radial) channels are a differential voltage against the ADC's
internal reference; stretch (arc) channels sit in a divider and convert to ohms. Both
end up in one array because the model consumes them as one block — after drift removal
and per-channel standardisation the absolute unit carries no information — but the
mixed units matter if you ever read the values directly.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from ..config import SensorConfig
from ..utils.logging import get_logger

log = get_logger(__name__)

COUNTS = "counts"
PHYSICAL = "physical"

# 94 dB SPL (1 Pa) at -26 dBFS over a 32768 full scale -> 1/(10**(-26/20) * 32768).
_MIC_FS_COUNTS = 32768.0


def _adc_volts(counts: np.ndarray, bits: int, v_ref: float) -> np.ndarray:
    """Signed ADC counts to volts. Counts are two's-complement, so full scale is 2^(bits-1)."""
    return counts / float(2 ** (bits - 1)) * v_ref


def strain_counts_to_resistance(strain: np.ndarray, sensors: SensorConfig,
                                units: str = COUNTS) -> np.ndarray:
    """Convert the 8 strain channels to physical units (bend: volts, stretch: ohms)."""
    strain = np.asarray(strain, dtype=np.float64)
    if units == PHYSICAL:
        return strain.astype(np.float32)

    cfg = sensors.raw["strain"]
    adc = cfg["adc"]
    bits = int(adc["bits"])
    n_bend = len(cfg["bend"])

    out = np.empty_like(strain, dtype=np.float64)

    bend = adc["bend"]
    out[:, :n_bend] = _adc_volts(strain[:, :n_bend], bits, float(bend["v_ref"]))

    stretch = adc["stretch"]
    v_ref = float(stretch["v_ref"])
    r_ref = float(stretch["r_ref_ohm"])
    v = _adc_volts(strain[:, n_bend:], bits, v_ref)
    # R_sensor = R_ref * (Vref - V) / V; V -> 0 is an open circuit, not an infinite resistance
    floor = v_ref * 1e-6
    safe_v = np.where(np.abs(v) < floor, np.where(v < 0.0, -floor, floor), v)
    out[:, n_bend:] = r_ref * (v_ref - safe_v) / safe_v

    return out.astype(np.float32)


def imu_counts_to_physical(imu: np.ndarray, sensors: SensorConfig,
                           units: str = COUNTS) -> np.ndarray:
    """Convert the 6 IMU channels to m/s² (accel) and deg/s (gyro)."""
    imu = np.asarray(imu, dtype=np.float64)
    if units == PHYSICAL:
        return imu.astype(np.float32)

    cfg = sensors.raw["imu"]
    n_accel = len(cfg["accel_channels"])
    scale = np.ones(imu.shape[1], dtype=np.float64)
    scale[:n_accel] = float(cfg["accel_ms2_per_count"])
    scale[n_accel:] = float(cfg["gyro_dps_per_count"])
    return (imu * scale).astype(np.float32)


def mic_pcm_to_pa(mic: np.ndarray, sensors: SensorConfig,
                  units: str = COUNTS) -> np.ndarray:
    """Convert mic PCM counts to pascals."""
    mic = np.asarray(mic, dtype=np.float64)
    if units == PHYSICAL:
        return mic.astype(np.float32)

    cfg = sensors.raw["mic"]
    pa_per_count = cfg.get("pa_per_count")
    if pa_per_count is None:
        sensitivity_dbfs = float(cfg["sensitivity_dbfs"])
        pa_per_count = 1.0 / (10.0 ** (sensitivity_dbfs / 20.0) * _MIC_FS_COUNTS)
        log.debug("mic pa_per_count derived from sensitivity: %.4g", pa_per_count)
    return (mic * float(pa_per_count)).astype(np.float32)


def ema_drift_removal(x: np.ndarray, alpha: float, seed_samples: int) -> np.ndarray:
    """Subtract an exponential-moving-average baseline from each channel.

    The strain baseline wanders with mount creep and temperature over tens of seconds,
    which the model must not read as signal. ``alpha`` is per-sample on the unified
    grid; the baseline is seeded from the mean of the first ``seed_samples`` so the
    output doesn't start with a large transient while the EMA converges.
    """
    x = np.asarray(x, dtype=np.float64)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if x.shape[0] == 0:
        return x.astype(np.float32)

    seed_samples = max(1, min(int(seed_samples), x.shape[0]))
    seed = x[:seed_samples].mean(axis=0)

    # y[n] = alpha*x[n] + (1-alpha)*y[n-1], as a first-order IIR so a 100k-sample
    # session doesn't run a Python loop. zi holds (1-alpha)*y[-1] for this form.
    b = np.array([alpha], dtype=np.float64)
    a = np.array([1.0, -(1.0 - alpha)], dtype=np.float64)
    zi = np.atleast_2d((1.0 - alpha) * seed)
    baseline, _ = lfilter(b, a, x, axis=0, zi=zi)
    return (x - baseline).astype(np.float32)
