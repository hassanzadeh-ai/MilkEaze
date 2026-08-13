import numpy as np
import pytest

from milkeaze.config import SensorConfig
from milkeaze.data.calibration import (
    PHYSICAL, ema_drift_removal, imu_counts_to_physical, mic_pcm_to_pa,
    strain_counts_to_resistance,
)


def test_bend_counts_convert_to_volts():
    sensors = SensorConfig.load()
    counts = np.zeros((4, 8), dtype=np.float64)
    counts[:, 0] = 2 ** 23  # positive full scale on the first bend channel
    out = strain_counts_to_resistance(counts, sensors)
    assert out[0, 0] == pytest.approx(sensors.raw["strain"]["adc"]["bend"]["v_ref"], rel=1e-6)


def test_stretch_counts_convert_to_ohms():
    sensors = SensorConfig.load()
    stretch = sensors.raw["strain"]["adc"]["stretch"]
    v_ref, r_ref = float(stretch["v_ref"]), float(stretch["r_ref_ohm"])
    # a half-scale divider reading means sensor resistance equals the reference
    counts = np.zeros((2, 8), dtype=np.float64)
    counts[:, 4] = 2 ** 23 * 0.5
    out = strain_counts_to_resistance(counts, sensors)
    assert out[0, 4] == pytest.approx(r_ref, rel=1e-3)
    assert v_ref > 0


def test_zero_stretch_reading_does_not_produce_inf():
    sensors = SensorConfig.load()
    out = strain_counts_to_resistance(np.zeros((3, 8)), sensors)
    assert np.isfinite(out).all()


def test_physical_units_pass_through_untouched():
    sensors = SensorConfig.load()
    imu = np.arange(12, dtype=np.float64).reshape(2, 6)
    assert np.allclose(imu_counts_to_physical(imu, sensors, PHYSICAL), imu)


def test_imu_counts_use_per_axis_scale():
    sensors = SensorConfig.load()
    out = imu_counts_to_physical(np.ones((1, 6)), sensors)
    assert out[0, 0] == pytest.approx(sensors.raw["imu"]["accel_ms2_per_count"])
    assert out[0, 3] == pytest.approx(sensors.raw["imu"]["gyro_dps_per_count"])


def test_mic_pa_factor_matches_the_datasheet_sensitivity():
    """-26 dBFS over a 32768 full scale is the 6.09e-4 Pa/count Steve gave."""
    sensors = SensorConfig.load()
    configured = mic_pcm_to_pa(np.ones((1, 2)), sensors)[0, 0]
    derived = 1.0 / (10.0 ** (-26.0 / 20.0) * 32768.0)
    assert configured == pytest.approx(derived, rel=0.01)


def test_ema_removes_a_constant_baseline():
    n, fs = 4000, 200.0
    t = np.arange(n) / fs
    signal = np.sin(2 * np.pi * 2.0 * t)[:, None]
    out = ema_drift_removal(1000.0 + signal, alpha=0.001, seed_samples=int(2 * fs))
    assert abs(out[-500:].mean()) < 0.05


def test_ema_makes_a_drifting_channel_stationary():
    """A ramp leaves a fixed lag offset, which is harmless; what must go is the growth."""
    n, fs = 8000, 200.0
    t = np.arange(n) / fs
    drift = 5.0 * t[:, None]  # +5 units/s, i.e. 200 units over the record
    out = ema_drift_removal(drift, alpha=0.001, seed_samples=int(2 * fs))
    early, late = out[2000:2500].mean(), out[-500:].mean()
    assert abs(late - early) < 0.05 * np.ptp(drift)


def test_ema_preserves_signal_in_the_suckling_band():
    """Removing drift must not cost amplitude at suckling rates."""
    n, fs = 4000, 200.0
    t = np.arange(n) / fs
    signal = np.sin(2 * np.pi * 2.0 * t)[:, None]
    seed = int(2 * fs)
    with_drift = ema_drift_removal(5.0 * t[:, None] + signal, alpha=0.001, seed_samples=seed)
    without_drift = ema_drift_removal(signal, alpha=0.001, seed_samples=seed)
    assert np.ptp(with_drift[-500:]) == pytest.approx(np.ptp(without_drift[-500:]), rel=0.2)


def test_ema_rejects_out_of_range_alpha():
    with pytest.raises(ValueError, match="alpha"):
        ema_drift_removal(np.zeros((10, 2)), alpha=1.5, seed_samples=2)
