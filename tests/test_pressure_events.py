import numpy as np
import pytest

from milkeaze.data.pressure_events import DetectorConfig, detect_suck_events, estimate_cycle_rate_cpm


def _pump_trace(duration_s=120.0, cpm=42.0, fs=180.0, harmonic=0.0, noise=0.0, seed=0):
    """Synthetic vacuum trace: suction troughs at `cpm`, optional 2nd harmonic + noise."""
    rng = np.random.default_rng(seed)
    t_s = np.arange(0.0, duration_s, 1.0 / fs)
    phase = 2 * np.pi * (cpm / 60.0) * t_s
    psi = -1.0 * np.cos(phase) - harmonic * np.cos(2 * phase)
    psi = psi - 0.5 + noise * rng.standard_normal(t_s.size)
    return t_s * 1000.0, psi


def test_recovers_set_cycle_rate():
    t_ms, psi = _pump_trace(cpm=42.0)
    result = detect_suck_events(t_ms, psi)
    assert result.n_events == pytest.approx(42 * 2, abs=2)  # 2 minutes at 42 cpm
    assert result.cycle_rate_cpm == pytest.approx(42.0, abs=1.0)
    assert result.cycle_rate_cv < 0.1


def test_harmonic_does_not_double_the_count():
    """A strong second harmonic is what made the naive detector read ~60 cpm at L3-L5."""
    t_ms, psi = _pump_trace(cpm=42.0, harmonic=0.6)
    result = detect_suck_events(t_ms, psi)
    assert result.cycle_rate_cpm == pytest.approx(42.0, abs=1.5)
    assert result.n_events == pytest.approx(84, abs=3)


def test_amplitude_scales_with_suction_depth():
    quiet = detect_suck_events(*_pump_trace(cpm=42.0))
    loud_t, loud_psi = _pump_trace(cpm=42.0)
    loud = detect_suck_events(loud_t, loud_psi * 4.0)
    assert loud.events["amplitude_psi"].median() > 3 * quiet.events["amplitude_psi"].median()


def test_events_are_suck_only_and_sorted():
    result = detect_suck_events(*_pump_trace())
    assert set(result.events["type"]) == {"suck"}
    assert result.events["t_ms"].is_monotonic_increasing
    assert ((result.events["confidence"] >= 0) & (result.events["confidence"] <= 1)).all()


def test_onset_convention_precedes_peak():
    t_ms, psi = _pump_trace()
    peaks = detect_suck_events(t_ms, psi, DetectorConfig(convention="peak"))
    onsets = detect_suck_events(t_ms, psi, DetectorConfig(convention="onset"))
    n = min(len(peaks.events), len(onsets.events))
    assert (onsets.events["t_ms"].to_numpy()[:n] <= peaks.events["t_ms"].to_numpy()[:n]).all()


def test_band_is_configurable_not_pinned_to_42():
    """The pump becomes controllable and infants suckle faster; the band must move."""
    t_ms, psi = _pump_trace(cpm=110.0, duration_s=60.0)
    result = detect_suck_events(t_ms, psi, DetectorConfig(band_cpm=(60.0, 180.0)))
    assert result.cycle_rate_cpm == pytest.approx(110.0, abs=3.0)


def test_rate_outside_band_is_not_reported():
    t_ms, psi = _pump_trace(cpm=42.0)
    rate = estimate_cycle_rate_cpm(psi - psi.mean(), fs=180.0, band_cpm=(90.0, 180.0))
    assert 90.0 <= rate <= 180.0


def test_too_short_a_trace_raises():
    t_ms, psi = _pump_trace(duration_s=3.0)
    with pytest.raises(ValueError, match="need"):
        detect_suck_events(t_ms, psi)


def test_pinned_params_round_trip():
    result = detect_suck_events(*_pump_trace())
    assert result.params["detector"] == "pressure_trough_v1"
    assert result.params["config"]["convention"] == "peak"
    assert result.params["resolved"]["bandpass_hz"][0] < result.params["resolved"]["bandpass_hz"][1]
