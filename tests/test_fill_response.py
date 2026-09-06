import numpy as np
import pytest

from milkeaze.eval.fill_response import (
    fill_response, per_channel_fill_response, split_half_consistency, window_amplitudes,
)


def _frames(n_windows=60, n_channels=16, win_len=400, gains=None, seed=0):
    """A rhythmic frame stack whose per-window amplitude follows ``gains``."""
    rng = np.random.default_rng(seed)
    gains = np.ones(n_windows) if gains is None else np.asarray(gains, dtype=float)
    t = np.arange(win_len) / 200.0
    wave = np.sin(2 * np.pi * 0.7 * t)

    frames = np.empty((n_windows, n_channels, win_len), dtype=np.float32)
    for i in range(n_windows):
        frames[i] = gains[i] * wave + rng.normal(0, 1e-3, (n_channels, win_len))
    return frames


def test_amplitude_is_measured_about_each_window_mean():
    frames = _frames(n_windows=4)
    offset = frames + 1000.0  # a baseline the drift filter would have left behind
    assert window_amplitudes(frames) == pytest.approx(window_amplitudes(offset), rel=1e-3)

    rms = window_amplitudes(_frames(n_windows=2, gains=[1.0, 1.0]))
    assert rms == pytest.approx(1 / np.sqrt(2), rel=0.02)  # RMS of a unit sine


def test_amplitude_tracks_the_gain_it_was_built_with():
    amps = window_amplitudes(_frames(n_windows=10, gains=np.linspace(1.0, 0.5, 10)))
    assert amps[:, 0] == pytest.approx(np.linspace(1.0, 0.5, 10) / np.sqrt(2), rel=0.05)


def test_a_gain_that_halves_from_full_to_empty_reads_as_minus_fifty_percent():
    """Steve's water-vs-air prediction, stated as a number this can confirm or refute."""
    n = 60
    fill = np.linspace(1.0, 0.0, n)
    amps = window_amplitudes(_frames(n_windows=n, gains=1.0 - 0.5 * (1.0 - fill)))

    response = fill_response(amps[:, 0], fill)
    assert response.pct_change_full_to_empty == pytest.approx(-50.0, abs=3.0)
    assert response.rank_correlation > 0.99
    assert response.slope_per_fraction > 0
    assert response.amplitude_full > response.amplitude_empty


def test_a_stationary_gain_shows_no_response():
    n = 40
    fill = np.linspace(1.0, 0.0, n)
    amps = window_amplitudes(_frames(n_windows=n))
    response = fill_response(amps[:, 0], fill)
    assert abs(response.pct_change_full_to_empty) < 5.0
    assert abs(response.rank_correlation) < 0.5


def test_a_constant_fill_axis_cannot_be_fitted():
    response = fill_response(np.arange(10.0), np.ones(10))
    assert np.isnan(response.slope_per_fraction)
    assert response.intercept == pytest.approx(np.arange(10.0).mean())


def test_split_half_agrees_for_a_real_effect_and_disagrees_for_a_step():
    n = 60
    fill = np.linspace(1.0, 0.0, n)

    smooth = window_amplitudes(_frames(n_windows=n, gains=1.0 - 0.5 * (1.0 - fill)))
    first, second = split_half_consistency(smooth[:, 0], fill)
    assert first.rank_correlation > 0.9 and second.rank_correlation > 0.9

    # a one-off drop midway is not a gain-versus-fill relationship
    step = np.ones(n)
    step[n // 2:] = 0.5
    stepped = window_amplitudes(_frames(n_windows=n, gains=step))
    a, b = split_half_consistency(stepped[:, 0], fill)
    assert abs(a.rank_correlation) < 0.5 and abs(b.rank_correlation) < 0.5


def test_per_channel_returns_one_fit_for_each_channel():
    amps = window_amplitudes(_frames(n_windows=20))
    fits = per_channel_fill_response(amps, np.linspace(1.0, 0.0, 20))
    assert len(fits) == amps.shape[1]


def test_too_few_windows_to_fit_raises():
    with pytest.raises(ValueError, match="at least 3 windows"):
        fill_response([1.0, 2.0], [1.0, 0.0])
