import numpy as np
import pytest

from milkeaze.eval.baselines import (
    STRAIN_POLARITY, fit_ridge, in_band_power_fraction, mean_predictor_scores,
    most_rhythmic_channel, regression_scores, strain_consensus_events,
    strain_consensus_signal, strain_event_baseline, strain_event_candidates,
    strain_rate_cpm,
)

FS = 200.0
RATE_CPM = 42.0


def _rhythmic(duration_s=60.0, rate_cpm=RATE_CPM, amplitude=1.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * FS)) / FS
    signal = amplitude * np.sin(2 * np.pi * (rate_cpm / 60.0) * t)
    return t * 1000.0, signal + rng.normal(0, 0.02 * amplitude, t.size)


def test_ridge_recovers_a_linear_relationship():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    y = 3.0 + X @ np.array([2.0, -1.0, 0.5, 0.0, 0.0])

    model = fit_ridge(X, y, alpha=1e-6)
    assert regression_scores(model.predict(X), y).r2 > 0.999
    assert model.intercept == pytest.approx(y.mean(), rel=1e-6)


def test_features_with_no_training_variance_get_zero_weight():
    """The zero-filled mic slot produces several such columns on real data."""
    rng = np.random.default_rng(1)
    X = np.column_stack([rng.normal(size=200), np.zeros(200), np.full(200, 7.0)])
    y = X[:, 0] * 2.0

    model = fit_ridge(X, y, alpha=1.0)
    assert np.isfinite(model.coef).all()
    assert model.coef[1] == 0.0 and model.coef[2] == 0.0
    assert model.n_active_features == 1


def test_ridge_shrinks_towards_the_mean_as_alpha_grows():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 3))
    y = X @ np.array([1.0, 1.0, 1.0])
    weak = fit_ridge(X, y, alpha=1e6)
    assert np.abs(weak.coef).max() < 0.05
    assert weak.predict(X) == pytest.approx(np.full(100, y.mean()), abs=0.05)


def test_r2_is_zero_for_the_mean_predictor_and_negative_below_it():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert regression_scores(np.full(4, y.mean()), y).r2 == pytest.approx(0.0)
    assert regression_scores(np.zeros(4), y).r2 < 0.0
    assert mean_predictor_scores(y, y).mae == pytest.approx(np.abs(y - y.mean()).mean())


def test_regression_scores_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        regression_scores(np.zeros(3), np.zeros(4))


def test_channel_selection_prefers_rhythm_over_raw_variance():
    """A loud noisy channel must not win: on real hardware that is an arc channel."""
    rng = np.random.default_rng(3)
    _, rhythmic = _rhythmic(amplitude=1.0)
    loud_noise = rng.normal(0, 500.0, rhythmic.size)

    block = np.column_stack([loud_noise, rhythmic])
    assert most_rhythmic_channel(block, FS) == 1


def test_rate_is_recovered_from_strain_regardless_of_channel_scale():
    _, rhythmic = _rhythmic(amplitude=1e-3)
    block = np.column_stack([rhythmic, np.zeros_like(rhythmic)])
    assert strain_rate_cpm(block, FS) == pytest.approx(RATE_CPM, abs=1.5)


def test_strain_event_baseline_finds_cycles_in_either_polarity():
    t_ms, signal = _rhythmic(duration_s=60.0)
    expected = 60.0 / (60.0 / RATE_CPM)

    for sign in (1.0, -1.0):
        block = np.column_stack([sign * signal, np.zeros_like(signal)])
        result = strain_event_baseline(t_ms, block)
        assert result.cycle_rate_cpm == pytest.approx(RATE_CPM, abs=2.0)
        assert result.n_events == pytest.approx(expected, abs=3)


def test_both_polarities_are_offered_half_a_cycle_apart():
    """The channel map is unverified, so the caller has to see both hypotheses."""
    t_ms, signal = _rhythmic(duration_s=60.0)
    block = np.column_stack([signal, np.zeros_like(signal)])

    candidates = strain_event_candidates(t_ms, block)
    assert len(candidates) == 2

    # same rate either way up, but the events land half a period apart
    period_ms = 60_000.0 / RATE_CPM
    rates = [c.cycle_rate_cpm for c in candidates]
    assert rates[0] == pytest.approx(rates[1], abs=2.0)

    first = candidates[0].events["t_ms"].to_numpy(dtype=float)
    second = candidates[1].events["t_ms"].to_numpy(dtype=float)
    n = min(first.size, second.size) - 1
    offset = np.abs(np.median(first[:n] - second[:n]))
    assert offset == pytest.approx(period_ms / 2, rel=0.35)


def test_the_baseline_returns_the_steadiest_candidate():
    t_ms, signal = _rhythmic(duration_s=60.0)
    block = np.column_stack([signal, np.zeros_like(signal)])

    best = strain_event_baseline(t_ms, block)
    assert best is not None
    candidates = strain_event_candidates(t_ms, block)
    assert best.cycle_rate_cv == pytest.approx(candidates[0].cycle_rate_cv, nan_ok=True)


def test_flat_strain_has_no_cycles_to_find():
    t_ms = np.arange(int(60 * FS)) * (1000.0 / FS)
    block = np.zeros((t_ms.size, 2))
    with pytest.raises(ValueError):
        strain_event_baseline(t_ms, block)


def test_in_band_fraction_is_scale_free():
    """The arc taps run six orders of magnitude hotter than the radials."""
    _, rhythmic = _rhythmic(amplitude=1.0)
    block = np.column_stack([rhythmic, 1e6 * rhythmic])
    fraction = in_band_power_fraction(block, FS)
    assert fraction[0] == pytest.approx(fraction[1], rel=1e-6)


# --- consensus detection -------------------------------------------------------------

CONSENSUS_CHANNELS = ["Radial_1", "Radial_2", "Radial_3", "Radial_4"]


def _ring(amplitudes=(1.0, 1.0, 1.0, 1.0), seed=7):
    """A four-channel ring where suction follows :data:`STRAIN_POLARITY` per channel.

    Suction is modelled as a positive deflection, so each channel carries the cycle
    multiplied by its own polarity — which for ``Radial_2`` means inverted.
    """
    t_ms, signal = _rhythmic(duration_s=60.0, seed=seed)
    columns = [STRAIN_POLARITY[name] * amp * signal
               for name, amp in zip(CONSENSUS_CHANNELS, amplitudes)]
    return t_ms, np.column_stack(columns)


def test_consensus_puts_suction_in_the_trough_despite_an_inverted_channel():
    """Radial_2 runs inverted on this hardware; averaging raw would cancel it out."""
    t_ms, block = _ring()
    signal = strain_consensus_signal(block, FS, CONSENSUS_CHANNELS)

    # every channel contributed rather than cancelling
    assert signal.std() > 0.5
    # suction (a positive deflection on Radial_1) must come out as a trough
    assert np.corrcoef(signal, block[:, 0])[0, 1] < -0.9


def test_consensus_ignores_a_dead_channel_instead_of_averaging_it_in():
    t_ms, block = _ring()
    healthy = strain_consensus_signal(block, FS, CONSENSUS_CHANNELS)

    rng = np.random.default_rng(11)
    with_dead = block.copy()
    with_dead[:, 2] = rng.normal(0, 5.0, block.shape[0])  # noise-only, no rhythm

    degraded = strain_consensus_signal(with_dead, FS, CONSENSUS_CHANNELS)
    assert np.corrcoef(healthy, degraded)[0, 1] > 0.95


def test_consensus_survives_losing_any_single_channel():
    """The overmold has been killing individual bend-sensor channels in the field."""
    t_ms, block = _ring()
    expected = 60.0 / (60.0 / RATE_CPM)

    for drop in range(len(CONSENSUS_CHANNELS)):
        keep = [i for i in range(len(CONSENSUS_CHANNELS)) if i != drop]
        names = [CONSENSUS_CHANNELS[i] for i in keep]
        result = strain_consensus_events(t_ms, block[:, keep], names)
        assert result.n_events == pytest.approx(expected, abs=3)
        assert result.cycle_rate_cpm == pytest.approx(RATE_CPM, abs=2.0)


def test_consensus_events_land_on_the_suction_peak_not_half_a_cycle_out():
    """Scored against the analytic peak times, so no second detector can drift with it."""
    t_ms, block = _ring()
    consensus = strain_consensus_events(t_ms, block, CONSENSUS_CHANNELS)

    period_ms = 60_000.0 / RATE_CPM
    found = consensus.events["t_ms"].to_numpy(dtype=float)
    expected = 0.25 * period_ms + np.arange(found.size) * period_ms

    offset = float(np.median(found - expected))
    assert abs(offset) < 0.1 * period_ms, "events are not on the suction peak"


def test_consensus_refuses_a_block_that_does_not_match_its_channel_names():
    t_ms, block = _ring()
    with pytest.raises(ValueError, match="does not match"):
        strain_consensus_signal(block, FS, CONSENSUS_CHANNELS[:2])


def test_consensus_needs_at_least_one_channel_of_known_polarity():
    t_ms, block = _ring()
    with pytest.raises(ValueError, match="known suction polarity"):
        strain_consensus_signal(block, FS, CONSENSUS_CHANNELS,
                                polarity={name: 0 for name in CONSENSUS_CHANNELS})
