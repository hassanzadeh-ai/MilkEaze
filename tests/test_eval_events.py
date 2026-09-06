import numpy as np
import pytest

from milkeaze.eval.events import match_events, rate_cpm, tolerance_from_period

PERIOD_MS = 60_000.0 / 42.0  # the rig's pump, ~1429 ms


def _cycles(n=20, start=1000.0, period=PERIOD_MS):
    return start + np.arange(n) * period


def test_identical_lists_match_completely():
    t = _cycles()
    m = match_events(t, t, tolerance_ms=100.0)
    assert m.n_matched == t.size
    assert m.precision == m.recall == m.f1 == 1.0
    assert m.count_error == 0
    assert m.bias_ms == pytest.approx(0.0)


def test_a_constant_lag_shows_up_as_bias_not_as_a_miss():
    truth = _cycles()
    late = truth + 40.0
    m = match_events(late, truth, tolerance_ms=100.0)
    assert m.n_matched == truth.size
    assert m.bias_ms == pytest.approx(40.0)
    assert m.mad_ms == pytest.approx(0.0)


def test_events_beyond_the_tolerance_do_not_match():
    truth = _cycles()
    m = match_events(truth + 200.0, truth, tolerance_ms=100.0)
    assert m.n_matched == 0
    assert m.f1 == 0.0
    assert np.isnan(m.bias_ms)


def test_a_doubling_detector_scores_half_precision_not_perfect():
    """One-to-one matching is the point: a duplicate cannot hide behind its twin."""
    truth = _cycles(n=10)
    doubled = np.sort(np.concatenate([truth, truth + 30.0]))
    m = match_events(doubled, truth, tolerance_ms=100.0)
    assert m.recall == pytest.approx(1.0)
    assert m.precision == pytest.approx(0.5)
    assert m.count_error == 10
    assert m.count_error_pct == pytest.approx(100.0)


def test_undercounting_shows_as_recall_loss():
    truth = _cycles(n=10)
    m = match_events(truth[::2], truth, tolerance_ms=100.0)
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(0.5)
    assert m.count_error == -5


def test_nearest_pair_wins_when_two_predictions_compete():
    truth = np.array([1000.0])
    m = match_events(np.array([1010.0, 1080.0]), truth, tolerance_ms=100.0)
    assert m.n_matched == 1
    assert m.bias_ms == pytest.approx(10.0)


def test_unsorted_input_is_handled():
    truth = _cycles(n=6)
    shuffled = truth[[3, 0, 5, 1, 4, 2]]
    assert match_events(shuffled, truth, tolerance_ms=50.0).f1 == 1.0


def test_tolerance_defaults_to_a_quarter_period_with_a_floor():
    assert tolerance_from_period(_cycles()) == pytest.approx(PERIOD_MS / 4, rel=1e-6)
    # a very fast stream must not produce a tolerance finer than the alignment error
    assert tolerance_from_period(np.arange(10) * 4.0, floor_ms=50.0) == 50.0
    assert tolerance_from_period(np.array([5.0])) == 50.0


def test_rate_uses_the_median_period_so_one_gap_does_not_move_it():
    t = _cycles(n=30)
    dropped = np.delete(t, 15)
    assert rate_cpm(t) == pytest.approx(42.0, abs=0.01)
    assert rate_cpm(dropped) == pytest.approx(42.0, abs=0.01)


def test_empty_lists_are_reported_not_crashed():
    m = match_events([], _cycles(), tolerance_ms=100.0)
    assert m.n_matched == 0 and m.precision == 0.0 and m.recall == 0.0
    assert rate_cpm([]) == 0.0


def test_non_positive_tolerance_raises():
    with pytest.raises(ValueError, match="tolerance_ms"):
        match_events(_cycles(), _cycles(), tolerance_ms=0.0)
