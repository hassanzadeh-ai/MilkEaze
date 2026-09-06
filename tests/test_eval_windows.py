import numpy as np
import pytest

from milkeaze.config import PipelineConfig
from milkeaze.data.windowing import make_windows
from milkeaze.eval.windows import (
    class_balance, label_granularity, majority_class_rate, skill_score,
)

PERIOD_MS = 60_000.0 / 42.0


def _windows(duration_s=60.0):
    pipeline = PipelineConfig.load()
    grid_hz = pipeline.target_grid_hz
    n = int(duration_s * grid_hz)
    grid = np.arange(n) * (1000.0 / grid_hz)
    return make_windows(grid, np.zeros((n, 16), np.float32), pipeline)


def test_two_second_windows_cannot_resolve_events_at_42_cpm():
    """The trap this module exists to expose: 2 s windows, 1.43 s cycles."""
    windows = _windows()
    events = np.arange(0.0, 60_000.0, PERIOD_MS)
    gran = label_granularity(events, windows)

    assert gran.events_per_window_mean > 1.0
    assert gran.median_period_ms == pytest.approx(PERIOD_MS, rel=1e-6)
    assert gran.max_window_s_for_single_event == pytest.approx(PERIOD_MS / 1000.0, rel=1e-6)
    assert not gran.resolves_single_events
    assert gran.occupancy > 0.95


def test_sparse_events_do_resolve_and_leave_windows_empty():
    windows = _windows()
    events = np.arange(0.0, 60_000.0, 6_000.0)  # one every 6 s, well under a 2 s window
    gran = label_granularity(events, windows)

    assert gran.events_per_window_max == 1
    assert gran.resolves_single_events
    assert gran.occupancy < 0.5


def test_majority_class_rate_is_the_floor_any_classifier_must_clear():
    labels = np.array([1] * 99 + [0])
    assert majority_class_rate(labels) == pytest.approx(0.99)


def test_ignore_index_is_excluded_from_the_floor():
    labels = np.array([1, 1, 0, -100, -100])
    assert majority_class_rate(labels) == pytest.approx(2 / 3)
    assert np.isnan(majority_class_rate(np.array([-100, -100])))


def test_skill_score_calls_99_percent_against_a_99_percent_floor_zero():
    assert skill_score(0.99, 0.99) == pytest.approx(0.0)
    assert skill_score(0.995, 0.99) == pytest.approx(0.5)
    assert skill_score(1.0, 0.99) == pytest.approx(1.0)
    assert skill_score(0.98, 0.99) < 0.0
    assert np.isnan(skill_score(1.0, 1.0))


def test_class_balance_reports_shares_over_labelled_windows_only():
    balance = class_balance(np.array([0, 1, 1, 2, -100]), num_classes=3)
    assert balance == pytest.approx([0.25, 0.5, 0.25])
    assert class_balance(np.array([-100]), num_classes=3).sum() == 0.0
