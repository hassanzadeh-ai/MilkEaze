import numpy as np
import pandas as pd
import pytest

from milkeaze.config import PipelineConfig, SensorConfig
from milkeaze.data.labels import events_to_window_labels, load_events
from milkeaze.data.windowing import make_windows


def _windows(duration_s=10.0):
    pipeline = PipelineConfig.load()
    grid_hz = pipeline.target_grid_hz
    n = int(duration_s * grid_hz)
    grid = np.arange(n) * (1000.0 / grid_hz)
    return make_windows(grid, np.zeros((n, 16), np.float32), pipeline)


def test_windows_without_events_are_silence():
    sensors = SensorConfig.load()
    windows = _windows()
    events = pd.DataFrame({"t_ms": [], "type": []})
    labels = events_to_window_labels(events, windows, sensors)
    assert set(labels) == {sensors.classes.index("silence")}


def test_event_inside_a_window_labels_it():
    sensors = SensorConfig.load()
    windows = _windows()
    events = pd.DataFrame({"t_ms": [2500.0], "type": ["suck"]})
    labels = events_to_window_labels(events, windows, sensors)
    suck = sensors.classes.index("suck")
    hit = [w.index for w in windows if w.t_start_ms <= 2500.0 <= w.t_end_ms]
    assert all(labels[i] == suck for i in hit)
    assert labels.sum() == suck * len(hit)


def test_majority_class_wins_within_a_window():
    sensors = SensorConfig.load()
    windows = _windows()
    # three sucks and one swallow inside the first window
    events = pd.DataFrame({
        "t_ms": [100.0, 400.0, 800.0, 1200.0],
        "type": ["suck", "suck", "suck", "swallow"],
    })
    labels = events_to_window_labels(events, windows, sensors)
    assert labels[0] == sensors.classes.index("suck")


def test_unknown_event_type_raises():
    sensors = SensorConfig.load()
    events = pd.DataFrame({"t_ms": [100.0], "type": ["hiccup"]})
    with pytest.raises(ValueError, match="hiccup"):
        events_to_window_labels(events, _windows(), sensors)


def test_confidence_filter_drops_weak_events():
    sensors = SensorConfig.load()
    windows = _windows()
    events = pd.DataFrame({"t_ms": [500.0], "type": ["suck"], "confidence": [0.1]})
    labels = events_to_window_labels(events, windows, sensors, min_confidence=0.5)
    assert set(labels) == {sensors.classes.index("silence")}


def test_missing_events_file_is_not_an_error(tmp_path):
    assert load_events(tmp_path) is None


def test_malformed_events_file_raises(tmp_path):
    (tmp_path / "events.csv").write_text("t_ms,kind\n100,suck\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        load_events(tmp_path)


def test_stem_namespaced_events_are_found(tmp_path):
    (tmp_path / "cap1_events.csv").write_text("t_ms,type\n100,suck\n", encoding="utf-8")
    assert load_events(tmp_path, stem="cap1") is not None
    assert load_events(tmp_path, stem="cap2") is None
