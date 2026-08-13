"""Per-event labels and their projection onto windows.

The contract is a per-session ``events.csv`` with one row per discrete event:

    ``t_ms``       event time on the session timebase
    ``type``       one of the non-silence class names in ``configs/sensors.yaml``
    ``amplitude``  optional, event strength in source units
    ``confidence`` optional, 0-1 detector confidence

Only ``t_ms`` and ``type`` are required; the optional columns let weak or partial
cycles be filtered before training rather than dropped at detection time.

``t_ms`` marks **peak intensity** of the event, not its onset — see
:data:`T_MS_CONVENTION`. Windows are 2 s and events are ~1.4 s apart at 42 cpm, so a
window usually contains several events; the window takes the label of the class with
the most events in it, and ``silence`` when it contains none.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import SensorConfig
from ..utils.logging import get_logger
from .windowing import Window

log = get_logger(__name__)

EVENTS_FILENAME = "events.csv"
REQUIRED_COLUMNS = ("t_ms", "type")

#: Documented meaning of ``events.csv:t_ms``. Detectors must agree with this.
T_MS_CONVENTION = "peak"

SILENCE_CLASS = "silence"


def events_path(session_dir: str | Path, stem: str | None = None) -> Path:
    """Where a session's events file lives.

    Flat sessions own their directory and use a bare ``events.csv``. A rig-layout
    directory holds several captures side by side, so those are namespaced by stem.
    """
    session_dir = Path(session_dir)
    if stem:
        candidate = session_dir / f"{stem}_{EVENTS_FILENAME}"
        if candidate.exists():
            return candidate
    return session_dir / EVENTS_FILENAME


def load_events(session_dir: str | Path, stem: str | None = None) -> pd.DataFrame | None:
    """Load a session's events file if present, else ``None``.

    A missing file is a normal state (the classifier simply has no targets for that
    session). A *malformed* file is not, and raises.
    """
    path = events_path(session_dir, stem)
    if not path.exists():
        return None

    events = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"{path}: events.csv missing required column(s) {missing}")
    if events.empty:
        log.warning("%s: events.csv is empty", path)
        return events

    events = events.sort_values("t_ms").reset_index(drop=True)
    if not np.isfinite(events["t_ms"]).all():
        raise ValueError(f"{path}: events.csv contains non-finite t_ms")
    return events


def events_to_window_labels(events: pd.DataFrame, windows: list[Window],
                            sensors: SensorConfig,
                            min_confidence: float = 0.0) -> np.ndarray:
    """Project events onto windows, returning ``(len(windows),)`` class indices."""
    classes = sensors.classes
    if SILENCE_CLASS not in classes:
        raise ValueError(f"class list {classes} must contain '{SILENCE_CLASS}'")
    silence_idx = classes.index(SILENCE_CLASS)
    class_index = {name: i for i, name in enumerate(classes)}

    labels = np.full(len(windows), silence_idx, dtype=np.int64)
    if events.empty or not windows:
        return labels

    if min_confidence > 0.0 and "confidence" in events.columns:
        before = len(events)
        events = events[events["confidence"] >= min_confidence]
        log.info("filtered %d/%d events below confidence %.2f",
                 before - len(events), before, min_confidence)
        if events.empty:
            return labels

    unknown = sorted(set(events["type"]) - set(classes))
    if unknown:
        raise ValueError(
            f"events.csv contains type(s) {unknown} that are not in the configured "
            f"class list {classes}"
        )

    t_events = events["t_ms"].to_numpy(dtype=np.float64)
    type_idx = np.array([class_index[t] for t in events["type"]], dtype=np.int64)

    starts = np.array([w.t_start_ms for w in windows], dtype=np.float64)
    ends = np.array([w.t_end_ms for w in windows], dtype=np.float64)

    for i, (t0, t1) in enumerate(zip(starts, ends)):
        lo, hi = np.searchsorted(t_events, [t0, t1], side="left")
        if hi <= lo:
            continue
        present = type_idx[lo:hi]
        counts = np.bincount(present, minlength=len(classes))
        counts[silence_idx] = 0
        labels[i] = int(counts.argmax())

    n_labelled = int((labels != silence_idx).sum())
    log.info("labelled %d/%d windows from %d events", n_labelled, len(windows), len(events))
    return labels
