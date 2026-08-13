"""Sidecar schema validation for production captures.

Separate from :mod:`milkeaze.data.contract`, which validates *signals*. This module
validates the *metadata* — the JSON sidecars that say what a capture is. The two fail
differently: a bad signal corrupts a session, a bad sidecar silently mislabels an
entire collection run.

That distinction is why this runs as a graded report rather than a single raise. Some
fields are wrong today for reasons we already understand and have agreed with Steve
(the rig board hardcodes ``orientation``; the channel map is being revised), and those
need to stay visible without blocking work. Others — a missing skew, an unresolved
clock, a capture with no vacuum level — make the data unusable and are errors.

``fill_state`` is checked here ahead of the schema freeze. Water level drops during a
run at a vacuum-dependent rate, and a water-backed dome responds roughly twice as
strongly as an air-backed one, so vacuum and fill are confounded in any batch that
does not record fill. Since fill state *is* the phantom's volume state, that confound
lands directly on the transfer function the volume head has to learn. It is a warning
now and becomes an error once the schema is frozen with the field in it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.logging import get_logger
from .rig_session import RIG_BOARD, SENSOR_BOARD, RigCapture, discover_stems, open_capture

log = get_logger(__name__)

#: Bumped whenever the sidecar layout changes in a way ingestion must branch on.
SCHEMA_VERSION = 2

ERROR = "error"
WARNING = "warning"

#: Fields promised for the frozen schema but not yet shipping; warn, don't fail.
PENDING_FIELDS = ("schema_version", "run.fill_state")


@dataclass
class SidecarIssue:
    level: str
    field: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.field}: {self.message}"


@dataclass
class SidecarReport:
    stem: str
    issues: list[SidecarIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[SidecarIssue]:
        return [i for i in self.issues if i.level == ERROR]

    @property
    def warnings(self) -> list[SidecarIssue]:
        return [i for i in self.issues if i.level == WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, level: str, field_name: str, message: str) -> None:
        self.issues.append(SidecarIssue(level, field_name, message))

    def summary(self) -> str:
        return f"{self.stem}: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"


def _get(meta: dict[str, Any], path: str) -> Any:
    node: Any = meta
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _check_alignment(capture: RigCapture, report: SidecarReport) -> None:
    alignment = capture.session_meta.get("alignment")
    if not alignment:
        report.add(ERROR, "alignment", "absent; cross-board timestamps cannot be trusted")
        return

    for board in (SENSOR_BOARD, RIG_BOARD):
        info = alignment.get(board)
        if not info:
            report.add(ERROR, f"alignment.{board}", "missing board alignment block")
            continue
        if info.get("skew_applied", info.get("skew_ppm")) is None:
            report.add(ERROR, f"alignment.{board}.skew_ppm", "no skew estimate")

        confidence = info.get("skew_confidence")
        if confidence != "resolved":
            se = info.get("skew_se_ppm")
            duration = info.get("duration_s")
            detail = f"is {confidence!r}"
            if se is not None and duration is not None:
                detail += f" (+/- {float(se):.1f} ppm over {float(duration):.0f} s)"
            report.add(WARNING, f"alignment.{board}.skew_confidence",
                       f"{detail}; ingestion substitutes the pooled dataset skew")

        jitter = info.get("jitter_ms")
        if jitter is not None and float(jitter) > 5.0:
            report.add(WARNING, f"alignment.{board}.jitter_ms",
                       f"{float(jitter):.1f} ms is large for cross-modal windowing")


def _check_run_block(capture: RigCapture, report: SidecarReport) -> None:
    run = capture.sensor_meta.get("run") or {}

    if run.get("vacuum_level") is None:
        report.add(ERROR, "run.vacuum_level", "absent; the capture's condition is unknown")
    if run.get("cycle_rate_cpm") is None:
        report.add(WARNING, "run.cycle_rate_cpm", "absent; falling back to detected rate")

    if "fill_state" not in run:
        report.add(
            WARNING, "run.fill_state",
            "absent; vacuum and fill state are confounded without it "
            "(required once the schema is frozen)",
        )

    sensor_orientation = run.get("orientation")
    rig_orientation = (capture.rig_meta.get("run") or {}).get("orientation")
    if sensor_orientation and rig_orientation and sensor_orientation != rig_orientation:
        report.add(
            WARNING, "run.orientation",
            f"boards disagree (sensor {sensor_orientation!r}, rig {rig_orientation!r}); "
            "the rig value is hardcoded, so trust the sensor board",
        )

    for key in ("weight_before_g", "weight_after_g"):
        if key in run and run[key] is None:
            report.add(WARNING, f"run.{key}",
                       "null; the continuous scale stream covers it, so drop or populate")


def _check_device_block(capture: RigCapture, report: SidecarReport) -> None:
    device = capture.sensor_meta.get("device") or {}
    active = device.get("active") or {}

    strain_active = bool(active.get("bend") or active.get("stretch"))
    if strain_active and not device.get("strain_ch_mask"):
        report.add(WARNING, "device.strain_ch_mask",
                   "is 0 while strain is active; per-channel enablement is unrecorded")

    if active.get("mic_l") and active.get("mic_r"):
        report.add(WARNING, "device.active.mic_r",
                   "flagged active, but the second MP34DT01-M has a known fault; "
                   "ingestion treats the capture as mono")


def _check_conversions(capture: RigCapture, report: SidecarReport) -> None:
    pressure = _get(capture.rig_meta, "conversion.rig.pressure")
    if not pressure:
        report.add(ERROR, "conversion.rig.pressure",
                   "absent; pressure cannot be converted, so no derived labels")
        return
    for key in ("out_min_counts", "out_max_counts", "p_min_psi", "p_max_psi"):
        if pressure.get(key) is None:
            report.add(ERROR, f"conversion.rig.pressure.{key}", "missing conversion constant")

    if _get(capture.sensor_meta, "conversion.strain") is None:
        report.add(ERROR, "conversion.strain", "absent; strain counts cannot be calibrated")


def validate_sidecars(capture: RigCapture) -> SidecarReport:
    """Grade one capture's sidecars against the expected schema."""
    report = SidecarReport(stem=capture.stem)

    version = capture.sensor_meta.get("schema_version")
    if version is None:
        report.add(WARNING, "schema_version",
                   f"absent; assuming v1. Ingestion needs it to branch on future "
                   f"changes (current: v{SCHEMA_VERSION})")
    elif int(version) > SCHEMA_VERSION:
        report.add(ERROR, "schema_version",
                   f"capture is v{version} but this build understands v{SCHEMA_VERSION}")

    _check_alignment(capture, report)
    _check_run_block(capture, report)
    _check_device_block(capture, report)
    _check_conversions(capture, report)

    for board in (SENSOR_BOARD, RIG_BOARD):
        files = (capture.sensor_meta if board == SENSOR_BOARD else capture.rig_meta).get("files", {})
        for role, name in files.items():
            if role == "bin":
                continue
            if not (capture.root / name).exists():
                report.add(ERROR, f"files.{role}", f"sidecar lists {name}, which is not on disk")

    return report


def validate_dataset(root: str | Path) -> list[SidecarReport]:
    """Validate every capture in a rig-layout directory."""
    root = Path(root)
    stems = discover_stems(root)
    if not stems:
        raise FileNotFoundError(f"{root}: no captures found")
    return [validate_sidecars(open_capture(root, stem)) for stem in stems]
