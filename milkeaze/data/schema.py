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

Fill state is checked here because water level drops during a run at a vacuum-dependent
rate, and a water-backed dome responds roughly twice as strongly as an air-backed one,
so vacuum and fill are confounded in any batch that does not record fill. Schema 2 ships
the field as session-level ``fill``; the older ``run.fill_state`` spelling is still
accepted so v1 captures grade the same way.

**The clock check is the one that earns its keep.** Every other field here is either
present or it is not, and a human reading the sidecar would notice. The alignment
residual is different: the sidecar reports its own ``jitter_ms``, computed by the
capture tool under its own outlier handling, and on the Wi-Fi sensor board that number
reads several times lower than the scatter a consumer actually gets when it fits the
sync pairs. A capture can therefore advertise a clean clock and still misplace
cross-board windows. So this module recomputes the residual from the sync file and
grades *that*, and separately flags captures where the sidecar's own figure is
optimistic by more than a factor of three.
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
#: Pass ``frozen=True`` to promote them to errors, which is the gate the collection runs
#: behind once Steve's tooling emits them. Keeping the list here rather than in the CLI
#: means the freeze is one edit, and the same list documents what the freeze is waiting on.
PENDING_FIELDS = ("schema", "fill", "run.outlet")

#: Clock-residual thresholds, in ms. The warn level is set at the precision of the event
#: labels themselves: the pressure detector's trough timestamps fit a metronome to
#: 3.4-17.1 ms RMS, so an alignment residual past ~10 ms stops being negligible against
#: the labels it is being used to position. The error level is a third of the detector's
#: 250 ms refractory window, past which a window can pick up the neighbouring cycle.
RESIDUAL_WARN_MS = 10.0
RESIDUAL_ERROR_MS = 80.0

#: How far the sidecar's own jitter figure may understate the recomputed residual.
JITTER_OPTIMISM_FACTOR = 3.0

#: ``stream_coverage`` is reported per stream as span/missing/coverage. The coverage
#: number is not a fraction — it runs slightly above 1.0 on every capture in the
#: 20260816 batch — so it is graded as a deviation from unity, not as a floor.
COVERAGE_TOLERANCE = 0.05

#: The event detector's timestamp anchor. Changing it shifts every label, and the choice
#: was made by measurement (trough offsets are level-invariant; half-depth crossings walk
#: 321-412 ms across L1-L5), so a capture using a different anchor is not poolable.
EXPECTED_TIMESTAMP_CONVENTION = "trough_subsample"


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

        # The check the sidecar cannot do for itself: how well the linear clock model
        # actually fits this capture's sync pairs, recomputed rather than reported.
        clock = capture.clocks.get(board)
        if clock is None:
            continue
        residual = float(clock.residual_ms)
        if residual > RESIDUAL_ERROR_MS:
            report.add(ERROR, f"alignment.{board}.residual_ms",
                       f"{residual:.1f} ms RMS misfit to the sync pairs exceeds "
                       f"{RESIDUAL_ERROR_MS:.0f} ms; cross-board windows may straddle "
                       "the neighbouring suck cycle")
        elif residual > RESIDUAL_WARN_MS:
            report.add(WARNING, f"alignment.{board}.residual_ms",
                       f"{residual:.1f} ms RMS misfit to the sync pairs is comparable to "
                       "the event-label precision itself")

        if jitter is not None and float(jitter) > 0:
            ratio = residual / float(jitter)
            if ratio > JITTER_OPTIMISM_FACTOR:
                report.add(WARNING, f"alignment.{board}.jitter_ms",
                           f"sidecar reports {float(jitter):.2f} ms but the sync pairs fit "
                           f"to {residual:.2f} ms ({ratio:.1f}x worse); anything trusting "
                           "the sidecar figure is under-estimating its window error")

        if clock.wraps:
            report.add(WARNING, f"alignment.{board}.device_ts_us",
                       f"the 32-bit microsecond counter rolled over {clock.wraps} time(s) "
                       "mid-run; ingestion unwraps it, but any consumer that does not "
                       "will read this board's time as running backwards")

        if clock.n_sync_points < 4:
            report.add(WARNING, f"alignment.{board}.n_sync_points",
                       f"only {clock.n_sync_points} sync pairs; the skew is barely constrained")


def _check_run_block(capture: RigCapture, report: SidecarReport) -> None:
    run = capture.sensor_meta.get("run") or {}

    if run.get("vacuum_level") is None:
        report.add(ERROR, "run.vacuum_level", "absent; the capture's condition is unknown")
    if run.get("cycle_rate_cpm") is None:
        report.add(WARNING, "run.cycle_rate_cpm", "absent; falling back to detected rate")

    # Schema 2 moved this to the session block; accept either spelling.
    fill = capture.session_meta.get("fill", run.get("fill_state"))
    if fill is None:
        report.add(
            WARNING, "fill",
            "absent; vacuum and fill state are confounded without it "
            "(required once the schema is frozen)",
        )

    # Once the outlet is actuated rather than passive it becomes an independent factor,
    # and a batch that varies it without recording it is confounded the same way this
    # one was confounded by fill.
    if "outlet" not in run:
        report.add(
            WARNING, "run.outlet",
            "absent; with a controllable outlet, vacuum no longer determines flow, so "
            "the outlet setting has to be recorded to keep the two separable",
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


def _check_streams(capture: RigCapture, report: SidecarReport) -> None:
    """Grade ``stream_coverage``, which is per-stream and is not a fraction."""
    coverage = capture.session_meta.get("stream_coverage")
    if not isinstance(coverage, dict) or not coverage:
        report.add(WARNING, "stream_coverage",
                   "absent; dropped samples would not be visible")
        return

    for stream, info in coverage.items():
        if not isinstance(info, dict):
            continue
        missing = info.get("missing_s")
        if missing is not None and float(missing) > 0:
            span = float(info.get("span_s") or 0.0)
            frac = 100.0 * float(missing) / span if span else float("nan")
            level = ERROR if frac > 5.0 else WARNING
            report.add(level, f"stream_coverage.{stream}.missing_s",
                       f"{float(missing):.1f} s absent of {span:.1f} s ({frac:.1f}%)")

        value = info.get("coverage")
        if value is not None and abs(float(value) - 1.0) > COVERAGE_TOLERANCE:
            report.add(WARNING, f"stream_coverage.{stream}.coverage",
                       f"{float(value):.3f} departs from unity by more than "
                       f"{COVERAGE_TOLERANCE:.0%}; the stream is not the length the "
                       "nominal rate implies")

    if capture.session_meta.get("interrupted"):
        # True on 29 of the 34 captures in the 20260816 batch, where runs are stopped by
        # the operator once the phantom empties. So it cannot be used as a quality gate
        # until it distinguishes "operator stopped it" from "the link dropped".
        report.add(WARNING, "interrupted",
                   "true; on the 20260816 batch this is the normal case (29/34) rather "
                   "than a fault, so it is not usable as a rejection criterion as defined")


def _check_events(capture: RigCapture, report: SidecarReport, version: int) -> None:
    """Grade the derived-label block, which is what the volume head trains against.

    Derived labels arrived with Schema 2. On a v1 capture their absence is a fact about
    the batch rather than a defect, so it only becomes an error once the capture claims
    to be Schema 2 or later.
    """
    required = ERROR if version >= 2 else WARNING
    events = capture.session_meta.get("events")
    if not isinstance(events, dict) or not events:
        report.add(required, "events",
                   "absent; the capture carries no derived suck labels"
                   + ("" if version >= 2 else " (expected on a v1 capture)"))
        return

    for key in ("detector", "detector_version"):
        if not events.get(key):
            report.add(required, f"events.{key}",
                       "missing; labels cannot be attributed to a detector build")

    convention = events.get("timestamp_convention")
    if convention is None:
        report.add(required, "events.timestamp_convention",
                   "missing; the label anchor is unknown and captures are not poolable")
    elif convention != EXPECTED_TIMESTAMP_CONVENTION:
        report.add(ERROR, "events.timestamp_convention",
                   f"{convention!r} where this build expects "
                   f"{EXPECTED_TIMESTAMP_CONVENTION!r}; anchors differ by hundreds of ms "
                   "across vacuum levels, so mixing them shifts labels systematically")

    measured = events.get("measured") or {}
    n_events = measured.get("n_events")
    if n_events is None:
        report.add(required, "events.measured.n_events", "missing")
        return
    if int(n_events) == 0:
        report.add(required, "events.measured.n_events", "zero; no labels in this capture")
        return

    flagged = measured.get("n_flagged_low_amplitude") or 0
    if int(flagged) > 0:
        frac = 100.0 * int(flagged) / int(n_events)
        level = ERROR if frac > 20.0 else WARNING
        report.add(level, "events.measured.n_flagged_low_amplitude",
                   f"{int(flagged)} of {int(n_events)} events ({frac:.1f}%) below the "
                   "amplitude floor")

    depth = measured.get("depth_psi_median")
    floor = measured.get("depth_floor_psi")
    if depth is not None and floor is not None and float(floor) > 0:
        margin = float(depth) / float(floor)
        if margin < 3.0:
            report.add(WARNING, "events.measured.depth_psi_median",
                       f"{float(depth):.3f} psi is only {margin:.1f}x the detector floor "
                       f"({float(floor):.3f} psi); depth is weakly resolved here")

    empty = measured.get("n_empty") or 0
    if int(empty) > 0:
        report.add(WARNING, "events.measured.n_empty",
                   f"{int(empty)} event(s) after the phantom emptied; these carry no "
                   "transferable mass and must be excluded from volume targets")


def _check_scale(capture: RigCapture, report: SidecarReport) -> None:
    """The scale is the only ground truth for volume, so its faults are errors."""
    scale = capture.sensor_meta.get("scale")
    if not isinstance(scale, dict) or not scale:
        report.add(WARNING, "scale", "absent; no volume ground truth for this capture")
        return

    if scale.get("error"):
        report.add(ERROR, "scale.error", f"{scale['error']!r}")

    errors = scale.get("parse_errors")
    lines = scale.get("lines") or 0
    if errors is not None and int(errors) > 0:
        frac = 100.0 * int(errors) / int(lines) if lines else float("nan")
        level = ERROR if frac > 1.0 else WARNING
        report.add(level, "scale.parse_errors",
                   f"{int(errors)} of {int(lines)} lines ({frac:.1f}%) unparsed")

    if not scale.get("ground_truth"):
        report.add(WARNING, "scale.ground_truth",
                   "not flagged as ground truth; volume targets from this capture are "
                   "not trustworthy")


def _freeze(report: SidecarReport) -> SidecarReport:
    """Promote the promised-but-missing fields from warnings to errors."""
    for issue in report.issues:
        if issue.level == WARNING and issue.field in PENDING_FIELDS:
            issue.level = ERROR
            issue.message += " [required by the frozen schema]"
    return report


def validate_sidecars(capture: RigCapture, frozen: bool = False) -> SidecarReport:
    """Grade one capture's sidecars against the expected schema.

    ``frozen`` is the post-freeze gate: the fields in :data:`PENDING_FIELDS` stop being
    known gaps and start being failures.
    """
    report = SidecarReport(stem=capture.stem)

    # Schema 2 carries the version as session-level ``schema``; v1 captures spelled it
    # ``schema_version`` in the sensor sidecar. Reading only the old path silently
    # mis-grades every Schema 2 capture as an unversioned v1.
    declared = capture.session_meta.get("schema", capture.sensor_meta.get("schema_version"))
    if declared is None:
        report.add(WARNING, "schema",
                   f"absent; assuming v1. Ingestion needs it to branch on future "
                   f"changes (current: v{SCHEMA_VERSION})")
    elif int(declared) > SCHEMA_VERSION:
        report.add(ERROR, "schema",
                   f"capture is v{declared} but this build understands v{SCHEMA_VERSION}")
    version = 1 if declared is None else int(declared)

    _check_alignment(capture, report)
    _check_run_block(capture, report)
    _check_device_block(capture, report)
    _check_conversions(capture, report)
    _check_streams(capture, report)
    _check_events(capture, report, version)
    _check_scale(capture, report)

    for board in (SENSOR_BOARD, RIG_BOARD):
        files = (capture.sensor_meta if board == SENSOR_BOARD else capture.rig_meta).get("files", {})
        for role, name in files.items():
            if role == "bin":
                continue
            if not (capture.root / name).exists():
                report.add(ERROR, f"files.{role}", f"sidecar lists {name}, which is not on disk")

    return _freeze(report) if frozen else report


def validate_dataset(root: str | Path, frozen: bool = False) -> list[SidecarReport]:
    """Validate every capture in a rig-layout directory."""
    root = Path(root)
    stems = discover_stems(root)
    if not stems:
        raise FileNotFoundError(f"{root}: no captures found")
    return [validate_sidecars(open_capture(root, stem), frozen=frozen) for stem in stems]
