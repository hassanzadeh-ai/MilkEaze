"""Reader for the production dual-board capture layout.

A capture is two independent boards writing to one directory:

``<stem>_sensor_*``
    strain, IMU, audio, and the USB scale, on the sensor board's device clock.

``<stem>_rig_*``
    vacuum-line pressure and ambient temperature, on the rig board's device clock.

Neither device clock is the host clock and the two do not agree with each other, so
every stream is mapped onto host monotonic seconds before anything is compared across
boards. Each board ships a ``*_sync.csv`` of ``(host_mono_s, device_ts_us)`` pairs and
a per-board skew estimate in ``<stem>_session.json``.

**Alignment uses the sidecar skew, never a fresh fit.** The sensor board is on Wi-Fi
and its sync pairs carry outliers large enough that an ordinary least-squares fit
lands tens of milliseconds off, which is a whole strain sample at 82 Hz. The sidecar
skew is produced by a robust estimator, so this module takes that slope as given and
only solves for the offset, robustly.

The sidecar estimate is not always usable, though. On the 20260718 batch the sensor
board only resolves its skew on the two long captures: its standard error grows from
2.3 ppm at 500 s to 42 ppm at 91 s, and on the shortest runs the fitted skew even
changes sign. Wi-Fi jitter of ~2 ms simply does not constrain a slope over a 90 s
baseline. The rig board, being wired, stays under 3.6 ppm throughout.

A board's crystal offset belongs to the hardware rather than the run, so an
unresolved capture falls back to :func:`pooled_skew_ppm` — the inverse-variance
weighted skew across every capture in the directory that did resolve. Pass
``skew_fallback="strict"`` to refuse such captures instead, or ``"session"`` to use
the unresolved value as-is.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import SensorConfig
from ..utils.logging import get_logger
from .calibration import COUNTS, PHYSICAL
from .ingestion import RawSession, SessionLayout, effective_rate_hz, repair_monotonic

log = get_logger(__name__)

SENSOR_BOARD = "sensor"
RIG_BOARD = "rig"

RESOLVED = "resolved"
_MAD_TO_SIGMA = 1.4826

#: Device timestamps are 32-bit microsecond counters, so they roll over every
#: ``2**32 us`` = 4294.967 s (71.6 min) of board uptime. A run that straddles a boundary
#: has a single backward jump of most of that span. Left alone it does not degrade a
#: clock fit, it destroys it: on the one capture in the 20260816 batch where the rig
#: board wrapped, an unwrapped least-squares fit returns a slope of -0.045 instead of
#: 1.000, i.e. rig time running backwards at 45x. Steve's capture tool unwraps before
#: writing its sidecar estimate, which is why the sidecar reports a clean 0.36 ms jitter
#: for that capture and nothing upstream looks wrong.
_WRAP_US = 2 ** 32

#: Only a jump larger than half the counter range is read as a rollover; smaller
#: backward steps are ordinary out-of-order packets and belong to ``repair_monotonic``.
_WRAP_THRESHOLD_US = _WRAP_US / 2

#: what to do when a capture's own skew estimate did not resolve
POOLED = "pooled"    # substitute the dataset-level estimate (default)
SESSION = "session"  # use the unresolved per-session value anyway
STRICT = "strict"    # refuse to load the capture

#: sensor-board strain CSV columns are ``ch<i>_<Name>``; config uses lowercase names
_STRAIN_COLUMN_PREFIX = "ch"

#: config mic channel name -> audio CSV column
_MIC_COLUMNS = {"mic_L": "left", "mic_R": "right"}

_IMU_COLUMNS = {
    "acc_x": "ax_ms2", "acc_y": "ay_ms2", "acc_z": "az_ms2",
    "gyr_x": "gx_dps", "gyr_y": "gy_dps", "gyr_z": "gz_dps",
}


def unwrap_device_us(raw: np.ndarray) -> tuple[np.ndarray, int]:
    """Undo 32-bit rollover in a device microsecond counter.

    Returns the monotonic series and the number of rollovers found.
    """
    values = np.asarray(raw, dtype=np.float64)
    if values.size < 2:
        return values, 0
    rollovers = np.diff(values) < -_WRAP_THRESHOLD_US
    if not rollovers.any():
        return values, 0
    counts = np.concatenate(([0.0], np.cumsum(rollovers.astype(np.float64))))
    return values + counts * _WRAP_US, int(rollovers.sum())


@dataclass
class BoardClock:
    """Maps a board's device timestamps onto host monotonic seconds."""

    board: str
    skew_ppm: float
    offset_s: float
    jitter_ms: float
    n_sync_points: int
    confidence: str
    skew_source: str = "session"

    #: RMS of (host - predicted host) over the sync points, in ms
    residual_ms: float = 0.0

    #: rollovers seen in this board's sync file; >0 means the run straddled a boundary
    wraps: int = 0

    #: unwrapped device-time range covered by the sync file, in us. Streams are shifted
    #: by whole counter periods onto this range, so a stream that begins after a rollover
    #: is not read as belonging to the previous epoch.
    device_lo_us: float = 0.0
    device_hi_us: float = 0.0

    @property
    def slope(self) -> float:
        """``host_s = slope * device_s + offset_s``."""
        return 1.0 + self.skew_ppm * 1e-6

    def align_device_us(self, device_ts_us: np.ndarray) -> np.ndarray:
        """Unwrap a stream and put it in the same counter epoch as the sync file."""
        values, _ = unwrap_device_us(device_ts_us)
        if values.size == 0 or self.device_hi_us <= self.device_lo_us:
            return values
        centre = 0.5 * (self.device_lo_us + self.device_hi_us)
        shift = round((centre - float(np.median(values))) / _WRAP_US)
        return values + shift * _WRAP_US

    def to_host_s(self, device_ts_us: np.ndarray) -> np.ndarray:
        aligned = self.align_device_us(device_ts_us)
        return self.slope * (aligned / 1e6) + self.offset_s


@dataclass
class RigCapture:
    """A production capture, including the rig streams the model does not consume."""

    stem: str
    root: Path
    session_meta: dict[str, Any]
    sensor_meta: dict[str, Any]
    rig_meta: dict[str, Any]
    clocks: dict[str, BoardClock]
    t0_host_s: float
    #: per-stream timestamp repair stats, filled in as streams are read
    repairs: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def vacuum_level(self) -> int | None:
        return self.sensor_meta.get("run", {}).get("vacuum_level")

    @property
    def cycle_rate_cpm(self) -> float | None:
        rate = self.sensor_meta.get("run", {}).get("cycle_rate_cpm")
        return float(rate) if rate is not None else None


def discover_stems(root: str | Path) -> list[str]:
    """List the capture stems in a rig-layout directory, in filename order."""
    root = Path(root)
    return sorted(p.name[: -len("_session.json")] for p in root.glob("*_session.json"))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing sidecar: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def pooled_skew_ppm(root: str | Path, board: str) -> float | None:
    """Inverse-variance weighted skew for ``board``, over captures that resolved it.

    A board's crystal offset is a property of the hardware, not of the run, so the
    long captures that pin it down are the best estimate available for the short ones
    that cannot. Returns ``None`` when nothing in the directory resolved.
    """
    root = Path(root)
    skews: list[float] = []
    weights: list[float] = []
    for path in sorted(root.glob("*_session.json")):
        info = (json.loads(path.read_text(encoding="utf-8"))
                .get("alignment", {}).get(board, {}))
        if info.get("skew_confidence") != "resolved":
            continue
        skew = info.get("skew_applied", info.get("skew_ppm"))
        se = info.get("skew_se_ppm")
        if skew is None:
            continue
        skews.append(float(skew))
        weights.append(1.0 / max(float(se), 1e-6) ** 2 if se else 1.0)

    if not skews:
        return None
    return float(np.average(skews, weights=weights))


def _board_clock(root: Path, stem: str, board: str, alignment: dict[str, Any],
                 skew_fallback: str = POOLED) -> BoardClock:
    sync_path = root / f"{stem}_{board}_sync.csv"
    if not sync_path.exists():
        raise FileNotFoundError(f"missing clock sync file: {sync_path}")
    sync = pd.read_csv(sync_path)
    for col in ("host_mono_s", "device_ts_us"):
        if col not in sync.columns:
            raise ValueError(f"{sync_path.name}: missing column '{col}'")
    if len(sync) < 2:
        raise ValueError(f"{sync_path.name}: only {len(sync)} sync point(s); cannot align clocks")

    info = alignment.get(board, {})
    # `skew_ppm` is the estimate; `skew_applied` is what the capture tool already used,
    # and it is 0 exactly when the estimate did not resolve. We want the estimate.
    skew_ppm = info.get("skew_ppm")
    if skew_ppm is None:
        raise ValueError(
            f"{stem}: no skew for board '{board}' in session.json; refusing to guess "
            "(a naive fit on Wi-Fi sync points is not accurate enough)"
        )

    confidence = str(info.get("skew_confidence", "unknown"))
    source = "session"
    if confidence != RESOLVED:
        if skew_fallback == STRICT:
            raise ValueError(
                f"{stem}: board '{board}' skew is '{confidence}' and skew_fallback="
                f"'{STRICT}'; refusing to align on an unresolved clock"
            )
        if skew_fallback == POOLED:
            pooled = pooled_skew_ppm(root, board)
            if pooled is None:
                log.warning("%s: board '%s' skew is '%s' and no capture in %s resolved it; "
                            "using the session estimate", stem, board, confidence, root.name)
            else:
                log.warning(
                    "%s: board '%s' skew is '%s' (%.2f +/- %.2f ppm over %.0f s); "
                    "substituting the pooled estimate %.2f ppm",
                    stem, board, confidence, float(skew_ppm),
                    float(info.get("skew_se_ppm", float("nan"))),
                    float(info.get("duration_s", float("nan"))), pooled,
                )
                skew_ppm, source = pooled, "pooled"
        else:
            log.warning("%s: board '%s' skew confidence is '%s', not 'resolved'",
                        stem, board, confidence)

    host_s = sync["host_mono_s"].to_numpy(dtype=np.float64)
    device_us, wraps = unwrap_device_us(sync["device_ts_us"].to_numpy(dtype=np.float64))
    if wraps:
        log.warning(
            "%s: board '%s' device counter rolled over %d time(s) mid-run; unwrapping "
            "before the clock fit (an unwrapped fit here returns a negative slope)",
            stem, board, wraps,
        )
    device_s = device_us / 1e6

    # host_s = (1 + skew_ppm*1e-6) * device_s + offset. The slope is given; only the
    # offset is solved here, and by median so Wi-Fi outliers cannot drag it.
    slope = 1.0 + float(skew_ppm) * 1e-6
    offset_s = float(np.median(host_s - slope * device_s))

    residual_ms = (host_s - (slope * device_s + offset_s)) * 1e3
    jitter_ms = _robust_scatter_ms(residual_ms)
    _check_convention(stem, board, jitter_ms, host_s, device_s)

    return BoardClock(
        board=board,
        skew_ppm=float(skew_ppm),
        offset_s=offset_s,
        jitter_ms=jitter_ms,
        n_sync_points=len(sync),
        confidence=confidence,
        skew_source=source,
        residual_ms=float(np.sqrt(np.mean(residual_ms ** 2))),
        wraps=wraps,
        device_lo_us=float(device_us.min()),
        device_hi_us=float(device_us.max()),
    )


def _robust_scatter_ms(residual_ms: np.ndarray) -> float:
    return float(np.median(np.abs(residual_ms - np.median(residual_ms))) * _MAD_TO_SIGMA)


def _check_convention(stem: str, board: str, jitter_ms: float,
                      host_s: np.ndarray, device_s: np.ndarray) -> None:
    """Guard against the sidecar's skew sign/scale convention changing under us.

    Applying the skew in the wrong direction still yields a plausible-looking timebase
    — it just drifts, by twice the skew over the length of the run. The check compares
    our scatter against the best slope a plain fit could achieve on the same points: a
    correct convention lands close to it, while an inverted one is worse by the full
    accumulated drift. This is deliberately not compared against the sidecar's own
    ``jitter_ms``, which is computed by the capture tool under its own outlier
    handling and reads several times lower on the Wi-Fi sensor board.
    """
    if not np.isfinite(jitter_ms) or device_s.shape[0] < 3:
        return
    best_slope, _ = np.polyfit(device_s, host_s, 1)
    best_residual = (host_s - (best_slope * device_s
                               + np.median(host_s - best_slope * device_s))) * 1e3
    best = _robust_scatter_ms(best_residual)
    if jitter_ms > max(3.0 * best, best + 5.0):
        log.warning(
            "%s: board '%s' aligns to %.2f ms scatter where a direct fit reaches "
            "%.2f ms. Check that skew_ppm still means host = (1 + skew) * device.",
            stem, board, jitter_ms, best,
        )


def open_capture(root: str | Path, stem: str | None = None,
                 skew_fallback: str = POOLED) -> RigCapture:
    """Read the sidecars and solve both board clocks, without loading bulk streams."""
    root = Path(root)
    if stem is None:
        stems = discover_stems(root)
        if not stems:
            raise FileNotFoundError(f"{root}: no *_session.json found")
        if len(stems) > 1:
            raise ValueError(
                f"{root} holds {len(stems)} captures; pass stem= to choose one: {stems}"
            )
        stem = stems[0]

    session_meta = _read_json(root / f"{stem}_session.json")
    sensor_meta = _read_json(root / f"{stem}_sensor.json")
    rig_meta = _read_json(root / f"{stem}_rig.json")

    alignment = session_meta.get("alignment", {})
    clocks = {
        board: _board_clock(root, stem, board, alignment, skew_fallback)
        for board in (SENSOR_BOARD, RIG_BOARD)
    }

    # a common origin so both boards land on one zero-based millisecond timebase
    first_host = []
    for board in (SENSOR_BOARD, RIG_BOARD):
        sync = pd.read_csv(root / f"{stem}_{board}_sync.csv", usecols=["host_mono_s"])
        first_host.append(float(sync["host_mono_s"].iloc[0]))

    return RigCapture(
        stem=stem, root=root, session_meta=session_meta, sensor_meta=sensor_meta,
        rig_meta=rig_meta, clocks=clocks, t0_host_s=min(first_host),
    )


def _to_session_ms(capture: RigCapture, board: str, device_ts_us: np.ndarray,
                   stream: str | None = None) -> np.ndarray:
    host_s = capture.clocks[board].to_host_s(device_ts_us)
    t_ms = (host_s - capture.t0_host_s) * 1000.0
    if stream is None:
        return t_ms
    repaired, stats = repair_monotonic(t_ms, f"{capture.stem}:{stream}")
    capture.repairs[stream] = stats
    return repaired


def _strain_columns(df: pd.DataFrame, sensors: SensorConfig) -> list[str]:
    """Match ``ch<i>_<Name>`` CSV columns to the configured channel order."""
    by_index: dict[int, str] = {}
    for col in df.columns:
        if not col.startswith(_STRAIN_COLUMN_PREFIX):
            continue
        head, _, _ = col[len(_STRAIN_COLUMN_PREFIX):].partition("_")
        if head.isdigit():
            by_index[int(head)] = col

    names = sensors.strain_channel_names()
    missing = [i for i in range(len(names)) if i not in by_index]
    if missing:
        raise ValueError(
            f"strain CSV is missing channel index/indices {missing}; found {sorted(by_index)}"
        )
    return [by_index[i] for i in range(len(names))]


def _load_mic(root: Path, stem: str, capture: RigCapture,
              sensors: SensorConfig) -> tuple[np.ndarray, np.ndarray]:
    """Read the mic, keeping the full model-contract width.

    Only ``mic.active_channels`` are read; inactive slots (currently the faulty second
    MP34DT01-M) are zero-filled so the 16-channel input shape does not change when the
    hardware is repaired.
    """
    contract = sensors.mic_channel_names()
    active = [n for n in sensors.mic_active_channels() if n in contract]
    if not active:
        raise ValueError(f"no active mic channels configured; contract is {contract}")

    active_cols = [_MIC_COLUMNS[n] for n in active]
    audio_df = pd.read_csv(root / f"{stem}_sensor_audio.csv", usecols=["time_us", *active_cols])
    t_ms = _to_session_ms(capture, SENSOR_BOARD,
                          audio_df["time_us"].to_numpy(dtype=np.float64), "mic")

    mic = np.zeros((len(audio_df), len(contract)), dtype=np.float32)
    for name in active:
        mic[:, contract.index(name)] = audio_df[_MIC_COLUMNS[name]].to_numpy(dtype=np.float32)

    inactive = [n for n in contract if n not in active]
    if inactive:
        log.info("capture %s: mic channel(s) %s inactive, zero-filled", stem, inactive)
    return t_ms, mic


def load_pressure(capture: RigCapture) -> tuple[np.ndarray, np.ndarray]:
    """Vacuum-line pressure as ``(t_ms, psi)`` on the session timebase.

    Suction reads negative, per the sidecar sign convention. Conversion constants come
    from the rig sidecar rather than being hardcoded, since they are per-part.
    """
    path = capture.root / f"{capture.stem}_rig_pressure.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing rig pressure stream: {path}")
    df = pd.read_csv(path, usecols=["time_us", "p_raw"])

    cal = capture.rig_meta["conversion"]["rig"]["pressure"]
    out_min = float(cal["out_min_counts"])
    out_max = float(cal["out_max_counts"])
    p_min = float(cal["p_min_psi"])
    p_max = float(cal["p_max_psi"])

    raw = df["p_raw"].to_numpy(dtype=np.float64)
    psi = (raw - out_min) * (p_max - p_min) / (out_max - out_min) + p_min
    t_ms = _to_session_ms(capture, RIG_BOARD, df["time_us"].to_numpy(dtype=np.float64))
    return t_ms, psi.astype(np.float32)


def load_temperature(capture: RigCapture) -> tuple[np.ndarray, np.ndarray]:
    """Rig ambient temperature as ``(t_ms, degC)`` from the TMP117."""
    path = capture.root / f"{capture.stem}_rig_temp.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing rig temperature stream: {path}")
    df = pd.read_csv(path, usecols=["time_us", "tmp117_raw"])
    c_per_count = float(capture.rig_meta["conversion"]["rig"]["tmp117"]["c_per_count"])
    degc = df["tmp117_raw"].to_numpy(dtype=np.float64) * c_per_count
    t_ms = _to_session_ms(capture, RIG_BOARD, df["time_us"].to_numpy(dtype=np.float64))
    return t_ms, degc.astype(np.float32)


def load_rig_session(root: str | Path, sensors: SensorConfig | None = None,
                     stem: str | None = None,
                     skew_fallback: str = POOLED) -> RawSession:
    """Load one production capture into the layout-independent :class:`RawSession`."""
    sensors = sensors or SensorConfig.load()
    capture = open_capture(root, stem, skew_fallback)
    root = capture.root
    stem = capture.stem

    strain_df = pd.read_csv(root / f"{stem}_sensor_strain.csv")
    if "scan_t_us" not in strain_df.columns:
        raise ValueError(f"{stem}_sensor_strain.csv: missing 'scan_t_us'")
    strain_cols = _strain_columns(strain_df, sensors)
    strain_t_ms = _to_session_ms(capture, SENSOR_BOARD,
                                 strain_df["scan_t_us"].to_numpy(dtype=np.float64), "strain")
    strain = strain_df[strain_cols].to_numpy(dtype=np.float32)

    imu_names = sensors.imu_channel_names()
    imu_cols = [_IMU_COLUMNS[n] for n in imu_names]
    imu_df = pd.read_csv(root / f"{stem}_sensor_imu.csv", usecols=["time_us", *imu_cols])
    imu_t_ms = _to_session_ms(capture, SENSOR_BOARD,
                              imu_df["time_us"].to_numpy(dtype=np.float64), "imu")
    imu = imu_df[imu_cols].to_numpy(dtype=np.float32)

    mic_t_ms, mic = _load_mic(root, stem, capture, sensors)

    scale_path = root / f"{stem}_sensor_scale.csv"
    if scale_path.exists():
        scale_df = pd.read_csv(scale_path, usecols=["host_mono_s", "grams"])
        # the scale is read on the host, so it needs no device-clock correction
        scale_t_ms = (scale_df["host_mono_s"].to_numpy(dtype=np.float64)
                      - capture.t0_host_s) * 1000.0
        scale_g = scale_df["grams"].to_numpy(dtype=np.float32)
        has_scale = True
    else:
        scale_t_ms = np.empty(0, dtype=np.float64)
        scale_g = np.empty(0, dtype=np.float32)
        has_scale = False
        log.warning("capture %s: no scale stream — no volume targets available", stem)

    meta: dict[str, Any] = {
        "session_id": stem,
        "layout": SessionLayout.RIG.value,
        "session": capture.session_meta,
        "sensor": capture.sensor_meta,
        "rig": capture.rig_meta,
        "clocks": {b: vars(c) for b, c in capture.clocks.items()},
        "timestamp_repairs": dict(capture.repairs),
        "vacuum_level": capture.vacuum_level,
        "cycle_rate_cpm": capture.cycle_rate_cpm,
    }

    log.info(
        "loaded rig capture %s: strain %.1f Hz, imu %.1f Hz, mic %.0f Hz, %d mic channel(s)",
        stem, effective_rate_hz(strain_t_ms), effective_rate_hz(imu_t_ms),
        effective_rate_hz(mic_t_ms), mic.shape[1],
    )

    return RawSession(
        session_id=stem,
        strain_t_ms=strain_t_ms,
        imu_t_ms=imu_t_ms,
        mic_t_ms=mic_t_ms,
        scale_t_ms=scale_t_ms,
        strain=strain,
        imu=imu,
        mic=mic,
        scale_g=scale_g,
        has_scale=has_scale,
        measured_rates_hz={
            "strain": effective_rate_hz(strain_t_ms),
            "imu": effective_rate_hz(imu_t_ms),
            "mic": effective_rate_hz(mic_t_ms),
        },
        layout=SessionLayout.RIG,
        # the sensor board converts IMU on-device; strain and audio stay as counts
        units={"strain": COUNTS, "imu": PHYSICAL, "mic": COUNTS},
        meta=meta,
    )
