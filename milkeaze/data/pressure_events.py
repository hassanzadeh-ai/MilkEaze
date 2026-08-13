"""Derive per-event ``suck`` labels from the rig's vacuum-line pressure.

This is the label source that unblocks classifier training on real data. On the pump
rig the pressure cycle *is* the suck event, so a detector over the pressure channel
produces the ``events.csv`` the model has been missing. It produces suck only — the
rig has no swallow mechanism.

Why not the naive detector
--------------------------
Detrend-and-find-troughs recovers 42 cpm cleanly at vacuum L1-L2 but reads ~60 cpm
with a period CV around 0.45 at L3-L5. That is not the pump changing rate: at high
vacuum the cycle waveform grows a secondary trough, and a broadband trough finder
counts the harmonic as a second cycle.

Two changes fix it. First, the cycle rate is estimated from the spectrum inside a
configurable plausibility band and the signal is then band-passed *narrowly around
that estimate*, which attenuates the harmonic before peak-picking rather than trying
to reject it afterwards. Second, peaks are separated by a refractory interval derived
from the estimated period, so a surviving secondary trough cannot open a new cycle.

The band is configuration, not a constant: the rig's pump is fixed at 42 cpm today but
is due to become controllable, and infant suckling runs faster than either.

Timestamp convention
--------------------
``t_ms`` marks **peak suction** — the pressure minimum of the cycle. Set
``convention="onset"`` to mark the start of the downstroke instead. The choice is
recorded in the pinned parameters written alongside the events so a training run can
always be traced back to the labelling rule that produced it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

from ..utils.logging import get_logger
from .resampling import make_grid, resample_linear

log = get_logger(__name__)

SUCK = "suck"
PEAK = "peak"
ONSET = "onset"

_MAD_TO_SIGMA = 1.4826


@dataclass
class DetectorConfig:
    """Detection parameters. Pinned into the output so labels stay reproducible."""

    #: plausible cycle-rate band in cycles per minute; the pump runs at 42 today
    band_cpm: tuple[float, float] = (18.0, 90.0)
    #: analysis grid for filtering; the raw stream is ~180 Hz but unevenly spaced
    grid_hz: float = 100.0
    #: half-width of the band-pass around the estimated rate, as a fraction of it
    band_fraction: float = 0.45
    #: keep cycles at least this deep relative to the median candidate cycle
    prominence_fraction: float = 0.35
    #: absolute floor on prominence, in robust sigmas of sample-to-sample noise
    min_prominence_sigmas: float = 3.0
    #: minimum spacing between events, as a fraction of the estimated cycle period
    refractory_fraction: float = 0.6
    #: what ``t_ms`` marks: "peak" (peak suction) or "onset" (start of downstroke)
    convention: str = PEAK
    #: events below this confidence are still emitted, but flagged for filtering
    weak_event_confidence: float = 0.35
    butter_order: int = 2

    def __post_init__(self) -> None:
        lo, hi = self.band_cpm
        if not 0 < lo < hi:
            raise ValueError(f"band_cpm must be an increasing positive pair, got {self.band_cpm}")
        if self.convention not in (PEAK, ONSET):
            raise ValueError(f"convention must be '{PEAK}' or '{ONSET}', got {self.convention!r}")
        if not 0 < self.band_fraction < 1:
            raise ValueError(f"band_fraction must be in (0, 1), got {self.band_fraction}")


@dataclass
class DetectionResult:
    events: pd.DataFrame            # t_ms, type, amplitude_psi, confidence
    cycle_rate_cpm: float           # from the median inter-event period
    cycle_rate_cv: float            # coefficient of variation of the periods
    spectral_rate_cpm: float        # from the spectrum, before peak-picking
    n_events: int
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.n_events} events, {self.cycle_rate_cpm:.1f} cpm "
            f"(spectral {self.spectral_rate_cpm:.1f}), period CV {self.cycle_rate_cv:.2f}"
        )


def _robust_sigma(x: np.ndarray) -> float:
    mad = float(np.median(np.abs(x - np.median(x))))
    return mad * _MAD_TO_SIGMA


def estimate_cycle_rate_cpm(signal: np.ndarray, fs: float,
                            band_cpm: tuple[float, float]) -> float:
    """Dominant cycle rate inside ``band_cpm``, from the Welch spectrum.

    Estimating the rate spectrally — rather than from counted peaks — is what makes the
    harmonic rejection possible, because it does not depend on peak-picking first.
    """
    lo_hz, hi_hz = band_cpm[0] / 60.0, band_cpm[1] / 60.0
    # resolve the band to a few bins per cpm without exceeding the signal length
    nperseg = int(min(len(signal), max(256, fs / lo_hz * 4)))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    in_band = (freqs >= lo_hz) & (freqs <= hi_hz)
    if not in_band.any():
        raise ValueError(
            f"cycle band {band_cpm} cpm has no spectral bins at fs={fs} Hz "
            f"over {len(signal) / fs:.1f} s of signal"
        )
    return float(freqs[in_band][int(np.argmax(psd[in_band]))] * 60.0)


def detect_suck_events(t_ms: np.ndarray, psi: np.ndarray,
                       config: DetectorConfig | None = None) -> DetectionResult:
    """Detect pump suction cycles in a pressure trace.

    ``psi`` follows the rig sign convention where suction reads negative, so cycles are
    troughs. Returns the events plus the parameters that produced them.
    """
    config = config or DetectorConfig()
    t_ms = np.asarray(t_ms, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    if t_ms.shape[0] != psi.shape[0]:
        raise ValueError(f"length mismatch: {t_ms.shape[0]} timestamps, {psi.shape[0]} samples")
    if t_ms.shape[0] < 2:
        raise ValueError("need at least 2 pressure samples")

    fs = config.grid_hz
    grid = make_grid(float(t_ms[0]), float(t_ms[-1]), fs)
    x = resample_linear(t_ms, psi, grid)[:, 0].astype(np.float64)

    duration_s = (grid[-1] - grid[0]) / 1000.0
    min_duration_s = 2.0 * 60.0 / config.band_cpm[0]
    if duration_s < min_duration_s:
        raise ValueError(
            f"pressure trace is {duration_s:.1f} s; need >= {min_duration_s:.1f} s "
            f"to resolve {config.band_cpm[0]:.0f} cpm"
        )

    spectral_cpm = estimate_cycle_rate_cpm(x - x.mean(), fs, config.band_cpm)
    f0_hz = spectral_cpm / 60.0

    # narrow band around the estimated rate, clipped to the plausibility band, so the
    # second harmonic at 2*f0 is attenuated before any peak-picking happens
    lo_hz = max(f0_hz * (1.0 - config.band_fraction), config.band_cpm[0] / 60.0)
    hi_hz = min(f0_hz * (1.0 + config.band_fraction), config.band_cpm[1] / 60.0)
    nyquist = fs / 2.0
    hi_hz = min(hi_hz, nyquist * 0.99)
    if lo_hz >= hi_hz:
        raise ValueError(f"degenerate band-pass [{lo_hz:.3f}, {hi_hz:.3f}] Hz at fs={fs} Hz")

    sos = butter(config.butter_order, [lo_hz / nyquist, hi_hz / nyquist], btype="bandpass",
                 output="sos")
    filtered = sosfiltfilt(sos, x)  # zero-phase: no group delay to correct for

    if _robust_sigma(filtered) <= 0:
        raise ValueError("band-passed pressure has zero variance; is the rig board connected?")

    period_samples = fs / f0_hz
    refractory = max(1, int(round(config.refractory_fraction * period_samples)))

    # Threshold relative to the cycles actually present, not to the signal's variance.
    # After a narrow band-pass the trace is nearly sinusoidal, and a sinusoid's robust
    # sigma is roughly half its own peak-to-peak, so any fixed multiple of sigma either
    # rejects every real cycle or accepts every ripple. Measuring the candidate
    # prominences first makes the threshold self-scaling across a 6x amplitude range.
    candidates, cprops = find_peaks(-filtered, distance=refractory, prominence=0.0)
    noise_sigma = _robust_sigma(np.diff(x)) / np.sqrt(2.0)
    if candidates.size:
        threshold = max(
            config.prominence_fraction * float(np.median(cprops["prominences"])),
            config.min_prominence_sigmas * noise_sigma,
        )
    else:
        threshold = config.min_prominence_sigmas * noise_sigma

    troughs, props = find_peaks(-filtered, distance=refractory, prominence=threshold)
    if troughs.size == 0:
        log.warning("no pressure cycles found (band %.1f-%.1f cpm, threshold %.4g psi)",
                    config.band_cpm[0], config.band_cpm[1], threshold)
        return DetectionResult(
            events=pd.DataFrame(columns=["t_ms", "type", "amplitude_psi", "confidence"]),
            cycle_rate_cpm=float("nan"), cycle_rate_cv=float("nan"),
            spectral_rate_cpm=spectral_cpm, n_events=0,
            params=_pin_params(config, fs, f0_hz, lo_hz, hi_hz, threshold, noise_sigma),
        )

    # snap each detected cycle to the true minimum of the unfiltered trace nearby, so
    # amplitudes are read off real pressure rather than the band-passed proxy
    search = max(1, int(round(0.25 * period_samples)))
    event_idx = np.array([
        max(0, i - search) + int(np.argmin(x[max(0, i - search): i + search + 1]))
        for i in troughs
    ])

    baseline = pd.Series(x).rolling(int(round(3 * period_samples)), center=True,
                                    min_periods=1).median().to_numpy()
    amplitude = baseline[event_idx] - x[event_idx]  # positive = suction below baseline

    if config.convention == ONSET:
        event_idx = _downstroke_onsets(x, baseline, event_idx, int(round(period_samples)))

    prominences = props["prominences"]
    confidence = np.clip(prominences / (np.median(prominences) * 2.0), 0.0, 1.0)

    events = pd.DataFrame({
        "t_ms": grid[event_idx],
        "type": SUCK,
        "amplitude_psi": amplitude.astype(np.float32),
        "confidence": confidence.astype(np.float32),
    })

    periods_s = np.diff(grid[event_idx]) / 1000.0
    if periods_s.size:
        rate_cpm = float(60.0 / np.median(periods_s))
        rate_cv = float(np.std(periods_s) / np.mean(periods_s))
    else:
        rate_cpm = rate_cv = float("nan")

    n_weak = int((confidence < config.weak_event_confidence).sum())
    if n_weak:
        log.info("%d/%d events below confidence %.2f (kept, flagged for filtering)",
                 n_weak, len(events), config.weak_event_confidence)

    result = DetectionResult(
        events=events, cycle_rate_cpm=rate_cpm, cycle_rate_cv=rate_cv,
        spectral_rate_cpm=spectral_cpm, n_events=len(events),
        params=_pin_params(config, fs, f0_hz, lo_hz, hi_hz, threshold, noise_sigma),
    )
    log.info("pressure cycles: %s", result.summary())
    return result


def _downstroke_onsets(x: np.ndarray, baseline: np.ndarray, peaks: np.ndarray,
                       period_samples: int) -> np.ndarray:
    """Walk back from each peak to where pressure last crossed its local baseline."""
    onsets = np.empty_like(peaks)
    for k, p in enumerate(peaks):
        lo = max(0, p - period_samples)
        segment = x[lo:p + 1] - baseline[lo:p + 1]
        above = np.flatnonzero(segment >= 0)
        onsets[k] = lo + (int(above[-1]) if above.size else 0)
    return onsets


def _pin_params(config: DetectorConfig, fs: float, f0_hz: float, lo_hz: float,
                hi_hz: float, threshold: float, noise_sigma: float) -> dict:
    return {
        "detector": "pressure_trough_v1",
        "config": {**asdict(config), "band_cpm": list(config.band_cpm)},
        "resolved": {
            "analysis_fs_hz": fs,
            "estimated_rate_cpm": f0_hz * 60.0,
            "bandpass_hz": [lo_hz, hi_hz],
            "prominence_psi": threshold,
            "noise_sigma_psi": noise_sigma,
        },
    }


def write_events(result: DetectionResult, session_dir: str | Path,
                 filename: str = "events.csv") -> Path:
    """Write ``events.csv`` plus the pinned detector parameters beside it."""
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    events_path = session_dir / filename
    result.events.to_csv(events_path, index=False)

    params_path = events_path.with_name(f"{events_path.stem}_detector.json")
    params_path.write_text(
        json.dumps({**result.params,
                    "n_events": result.n_events,
                    "cycle_rate_cpm": result.cycle_rate_cpm,
                    "cycle_rate_cv": result.cycle_rate_cv}, indent=2),
        encoding="utf-8",
    )
    log.info("wrote %d events to %s", result.n_events, events_path)
    return events_path
