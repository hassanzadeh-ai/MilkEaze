"""Non-learned baselines the CNN+LSTM has to beat to justify itself.

Every number the model produces is meaningless without a floor to compare it against,
and on this project the floors are unusually strong:

* **Volume** — ridge regression on the hand-feature vector. Closed form, no iterations,
  no seed. Reported next to the *mean predictor*, because a session's windows all carry
  similar volume and predicting the training mean is already respectable.
* **Suck rate and events** — the same spectral estimate and trough finder used on rig
  pressure, pointed at the strain channels instead. If a band-pass and ``find_peaks``
  recover the cycle rate from the ring, then the network is not needed for *counting*
  and its job narrows to per-event timing, swallow discrimination, and volume.

Deliberately numpy and scipy only: the baselines have to be runnable wherever the data
is, including where the training stack will not import.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

from ..data.pressure_events import (
    DetectionResult, DetectorConfig, detect_suck_events, estimate_cycle_rate_cpm,
)
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class RidgeRegressor:
    """Standardising ridge, fit in closed form.

    Columns with no variance in the training split — the zero-filled mic slot produces
    several — are pinned to zero weight rather than dividing by zero.
    """

    coef: np.ndarray
    intercept: float
    mean: np.ndarray
    scale: np.ndarray
    alpha: float

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean) / self.scale

    def predict(self, X) -> np.ndarray:
        return self._standardize(X) @ self.coef + self.intercept

    @property
    def n_active_features(self) -> int:
        return int(np.count_nonzero(self.coef))


def fit_ridge(X, y, alpha: float = 1.0) -> RidgeRegressor:
    """Ridge regression with standardised inputs and a free intercept."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features), got shape {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")

    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    dead = scale <= 0
    scale = np.where(dead, 1.0, scale)

    Z = (X - mean) / scale
    Z[:, dead] = 0.0
    y_mean = float(y.mean())

    gram = Z.T @ Z + alpha * np.eye(Z.shape[1])
    coef = np.linalg.solve(gram, Z.T @ (y - y_mean))
    coef[dead] = 0.0

    if dead.any():
        log.debug("ridge: %d/%d features had no training variance", int(dead.sum()), dead.size)
    return RidgeRegressor(coef=coef, intercept=y_mean, mean=mean, scale=scale, alpha=float(alpha))


@dataclass
class RegressionScores:
    mae: float
    rmse: float
    bias: float
    r2: float

    def summary(self) -> str:
        return (f"MAE={self.mae:.3f} RMSE={self.rmse:.3f} "
                f"bias={self.bias:+.3f} R2={self.r2:.3f}")


def regression_scores(pred, true) -> RegressionScores:
    """Standard per-window regression scores.

    ``r2`` is against the *truth's own* mean, so 0 means "no better than predicting the
    mean of the evaluation set" and negative means actively worse than that.
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    true = np.asarray(true, dtype=np.float64).ravel()
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs true {true.shape}")
    err = pred - true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return RegressionScores(
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        bias=float(np.mean(err)),
        r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    )


def mean_predictor_scores(y_train, y_eval) -> RegressionScores:
    """The floor for the volume head: answer the training mean, every window."""
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    y_eval = np.asarray(y_eval, dtype=np.float64).ravel()
    return regression_scores(np.full(y_eval.shape, y_train.mean()), y_eval)


def in_band_power_fraction(strain_block: np.ndarray, fs: float,
                           band_cpm: tuple[float, float] = (18.0, 90.0)) -> np.ndarray:
    """Per-channel share of power inside the cycle band, as a scale-free health score.

    Ranking channels by raw variance instead picks the noisiest one, which on this
    hardware is wrong twice over: the arc taps sit near their divider's singularity, so
    the resistance inversion hands them variance six orders of magnitude above the radial
    channels while giving them the least mechanical response of the eight. The in-band
    *fraction* asks the question that matters — which channel is mostly rhythm rather
    than mostly noise — and is comparable across channels with wildly different gains.
    """
    block = np.asarray(strain_block, dtype=np.float64)
    if block.ndim != 2:
        raise ValueError(f"strain_block must be (n_time, n_channels), got {block.shape}")

    lo_hz, hi_hz = band_cpm[0] / 60.0, band_cpm[1] / 60.0
    nperseg = int(min(block.shape[0], max(256, fs / lo_hz * 4)))
    freqs, psd = welch(block - block.mean(axis=0, keepdims=True), fs=fs,
                       nperseg=nperseg, axis=0)

    in_band = (freqs >= lo_hz) & (freqs <= hi_hz)
    if not in_band.any():
        raise ValueError(f"cycle band {band_cpm} cpm has no spectral bins at fs={fs} Hz")

    total = psd.sum(axis=0)
    return np.where(total > 0, psd[in_band].sum(axis=0) / np.where(total > 0, total, 1.0), 0.0)


def most_rhythmic_channel(strain_block: np.ndarray, fs: float,
                          band_cpm: tuple[float, float] = (18.0, 90.0)) -> int:
    """Index of the channel with the largest share of its power inside the cycle band."""
    return int(np.argmax(in_band_power_fraction(strain_block, fs, band_cpm)))


def strain_rate_cpm(strain_block: np.ndarray, fs: float,
                    band_cpm: tuple[float, float] = (18.0, 90.0)) -> float:
    """Suck rate estimated straight from the strain channels, no model involved."""
    block = np.asarray(strain_block, dtype=np.float64)
    channel = most_rhythmic_channel(block, fs, band_cpm)
    centred = block[:, channel] - block[:, channel].mean()
    return estimate_cycle_rate_cpm(centred, fs, band_cpm)


def strain_event_candidates(t_ms: np.ndarray, strain_block: np.ndarray,
                            config: DetectorConfig | None = None) -> list[DetectionResult]:
    """Both polarity hypotheses for the strain cycle detector, steadiest period first.

    Whether suction deflects a given channel positive or negative depends on the physical
    channel map, which is still unverified, so *neither* sign can be assumed. Both are
    returned because the choice is not identifiable from strain alone: a sinusoid-like
    cycle has an equally steady period either way up, and the two hypotheses differ by
    half a cycle - roughly 700 ms at 42 cpm, far outside any sensible match tolerance.

    Callers scoring against reference events should score both and say which won, rather
    than accept a coin flip that lands on F1 = 1.0 or F1 = 0.0 with nothing in between.
    """
    config = config or DetectorConfig()
    block = np.asarray(strain_block, dtype=np.float64)
    t = np.asarray(t_ms, dtype=np.float64)

    # channel selection happens at the rate the block actually arrives at, not the
    # detector's internal analysis rate, which it resamples to later
    step_ms = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    if step_ms <= 0:
        raise ValueError("strain timestamps are not increasing")
    index = most_rhythmic_channel(block, 1000.0 / step_ms, config.band_cpm)
    channel = block[:, index] - block[:, index].mean()

    found = [detect_suck_events(t_ms, sign * channel, config) for sign in (1.0, -1.0)]
    found = [r for r in found if r.n_events > 0]
    if not found:
        raise ValueError("no strain cycles detected in either polarity")

    found.sort(key=lambda r: r.cycle_rate_cv if np.isfinite(r.cycle_rate_cv) else np.inf)
    return found


def strain_event_baseline(t_ms: np.ndarray, strain_block: np.ndarray,
                          config: DetectorConfig | None = None) -> DetectionResult:
    """The steadier-period polarity hypothesis, for callers that only need a rate.

    A count and a rate are polarity-independent; per-event *timing* is not. Use
    :func:`strain_event_candidates` whenever timing is being scored.
    """
    return strain_event_candidates(t_ms, strain_block, config)[0]


#: Which way suction deflects each strain channel, ``+1`` for positive and ``-1`` for
#: negative, with ``0`` marking a channel that carries no usable cycle at all.
#:
#: This is a *measured* map, not a wiring diagram. The polarity of a channel is not
#: identifiable from strain alone — see :func:`strain_event_candidates` — so it was
#: recovered by scoring both hypotheses for all eight channels against the pressure
#: reference events on all 34 captures of the 20260816 sweep (``scripts/strain_events.py``).
#: The result is unanimous rather than marginal: for every channel below except
#: ``Radial_2`` the winning sign took all 34 runs, and the losing sign scores F1 = 0.000
#: while still counting the right number of cycles, which is the half-cycle offset the
#: two hypotheses are expected to differ by. ``Radial_2`` runs inverted relative to the
#: other three radials and is the weakest of them (F1 0.699), and ``Arc_2_out`` is dead
#: — it detects at all on only 20 of 34 runs and undercounts by 95% when it does.
#:
#: Re-measure this if the ring build, the wiring or the channel map changes. A silent
#: polarity flip costs half a cycle of timing error, not a small one.
STRAIN_POLARITY: dict[str, int] = {
    "Radial_1": +1,
    "Radial_2": -1,
    "Radial_3": +1,
    "Radial_4": +1,
    "Arc_2_in": +1,
    "Arc_2_out": 0,
    "Arc_4_in": +1,
    "Arc_4_out": +1,
}


def strain_consensus_signal(strain_block: np.ndarray, fs: float, channels: list[str],
                            polarity: dict[str, int] | None = None,
                            band_cpm: tuple[float, float] = (18.0, 90.0),
                            min_fraction: float = 0.5) -> np.ndarray:
    """Average the healthy strain channels into one detector-ready trace.

    Returns a signal in which suction reads as a *trough*, matching the rig pressure
    convention, so it can be handed straight to :func:`detect_suck_events`.

    Single-channel detection already scores F1 0.998 on the bench batch, so the point of
    pooling is not accuracy, it is surviving the field. Bend-sensor channels have been
    failing after the silicone overmold, and a detector pinned to one named channel stops
    working the moment that channel is the one that dies. Channels are admitted on their
    in-band power fraction relative to the best channel present, so a dead or noise-only
    tap drops out on its own rather than needing to be named.

    Each channel is z-scored before averaging because the arc taps carry variance six
    orders of magnitude above the radials, and an unweighted mean of raw counts would be
    the arc channels wearing the radials as a rounding error.
    """
    block = np.asarray(strain_block, dtype=np.float64)
    if block.ndim != 2 or block.shape[1] != len(channels):
        raise ValueError(f"block {block.shape} does not match {len(channels)} channel names")
    polarity = STRAIN_POLARITY if polarity is None else polarity

    signs = np.array([polarity.get(name, 0) for name in channels], dtype=np.float64)
    fraction = in_band_power_fraction(block, fs, band_cpm)

    usable = signs != 0
    if not usable.any():
        raise ValueError("no channels with a known suction polarity")
    healthy = usable & (fraction >= min_fraction * fraction[usable].max())
    if not healthy.any():
        raise ValueError("no strain channel carries enough in-band power to detect on")

    centred = block[:, healthy] - block[:, healthy].mean(axis=0, keepdims=True)
    scale = centred.std(axis=0)
    scale[scale <= 0] = 1.0
    # negated: the map says which way suction deflects, the detector wants it downward
    return np.mean(centred / scale * -signs[healthy], axis=1)


def strain_consensus_events(t_ms: np.ndarray, strain_block: np.ndarray,
                            channels: list[str], config: DetectorConfig | None = None,
                            polarity: dict[str, int] | None = None,
                            min_fraction: float = 0.5) -> DetectionResult:
    """Suck events from the strain ring alone, with no polarity ambiguity left to resolve.

    This is the shippable path: it needs no vacuum line, no scale and no rig board, only
    the eight strain channels the product already carries.
    """
    config = config or DetectorConfig()
    t = np.asarray(t_ms, dtype=np.float64)
    step_ms = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    if step_ms <= 0:
        raise ValueError("strain timestamps are not increasing")

    signal = strain_consensus_signal(strain_block, 1000.0 / step_ms, channels,
                                     polarity, config.band_cpm, min_fraction)
    return detect_suck_events(t, signal, config)
