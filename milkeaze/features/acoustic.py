"""Acoustic features — the swallow cue.

Computed on the *native* 8 kHz mic stream for the window (not the coarse grid
envelope). Swallows have a characteristic broadband "click"; we summarize each mic
channel with log-energy, ZCR, spectral shape, a swallow-band energy ratio, and a
compact MFCC summary, plus an L/R difference for localization / artifact rejection.
"""
from __future__ import annotations

import numpy as np

# Swallow acoustic energy tends to concentrate in a mid band; tune once we have
# labeled swallow events to confirm the band edges on real recordings.
SWALLOW_BAND_HZ = (150.0, 1200.0)
N_MFCC = 13


def _safe_log(x: np.ndarray | float) -> np.ndarray | float:
    return np.log(np.asarray(x) + 1e-8)


def _spectral_shape(x: np.ndarray, fs: float) -> tuple[float, float, float]:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    p = spec + 1e-8
    centroid = float(np.sum(freqs * p) / np.sum(p))
    cumulative = np.cumsum(p)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85 * cumulative[-1]))
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * p) / np.sum(p)))
    return centroid, rolloff, bandwidth


def _swallow_band_ratio(x: np.ndarray, fs: float) -> float:
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    band = (freqs >= SWALLOW_BAND_HZ[0]) & (freqs <= SWALLOW_BAND_HZ[1])
    total = float(np.sum(spec)) + 1e-8
    return float(np.sum(spec[band]) / total)


def _mfcc_summary(x: np.ndarray, fs: float, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Mean MFCCs over the window. Uses librosa if available, else a mel-less DCT
    fallback so the pipeline still runs without the optional dependency."""
    try:
        import librosa

        m = librosa.feature.mfcc(y=x.astype(np.float32), sr=int(fs), n_mfcc=n_mfcc)
        return m.mean(axis=1)
    except Exception:  # pragma: no cover - fallback path
        spec = _safe_log(np.abs(np.fft.rfft(x)) + 1e-8)
        from numpy.fft import rfft
        dct = np.real(rfft(spec))[:n_mfcc]
        if dct.size < n_mfcc:
            dct = np.pad(dct, (0, n_mfcc - dct.size))
        return dct.astype(np.float32)


def mic_channel_features(ch: np.ndarray, fs: float) -> list[float]:
    log_energy = float(_safe_log(np.mean(ch ** 2)))
    rms = float(np.sqrt(np.mean(ch ** 2)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(ch - np.mean(ch)))) > 0))
    centroid, rolloff, bandwidth = _spectral_shape(ch, fs)
    swb = _swallow_band_ratio(ch, fs)
    mfcc = _mfcc_summary(ch, fs)
    return [log_energy, rms, zcr, centroid, rolloff, bandwidth, swb, *mfcc.tolist()]


def acoustic_features(mic_block: np.ndarray, fs: float) -> np.ndarray:
    """``mic_block`` is (n_time, 2) at native mic rate. Returns flat vector incl. L/R diff."""
    feats: list[float] = []
    for c in range(mic_block.shape[1]):
        feats.extend(mic_channel_features(mic_block[:, c], fs))
    if mic_block.shape[1] == 2:
        lr_diff = float(np.mean(np.abs(mic_block[:, 0] - mic_block[:, 1])))
        feats.append(lr_diff)
    return np.asarray(feats, dtype=np.float32)
