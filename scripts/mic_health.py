"""Measure what the two microphone channels actually carry, per capture.

The sidecar says whether each MP34DT01-M was enabled, which is a statement about
configuration rather than about the signal that came back. Three failures are known and
the sidecar distinguishes none of them:

* a dead element, which writes a constant or near-constant channel;
* PDM collapse, where one channel's data is duplicated into both, so the capture claims
  stereo and carries mono - this is the one Neal reports on the live units;
* genuine stereo, which is the only case where any spatial acoustic feature is real.

Run against a capture directory to see which case each capture is in.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milkeaze.data.rig_session import discover_stems


def best_lag(left: np.ndarray, right: np.ndarray, max_lag: int = 4) -> tuple[int, float]:
    """The sample shift that best explains right from left, and the residual there.

    Distinguishes the two ways a capture can be mono. If the residual collapses at a
    non-zero lag, the channels are one stream split by sample parity, which is a
    deinterleaving fault in firmware. If it is already minimal at lag 0, the channels
    are a straight copy of each other.
    """
    best, best_sd = 0, float("inf")
    for lag in range(-max_lag, max_lag + 1):
        a = left[max_lag:-max_lag]
        b = right[max_lag + lag: len(right) - max_lag + lag]
        sd = float((a - b).std())
        if sd < best_sd:
            best, best_sd = lag, sd
    return best, best_sd


def channel_report(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)

    identical = float(np.mean(left == right))
    diff = left - right
    with np.errstate(invalid="ignore"):
        corr = float(np.corrcoef(left, right)[0, 1]) if left.std() and right.std() else np.nan

    return {
        "n": float(left.size),
        "sd_l": float(left.std()),
        "sd_r": float(right.std()),
        "identical_frac": identical,
        "corr": corr,
        "sd_diff": float(diff.std()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N captures")
    args = ap.parse_args()

    root = Path(args.root)
    stems = discover_stems(root)
    if not stems:
        raise SystemExit(f"{root}: no captures found")

    if args.limit:
        stems = stems[:args.limit]

    print(f"{'capture':<44} {'sd_L':>10} {'sd_R':>10} {'ident%':>7} {'corr':>7} "
          f"{'sd_diff':>10} {'lag':>4} {'sd@lag':>8}")
    for stem in stems:
        path = root / f"{stem}_sensor_audio.csv"
        if not path.exists():
            print(f"{stem:<44} {'-- no audio stream --':>46}")
            continue
        df = pd.read_csv(path, usecols=lambda c: c in ("left", "right"))
        if "left" not in df or "right" not in df:
            print(f"{stem:<44} {'-- single channel on disk --':>46}")
            continue
        left = df["left"].to_numpy(dtype=np.float64)
        right = df["right"].to_numpy(dtype=np.float64)
        r = channel_report(left, right)
        lag, sd_at_lag = best_lag(left, right)
        print(f"{stem:<44} {r['sd_l']:>10.1f} {r['sd_r']:>10.1f} "
              f"{100 * r['identical_frac']:>7.1f} {r['corr']:>7.3f} {r['sd_diff']:>10.1f} "
              f"{lag:>4d} {sd_at_lag:>8.2f}")


if __name__ == "__main__":
    main()
