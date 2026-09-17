"""Ask whether the microphone recovers the suck rate on its own.

Rate already has two independent paths, strain and the IMU. Acoustics would be a third,
and a third path matters more than a marginally better second one: it is what lets the
device keep reporting when the ring is the part that fails, which on the live units is
the part that keeps failing.

The question is worth settling before the microphones are reoriented. Pointing one
outward at the infant's chin is a bet that the acoustic channel earns its place, and the
bet is cheaper to evaluate on the captures already in hand than after a board revision.

What this can and cannot answer: the phantom has no infant, so there is nothing here that
swallows, and swallow detection cannot be assessed from this batch at all. Suck rate can,
because the pump drives the phantom at a commanded rate that the sidecar records.

    python scripts/acoustic_rate.py --root new_dataset/Stage1_Sweeps_20260816/data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milkeaze.data.pressure_events import estimate_cycle_rate_cpm
from milkeaze.data.rig_session import discover_stems, fill_dropped_strain, unwrap_device_us
from milkeaze.eval.baselines import STRAIN_POLARITY, acoustic_rate_cpm, strain_consensus_signal

BAND_CPM = (18.0, 90.0)


def device_fs_hz(time_us: np.ndarray) -> float:
    unwrapped, _ = unwrap_device_us(np.asarray(time_us, dtype=np.float64))
    step_us = float(np.median(np.diff(unwrapped)))
    return 1e6 / step_us if step_us > 0 else float("nan")


def strain_rate_cpm(root: Path, stem: str) -> float:
    """Rate from the strain ring, as the established path to compare the mic against."""
    df = pd.read_csv(root / f"{stem}_sensor_strain.csv")
    # this batch names the columns as the polarity map does, with no channel-index prefix
    names = [c for c in df.columns if c in STRAIN_POLARITY]
    if not names:
        return float("nan")

    fs = device_fs_hz(df["scan_t_us"].to_numpy())
    block = fill_dropped_strain(df[names].to_numpy(dtype=np.float64))
    signal = strain_consensus_signal(block, fs, names, STRAIN_POLARITY, BAND_CPM)
    return estimate_cycle_rate_cpm(signal, fs, BAND_CPM)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    stems = discover_stems(root)
    if not stems:
        raise SystemExit(f"{root}: no captures found")

    rows = []
    print(f"{'capture':<44} {'cmd':>6} {'mic':>7} {'strain':>7} {'mic_err':>8} {'str_err':>8}")
    for stem in stems:
        meta = json.loads((root / f"{stem}_sensor.json").read_text(encoding="utf-8"))
        commanded = (meta.get("run") or {}).get("cycle_rate_cpm")
        if commanded is None:
            continue
        commanded = float(commanded)

        audio = pd.read_csv(root / f"{stem}_sensor_audio.csv", usecols=["time_us", "left"])
        fs = device_fs_hz(audio["time_us"].to_numpy())
        mic_cpm = acoustic_rate_cpm(audio["left"].to_numpy(), fs, BAND_CPM)

        str_cpm = strain_rate_cpm(root, stem)
        rows.append((commanded, mic_cpm, str_cpm))
        print(f"{stem:<44} {commanded:>6.0f} {mic_cpm:>7.1f} {str_cpm:>7.1f} "
              f"{mic_cpm - commanded:>8.1f} {str_cpm - commanded:>8.1f}")

    if not rows:
        return
    arr = np.array(rows, dtype=np.float64)
    for label, col in (("mic", 1), ("strain", 2)):
        err = arr[:, col] - arr[:, 0]
        good = np.isfinite(err)
        within = np.mean(np.abs(err[good]) <= 2.0) * 100 if good.any() else float("nan")
        print(f"\n{label}: median abs error {np.nanmedian(np.abs(err)):.2f} cpm, "
              f"max {np.nanmax(np.abs(err)):.2f} cpm, within 2 cpm on {within:.0f}% of captures")


if __name__ == "__main__":
    main()
