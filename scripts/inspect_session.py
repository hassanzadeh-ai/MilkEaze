"""Dataset sanity-check: run one session through the pipeline and summarize it.

Usually the first thing to do when picking up the data: confirm the session is
contract-valid and eyeball the per-channel stats, window count, volume target, and
(if events.csv is present) the suck/swallow/silence class balance.

    python scripts/inspect_session.py --session path/to/session_dir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from milkeaze.config import SensorConfig
from milkeaze.pipeline import process_session


def _channel_names(sensors: SensorConfig) -> list[str]:
    # unified-grid frame layout: 8 strain + 2 mic-envelope + 6 imu
    mic_env = [f"{m}_env" for m in sensors.mic_channel_names()]
    return sensors.strain_channel_names() + mic_env + sensors.imu_channel_names()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="session directory")
    args = ap.parse_args()

    sensors = SensorConfig.load()
    proc = process_session(args.session, sensors=sensors)

    print("\n=== session:", proc.session_id, "===")
    print(f"windows kept : {len(proc.windows)}")
    print(f"grid         : {proc.grid_hz:g} Hz")
    print(f"frames       : {proc.frames.shape}  (seq, channels, time)")
    print(f"hand features: {proc.hand.shape[1] if proc.hand.ndim == 2 else 0} dims/window")

    # per-channel stats over all windows/time
    names = _channel_names(sensors)
    flat = proc.frames.transpose(1, 0, 2).reshape(proc.frames.shape[1], -1)
    print("\nper-channel (unified grid):")
    print(f"  {'channel':<14}{'mean':>12}{'std':>12}{'min':>12}{'max':>12}")
    for i, nm in enumerate(names):
        c = flat[i]
        print(f"  {nm:<14}{c.mean():>12.3f}{c.std():>12.3f}{c.min():>12.3f}{c.max():>12.3f}")

    # volume target
    if proc.volume is not None:
        print(f"\nvolume target: total {proc.volume.sum():.2f} mL "
              f"({proc.volume.mean():.3f} mL/window mean)")
    else:
        print("\nvolume target: none (no scale.csv)")

    # class balance
    if proc.labels is not None:
        counts = np.bincount(proc.labels, minlength=len(sensors.classes))
        print("\nwindow class distribution:")
        for cls_i, name in enumerate(sensors.classes):
            n = int(counts[cls_i])
            pct = 100.0 * n / max(len(proc.labels), 1)
            print(f"  {name:<10}{n:>6}  ({pct:5.1f}%)")
    else:
        print("\nwindow class distribution: unavailable (no events.csv) "
              "-> classifier has no targets for this session")


if __name__ == "__main__":
    main()
