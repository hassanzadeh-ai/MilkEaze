"""Scan every capture, every device-timestamped file, for uint32 microsecond wraps.

``device_ts_us`` / ``time_us`` / ``scan_t_us`` are 32-bit microsecond counters, so they
roll over every 2**32 us = 4294.967 s (71.6 min) of board uptime. Any run that straddles
a boundary has a discontinuity, and a clock fit over it produces a nonsense slope rather
than a slightly worse one.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

WRAP_US = 2 ** 32
ROOTS = sys.argv[1:] or ["new_dataset/Stage1_Sweeps_20260816/data", "new_dataset/20260718"]

SUFFIXES = [
    ("_sensor_sync.csv", "device_ts_us"),
    ("_rig_sync.csv", "device_ts_us"),
    ("_sensor_strain.csv", "scan_t_us"),
    ("_sensor_imu.csv", "time_us"),
    ("_sensor_audio.csv", "time_us"),
    ("_rig_pressure.csv", "time_us"),
    ("_rig_temp.csv", "time_us"),
]

for root in ROOTS:
    stems = sorted(os.path.basename(p).replace("_session.json", "")
                   for p in glob.glob(os.path.join(root, "*_session.json")))
    print(f"\n===== {root}  ({len(stems)} captures) =====")
    hits = []
    for stem in stems:
        for suffix, col in SUFFIXES:
            path = os.path.join(root, stem + suffix)
            if not os.path.exists(path):
                continue
            try:
                series = pd.read_csv(path, usecols=[col])[col].to_numpy(np.float64)
            except (ValueError, KeyError):
                continue
            if series.size < 2:
                continue
            drops = np.where(np.diff(series) < -WRAP_US / 2)[0]
            small = np.where((np.diff(series) < 0) & (np.diff(series) >= -WRAP_US / 2))[0]
            if drops.size or small.size:
                hits.append((stem, suffix, col, drops.size, small.size,
                             float(series.max()), series.size))

    if not hits:
        print("  no wraps and no backward steps found")
        continue

    print(f"  {len(hits)} file(s) with a backward timestamp step:\n")
    print(f"  {'capture':34} {'file':22} {'wraps':>6} {'other back-steps':>17} {'max value':>16}")
    for stem, suffix, _col, nwrap, nsmall, mx, n in hits:
        print(f"  {stem[:33]:34} {suffix[1:][:21]:22} {nwrap:6d} {nsmall:17d} "
              f"{mx:16.0f}  ({n} rows, {100 * mx / WRAP_US:.1f}% of wrap)")

    print("\n  headroom check: how close each capture's streams run to the boundary")
    near = []
    for stem in stems:
        path = os.path.join(root, stem + "_rig_pressure.csv")
        if not os.path.exists(path):
            continue
        series = pd.read_csv(path, usecols=["time_us"])["time_us"].to_numpy(np.float64)
        frac = 100.0 * series.max() / WRAP_US
        near.append((frac, stem))
    near.sort(reverse=True)
    for frac, stem in near[:6]:
        print(f"    {frac:6.1f}% of the wrap boundary   {stem}")
    print(f"    {sum(1 for f, _ in near if f > 90)} capture(s) above 90%, i.e. within "
          f"7 min of a rollover")
