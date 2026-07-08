"""Write a contract-valid session to disk (dev/test only).

Produces the on-disk layout the real pipeline expects, so ``run_pipeline.py`` and
``inspect_session.py`` can be exercised end to end without any real hardware data:

    <out_dir>/
        strain.csv   mic.csv   imu.csv   scale.csv
        meta.json
        events.csv   (optional — lets the classifier path be tested too)

This is fabricated signal for plumbing tests, NOT physiologically realistic data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from milkeaze.config import SensorConfig
from milkeaze.utils.logging import get_logger

log = get_logger("write_session")

MILK_DENSITY_G_PER_ML = 1.03
_SUCK_RATE_HZ = 1.2


def _stream(rate_hz: float, names: list[str], duration_s: float,
            base: float, noise: float, rng: np.random.Generator) -> pd.DataFrame:
    n = int(duration_s * rate_hz)
    t = np.arange(n) / rate_hz * 1000.0
    x = base + rng.normal(0, noise, size=(n, len(names)))
    df = pd.DataFrame({"t_ms": t})
    for i, nm in enumerate(names):
        df[nm] = x[:, i]
    return df


def _events(duration_s: float, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    t = 0.0
    while t < duration_s:
        t += rng.exponential(1.0 / _SUCK_RATE_HZ)
        if t >= duration_s:
            break
        rows.append((t * 1000.0, "suck"))
        if rng.random() < 0.3:  # a swallow tends to follow some sucks
            rows.append((t * 1000.0 + 120.0, "swallow"))
    return pd.DataFrame(rows, columns=["t_ms", "type"]).sort_values("t_ms")


def write_session(out_dir: str | Path, duration_s: float = 180.0, total_ml: float = 45.0,
                  with_events: bool = True, seed: int | None = None,
                  sensors: SensorConfig | None = None) -> Path:
    sensors = sensors or SensorConfig.load()
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rates = sensors.sample_rates

    _stream(rates["strain"], sensors.strain_channel_names(), duration_s,
            base=32000, noise=200, rng=rng).to_csv(out / "strain.csv", index=False)
    _stream(rates["imu"], sensors.imu_channel_names(), duration_s,
            base=0, noise=1000, rng=rng).to_csv(out / "imu.csv", index=False)
    _stream(rates["mic"], sensors.mic_channel_names(), duration_s,
            base=0, noise=500, rng=rng).to_csv(out / "mic.csv", index=False)

    # scale: monotonically increasing grams up to total_ml worth of milk
    n_scale = int(duration_s * 10)
    ts = np.arange(n_scale) / 10.0 * 1000.0
    grams = np.linspace(0, total_ml * MILK_DENSITY_G_PER_ML, n_scale)
    pd.DataFrame({"t_ms": ts, sensors.raw["scale"]["channel"]: grams}).to_csv(
        out / "scale.csv", index=False)

    meta = {
        "session_id": out.name,
        "mic_model": "MP34DT01-M",
        "sample_rates_hz": {k: float(v) for k, v in rates.items()},
        "adc_vref": sensors.raw["strain"]["adc"]["v_ref"],
        "mounting": "ring",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if with_events:
        _events(duration_s, rng).to_csv(out / "events.csv", index=False)

    log.info("wrote session %s (%.0fs, events=%s) to %s",
             meta["session_id"], duration_s, with_events, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a contract-valid dev session to disk.")
    ap.add_argument("--out", required=True, help="output session directory")
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--total-ml", type=float, default=45.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-events", action="store_true", help="omit events.csv")
    args = ap.parse_args()
    write_session(args.out, duration_s=args.duration, total_ml=args.total_ml,
                  with_events=not args.no_events, seed=args.seed)


if __name__ == "__main__":
    main()
