"""Can we find sucks in the strain ring without the vacuum line?

Every label we have comes from rig pressure, which exists only on the bench. Annie's and
Lauren's sessions have no rig board, so unless cycles can be recovered from the ring the
pipeline has nothing to segment on at home and the volume integration has no periods to
multiply by.

The blocker used to be the channel polarity. Whether suction deflects a given channel
positive or negative depends on the physical channel map, and the sign is not
identifiable from strain alone — a roughly sinusoidal cycle has an equally steady period
either way up, and the two hypotheses differ by half a cycle, so guessing lands on F1
near 1 or near 0 with nothing in between. Choosing by period steadiness, the only signal
available without a reference, picked the matching polarity on 18% of runs.

It did not need bench time after all. The 34 captures here carry pressure-derived
reference events, and the sign that matches the reference *is* the sign. This script
sweeps all eight channels against both hypotheses over the whole batch, which turns the
polarity from a per-run coin flip into a measured constant — see
:data:`milkeaze.eval.baselines.STRAIN_POLARITY`, which this script is the source of.

It then scores the detector that would actually ship: a consensus over whichever
channels are carrying rhythm, which needs no named channel and degrades as channels fail
rather than stopping. That last property is the point, given that bend-sensor channels
have been failing after the silicone overmold.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milkeaze.data.pressure_events import DetectorConfig, detect_suck_events  # noqa: E402
from milkeaze.data.rig_session import (  # noqa: E402
    SENSOR_BOARD, discover_stems, fill_dropped_strain, open_capture,
)
from milkeaze.eval.baselines import strain_consensus_events  # noqa: E402
from milkeaze.eval.events import (  # noqa: E402
    EventMatch, match_events, rate_cpm, tolerance_from_period,
)
from milkeaze.utils.logging import get_logger  # noqa: E402

log = get_logger("strain_events")

STRAIN_CHANNELS = ["Radial_1", "Radial_2", "Radial_3", "Radial_4",
                   "Arc_2_in", "Arc_2_out", "Arc_4_in", "Arc_4_out"]

SIGNS = (1.0, -1.0)

#: A run needs this many reference events before it is worth scoring; below it the
#: period estimate that sets the matching tolerance is itself unreliable.
MIN_REFERENCE_EVENTS = 20

CONSENSUS = "consensus"


@dataclass
class Run:
    stem: str
    level: int
    cpm: float
    t_ms: np.ndarray                 # strain timestamps, host-mono ms
    block: np.ndarray                # (n_time, 8), dropped samples interpolated
    ref_t_ms: np.ndarray             # pressure reference events, host-mono ms
    ref_depth_psi: np.ndarray        # suction depth of each reference event


@dataclass
class Score:
    stem: str
    level: int
    channel: str
    sign: float
    f1: float
    precision: float
    recall: float
    count_err_pct: float
    bias_ms: float
    mad_ms: float
    pred_rate_cpm: float
    ref_rate_cpm: float


def _load(root: Path, stem: str) -> Run | None:
    capture = open_capture(root, stem)
    clock = capture.clocks[SENSOR_BOARD]

    events = pd.read_csv(root / f"{stem}_events.csv")
    ok = events["quality"].astype(str).str.strip() == "ok"
    reference = events.loc[ok].sort_values("t_ms")
    if len(reference) < MIN_REFERENCE_EVENTS:
        log.warning("%s: %d reference events, skipping", stem, len(reference))
        return None

    ref_t = reference["t_ms"].to_numpy(np.float64)
    depth = (reference["depth_psi"].to_numpy(np.float64)
             if "depth_psi" in reference else np.full(ref_t.size, np.nan))

    strain = pd.read_csv(root / f"{stem}_sensor_strain.csv",
                         usecols=["scan_t_us", *STRAIN_CHANNELS])
    t_ms = clock.to_host_s(strain["scan_t_us"].to_numpy(np.float64)) * 1000.0
    block = strain[STRAIN_CHANNELS].to_numpy(np.float64)

    order = np.argsort(t_ms)
    t_ms, block = t_ms[order], block[order]
    unique = np.concatenate(([True], np.diff(t_ms) > 0))
    t_ms, block = t_ms[unique], block[unique]

    # score on the window the reference covers, so strain is not penalised for cycles
    # during the fill transient and drain tail where no reference events exist; one
    # period of margin keeps a cycle at either end from being clipped mid-detection
    period_ms = float(np.median(np.diff(ref_t)))
    span = (t_ms >= ref_t[0] - period_ms) & (t_ms <= ref_t[-1] + period_ms)
    if span.sum() < 500:
        log.warning("%s: only %d strain samples in the reference window", stem,
                    int(span.sum()))
        return None

    return Run(
        stem=stem,
        level=int(capture.vacuum_level),
        cpm=float(capture.cycle_rate_cpm or np.nan),
        t_ms=t_ms[span],
        block=fill_dropped_strain(block[span]),
        ref_t_ms=ref_t,
        ref_depth_psi=depth,
    )


def _score_pred(run: Run, pred: np.ndarray, channel: str,
                sign: float) -> tuple[Score, EventMatch] | None:
    inside = (pred >= run.ref_t_ms[0]) & (pred <= run.ref_t_ms[-1])
    pred = pred[inside]
    if pred.size == 0:
        return None

    m = match_events(pred, run.ref_t_ms, tolerance_from_period(run.ref_t_ms))
    return Score(
        stem=run.stem, level=run.level, channel=channel, sign=sign,
        f1=m.f1, precision=m.precision, recall=m.recall,
        count_err_pct=m.count_error_pct, bias_ms=m.bias_ms, mad_ms=m.mad_ms,
        pred_rate_cpm=rate_cpm(pred), ref_rate_cpm=rate_cpm(run.ref_t_ms),
    ), m


def _score_channel(run: Run, index: int, sign: float,
                   config: DetectorConfig) -> Score | None:
    channel = STRAIN_CHANNELS[index]
    centred = run.block[:, index] - run.block[:, index].mean()
    try:
        found = detect_suck_events(run.t_ms, sign * centred, config)
    except ValueError as exc:
        log.debug("%s %s sign=%+.0f: %s", run.stem, channel, sign, exc)
        return None
    if found.n_events == 0:
        return None

    scored = _score_pred(run, found.events["t_ms"].to_numpy(np.float64), channel, sign)
    return None if scored is None else scored[0]


def _score_consensus(run: Run, channels: list[str], label: str,
                     config: DetectorConfig) -> tuple[Score, EventMatch] | None:
    keep = [STRAIN_CHANNELS.index(name) for name in channels]
    try:
        found = strain_consensus_events(run.t_ms, run.block[:, keep], channels, config)
    except ValueError as exc:
        log.debug("%s consensus %s: %s", run.stem, label, exc)
        return None
    if found.n_events == 0:
        return None
    return _score_pred(run, found.events["t_ms"].to_numpy(np.float64), label, 0.0)


def _table(scores: list[Score]) -> pd.DataFrame:
    return pd.DataFrame([s.__dict__ for s in scores])


def _summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Median over runs, because a single failed run should not set the headline.

    Ranked on F1 and then on timing spread: the leading channels tie at F1 0.998, so
    counting accuracy alone cannot separate them and per-event timing decides.
    """
    out = df.groupby(by, dropna=False).agg(
        n_runs=("f1", "size"),
        f1=("f1", "median"),
        worst_f1=("f1", "min"),
        abs_count_err_pct=("count_err_pct", lambda s: float(np.median(np.abs(s)))),
        count_err_pct=("count_err_pct", "median"),
        bias_ms=("bias_ms", "median"),
        mad_ms=("mad_ms", "median"),
    )
    return out.sort_values(["f1", "mad_ms"], ascending=[False, True])


def _report_polarity(df: pd.DataFrame) -> None:
    print("\n=== is the polarity a measured constant, or a per-run coin flip? ===")
    print("  a channel whose losing sign scores F1 0.000 while still counting the right")
    print("  number of cycles is exactly half a cycle out, which is the expected failure\n")
    for channel in STRAIN_CHANNELS:
        sub = df[df["channel"] == channel]
        if sub.empty:
            continue
        winners = sub.loc[sub.groupby("stem")["f1"].idxmax()]
        losers = sub.loc[sub.groupby("stem")["f1"].idxmin()]
        negative = int((winners["sign"] < 0).sum())
        # the detector finds troughs, so a winning sign of -1 means suction deflects the
        # channel positive
        reads = "positive" if negative > len(winners) / 2 else "negative"
        print(f"  {channel:12s} suction reads {reads:8s} on "
              f"{max(negative, len(winners) - negative):2d}/{len(winners):2d} runs   "
              f"winner F1 {winners['f1'].median():.3f}   "
              f"loser F1 {losers['f1'].median():.3f}")


def _report_missed_cycles(matches: list[tuple[Run, EventMatch]]) -> None:
    """Are the cycles the detector misses the cheap ones?

    A miss on a shallow suck costs far less volume than a miss on a deep one, so the
    count error and the volume error are not the same number.
    """
    pooled = []
    for run, match in matches:
        found = np.zeros(run.ref_t_ms.size, dtype=bool)
        if match.pairs.size:
            found[match.pairs[:, 1]] = True
        if (~found).any() and np.isfinite(run.ref_depth_psi).all():
            ranks = pd.Series(run.ref_depth_psi).rank(pct=True).to_numpy()
            pooled.append(ranks[~found])
    if not pooled:
        return

    depths = np.concatenate(pooled)
    print(f"\n=== the {depths.size} missed cycles, by suction depth within their run ===")
    print(f"  median depth percentile {100 * np.median(depths):.0f}; "
          f"{100 * (depths < 0.25).mean():.0f}% in the shallowest quartile, "
          f"{100 * (depths > 0.75).mean():.0f}% in the deepest")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="new_dataset/Stage1_Sweeps_20260816/data")
    parser.add_argument("--out", type=Path, default=Path("reports/strain_events_scores.csv"))
    args = parser.parse_args()
    root = Path(args.root)
    config = DetectorConfig()

    runs: list[Run] = []
    for stem in discover_stems(root):
        try:
            run = _load(root, stem)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            log.warning("%s: %s", stem, exc)
            continue
        if run is not None:
            runs.append(run)

    if not runs:
        print("no runs scored")
        return 1
    log.info("scoring %d captures x %d channels x 2 signs", len(runs), len(STRAIN_CHANNELS))

    scores = [s for run in runs for i in range(len(STRAIN_CHANNELS)) for sign in SIGNS
              if (s := _score_channel(run, i, sign, config)) is not None]
    df = _table(scores)

    pd.set_option("display.width", 220, "display.max_columns", 40)
    fmt = lambda v: f"{v:8.3f}"  # noqa: E731

    tolerances = [tolerance_from_period(r.ref_t_ms) for r in runs]
    print(f"\n{len(runs)} runs scored, tolerance = quarter period "
          f"({min(tolerances):.0f}-{max(tolerances):.0f} ms)")

    print("\n=== every channel x sign, median over runs ===")
    ranked = _summarise(df, ["channel", "sign"])
    print(ranked.to_string(float_format=fmt))

    _report_polarity(df)

    channel, sign = ranked.index[0]
    print(f"\n=== best single channel: {channel}, sign {sign:+.0f}, by vacuum level ===")
    single = df[(df["channel"] == channel) & (df["sign"] == sign)]
    print(_summarise(single, ["level"]).sort_index().to_string(float_format=fmt))

    # the shippable detector: no named channel, no polarity left to guess
    scored = [(run, out) for run in runs
              if (out := _score_consensus(run, STRAIN_CHANNELS, CONSENSUS, config))]
    consensus = _table([out[0] for _, out in scored])

    print("\n=== consensus of the healthy channels, by vacuum level ===")
    print(_summarise(consensus, ["level"]).sort_index().to_string(float_format=fmt))

    print("\n=== best single channel vs consensus, over all runs ===")
    print(_summarise(pd.concat([single.assign(channel=f"{channel} alone"), consensus]),
                     ["channel"]).to_string(float_format=fmt))

    # what a field failure of one channel costs, which is the question the overmold
    # reliability problem actually poses
    print("\n=== consensus with one channel removed (field-failure robustness) ===")
    dropped: list[Score] = []
    for name in STRAIN_CHANNELS:
        rest = [c for c in STRAIN_CHANNELS if c != name]
        dropped += [out[0] for run in runs
                    if (out := _score_consensus(run, rest, f"-{name}", config))]
    if dropped:
        print(_summarise(_table(dropped), ["channel"]).to_string(float_format=fmt))

    signed = consensus["count_err_pct"]
    print(f"\n=== cycle count, which is what volume integration inherits ===")
    print(f"  signed: mean {signed.mean():+.2f}%   median {signed.median():+.2f}%   "
          f"{100 * (signed < 0).mean():.0f}% of runs undercount")
    print("  (a one-sided miscount is a bias, not noise: it will not average out over a "
          "day\n   the way a symmetric error would)")
    print(f"  precision {consensus['precision'].median():.3f} against recall "
          f"{consensus['recall'].median():.3f} at the median, so the detector misses "
          f"cycles\n   rather than inventing them")

    _report_missed_cycles([(run, out[1]) for run, out in scored])

    rel = 100.0 * (consensus["pred_rate_cpm"] - consensus["ref_rate_cpm"]) / consensus["ref_rate_cpm"]
    print("\n=== cycle rate of the consensus (what a rate readout shows) ===")
    print(f"  median {rel.median():+.2f}%   90th pct |err| "
          f"{np.percentile(np.abs(rel), 90):.2f}%   worst {np.abs(rel).max():.2f}%")

    print("\n=== weakest five runs, consensus ===")
    print(consensus.nsmallest(5, "f1")[
        ["stem", "level", "f1", "count_err_pct", "bias_ms", "mad_ms"]
    ].to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([df, consensus]).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
