"""Can we find sucks in the strain ring without the vacuum line?

Every label we have comes from rig pressure, which exists only on the bench. Annie's and
Lauren's sessions have no rig board, so unless cycles can be recovered from the ring the
pipeline has nothing to segment on at home and the volume integration has no periods to
multiply by.

Scored against the pressure detector as reference, on the same support window, per run.
Both polarity hypotheses are scored because the physical channel map is still unverified
and a suck may deflect a given channel either way; guessing lands on F1 near 1 or near 0
with nothing between, so the winner is reported rather than assumed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milkeaze.data.pressure_events import DetectorConfig  # noqa: E402
from milkeaze.data.rig_session import SENSOR_BOARD, discover_stems, open_capture  # noqa: E402
from milkeaze.eval.baselines import strain_event_candidates  # noqa: E402
from milkeaze.eval.events import match_events, rate_cpm, tolerance_from_period  # noqa: E402
from milkeaze.utils.logging import get_logger  # noqa: E402

log = get_logger("strain_events")

STRAIN_CHANNELS = ["Radial_1", "Radial_2", "Radial_3", "Radial_4",
                   "Arc_2_in", "Arc_2_out", "Arc_4_in", "Arc_4_out"]


def _reference_events(root: Path, stem: str) -> pd.DataFrame:
    events = pd.read_csv(root / f"{stem}_events.csv")
    ok = events["quality"].astype(str).str.strip() == "ok"
    return events.loc[ok, ["t_ms", "depth_psi"]].sort_values("t_ms").reset_index(drop=True)


def _strain_block(root: Path, stem: str, capture) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(root / f"{stem}_sensor_strain.csv",
                        usecols=["scan_t_us", *STRAIN_CHANNELS])
    t_ms = capture.clocks[SENSOR_BOARD].to_host_s(
        frame["scan_t_us"].to_numpy(np.float64)) * 1000.0
    block = frame[STRAIN_CHANNELS].to_numpy(np.float64)

    order = np.argsort(t_ms)
    t_ms, block = t_ms[order], block[order]

    # zeros are dropped samples in this format; carry the last good value forward so the
    # detector sees a continuous trace rather than spikes to zero
    block[block == 0.0] = np.nan
    frame = pd.DataFrame(block).ffill().bfill()
    return t_ms, frame.to_numpy(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="new_dataset/Stage1_Sweeps_20260816/data")
    args = parser.parse_args()
    root = Path(args.root)

    rows = []
    missed_depth: list[np.ndarray] = []
    for stem in discover_stems(root):
        capture = open_capture(root, stem)
        reference_frame = _reference_events(root, stem)
        reference = reference_frame["t_ms"].to_numpy(np.float64)
        if reference.size < 20:
            continue
        t_ms, block = _strain_block(root, stem, capture)

        # score on the window the reference covers, so strain is not penalised for cycles
        # during the fill transient and drain tail where no reference events exist
        window = (t_ms >= reference.min()) & (t_ms <= reference.max())
        if window.sum() < 500:
            log.warning("%s: only %d strain samples in the reference window", stem,
                        int(window.sum()))
            continue

        tolerance = tolerance_from_period(reference)
        try:
            candidates = strain_event_candidates(t_ms[window], block[window],
                                                 DetectorConfig())
        except ValueError as exc:
            log.warning("%s: %s", stem, exc)
            continue

        scored = []
        for rank, result in enumerate(candidates):
            pred = result.events["t_ms"].to_numpy(np.float64)
            scored.append((match_events(pred, reference, tolerance), result, rank))
        best = max(scored, key=lambda s: s[0].f1)
        match, result, rank = best

        # what a home session would actually get: no reference to score against, so the
        # sign has to come from period steadiness alone
        deployable, deployable_result, _ = scored[0]

        # are the cycles strain misses the weak ones? a miss on a shallow suck costs far
        # less volume than a miss on a deep one, so the count error and the volume error
        # are not the same number
        depth = reference_frame["depth_psi"].to_numpy(np.float64)
        found_true = np.zeros(reference.size, dtype=bool)
        if match.pairs.size:
            found_true[match.pairs[:, 1]] = True
        if (~found_true).any() and np.isfinite(depth).all():
            ranks = pd.Series(depth).rank(pct=True).to_numpy()
            missed_depth.append(ranks[~found_true])

        rows.append({
            "f1_deployable": deployable.f1,
            "mad_ms_deployable": deployable.mad_ms,
            "count_err_pct_deployable": deployable.count_error_pct,
            "rate_pred_deployable": deployable_result.cycle_rate_cpm,
            "stem": stem,
            "level": capture.vacuum_level,
            "cpm": capture.cycle_rate_cpm,
            "n_true": match.n_true,
            "f1": match.f1,
            "precision": match.precision,
            "recall": match.recall,
            "count_err_pct": match.count_error_pct,
            "bias_ms": match.bias_ms,
            "mad_ms": match.mad_ms,
            "tol_ms": tolerance,
            "rate_true": rate_cpm(reference),
            "rate_pred": result.cycle_rate_cpm,
            "polarity_was_steadiest": rank == 0,
            "period_cv": result.cycle_rate_cv,
        })
        print(f"  {stem[:34]:36} F1={match.f1:.3f} count={match.count_error_pct:+6.1f}% "
              f"MAD={match.mad_ms:6.1f}ms", file=sys.stderr)

    table = pd.DataFrame(rows)
    if table.empty:
        print("no runs scored")
        return 1

    print(f"\n{len(table)} runs scored, tolerance = quarter period "
          f"({table['tol_ms'].min():.0f}-{table['tol_ms'].max():.0f} ms)\n")

    print("=== overall, against the pressure detector ===")
    for col, label, unit in (("f1", "F1", ""), ("precision", "precision", ""),
                             ("recall", "recall", ""), ("mad_ms", "timing MAD", " ms")):
        v = table[col]
        print(f"  {label:12} median {v.median():6.3f}{unit}   "
              f"worst {v.min() if col != 'mad_ms' else v.max():6.3f}{unit}")

    abs_count = table["count_err_pct"].abs()
    print(f"\n  cycle-count error, which is what volume integration inherits:")
    print(f"    median |{abs_count.median():.1f}%|   90th pct {abs_count.quantile(0.9):.1f}%"
          f"   worst {abs_count.max():.1f}%")
    print(f"    runs within +-2%: {100 * (abs_count <= 2).mean():.0f}%   "
          f"within +-5%: {100 * (abs_count <= 5).mean():.0f}%")

    signed = table["count_err_pct"]
    print(f"    signed: mean {signed.mean():+.1f}%   median {signed.median():+.1f}%   "
          f"{100 * (signed < 0).mean():.0f}% of runs undercount")
    print("    (a one-sided miscount is a bias, not noise: it will not average out over a "
          "day the way a symmetric error would)")

    if missed_depth:
        pooled = np.concatenate(missed_depth)
        print(f"\n  the {pooled.size} missed cycles sit at depth percentile "
              f"{100 * np.median(pooled):.0f} (median) within their own run;")
        print(f"    {100 * (pooled < 0.25).mean():.0f}% of them are in the shallowest "
              f"quartile, {100 * (pooled > 0.75).mean():.0f}% in the deepest")
        print("    (a miss on a shallow suck costs little volume, so the count bias and "
              "the volume bias are not the same number)")

    rate_err = 100 * (table["rate_pred"] - table["rate_true"]).abs() / table["rate_true"]
    print(f"\n  cycle-rate error: median {rate_err.median():.2f}%   worst {rate_err.max():.2f}%")

    print(f"\n=== the polarity problem: oracle vs what a home session gets ===")
    print(f"  the steadier-period hypothesis was also the better-matching one on "
          f"{100 * table['polarity_was_steadiest'].mean():.0f}% of runs\n")
    print("  metric                 oracle (best sign)   deployable (steadiest sign)")
    for col, label, fmt in (("f1", "F1 median", "{:.3f}"),
                            ("mad_ms", "timing MAD median", "{:.1f} ms"),
                            ("count_err_pct", "|count err| median", "{:.1f}%")):
        a = table[col].abs().median() if col == "count_err_pct" else table[col].median()
        b_col = table[f"{col}_deployable"]
        b = b_col.abs().median() if col == "count_err_pct" else b_col.median()
        print(f"  {label:22} {fmt.format(a):>18}   {fmt.format(b):>27}")
    rate_dep = (100 * (table["rate_pred_deployable"] - table["rate_true"]).abs()
                / table["rate_true"])
    print(f"  {'rate err median':22} {rate_err.median():17.2f}%   {rate_dep.median():26.2f}%")
    print("\n  counts and rates are polarity-independent and transfer as-is. Per-event")
    print("  timing does not, until the sign of each channel is settled on the bench.")

    print("\n=== by vacuum level ===")
    print("  level  n   median F1   median |count err|   median MAD")
    for level, group in table.groupby("level"):
        print(f"  L{level}     {len(group):2d}   {group['f1'].median():9.3f}   "
              f"{group['count_err_pct'].abs().median():17.1f}%   "
              f"{group['mad_ms'].median():8.1f} ms")

    worst = table.nsmallest(5, "f1")[["stem", "level", "f1", "count_err_pct", "mad_ms"]]
    print("\n=== weakest five runs ===")
    print(worst.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
