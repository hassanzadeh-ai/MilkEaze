"""Paired comparison of two event-timing sources on the same runs.

Comparing the summary lines of two ``volume_regression.py`` runs is the wrong test.
Per-feed error has an SD of 17% across runs, so any two conditions differ by several
points on the median for free, and with 34 runs the "within +-5%" figure moves a whole
8 points when three runs cross the threshold. The runs are the same in both conditions,
so the comparison should be paired and the question should be asked about the *difference*
per run rather than about two independent-looking medians.

Reports the paired difference with a bootstrap interval and a Wilcoxon signed-rank test,
which needs no normality assumption and is the right test for 34 paired errors.

Usage::

    python scripts/volume_regression.py --events reference --dump-errors reports/a.csv
    python scripts/volume_regression.py --events strain    --dump-errors reports/b.csv
    python scripts/compare_event_sources.py reports/a.csv reports/b.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BOOTSTRAP_DRAWS = 20000
SEED = 20260904


def _bootstrap_median_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    draws = rng.choice(values, size=(BOOTSTRAP_DRAWS, values.size), replace=True)
    medians = np.median(draws, axis=1)
    return (float(np.quantile(medians, alpha / 2)),
            float(np.quantile(medians, 1 - alpha / 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    a = pd.read_csv(args.baseline)
    b = pd.read_csv(args.candidate)
    merged = a.merge(b, on=["stem", "level"], suffixes=("_a", "_b"))
    if merged.empty:
        raise SystemExit("no runs in common between the two files")
    if len(merged) != len(a) or len(merged) != len(b):
        print(f"note: {len(a)} and {len(b)} runs, {len(merged)} paired")

    name_a = str(a["events"].iloc[0])
    name_b = str(b["events"].iloc[0])
    abs_a = merged["signed_err_pct_a"].abs().to_numpy(np.float64)
    abs_b = merged["signed_err_pct_b"].abs().to_numpy(np.float64)
    diff = abs_b - abs_a

    print(f"{len(merged)} paired runs: '{name_b}' minus '{name_a}'\n")
    for label, values in ((name_a, abs_a), (name_b, abs_b)):
        print(f"  {label:12s} median |err| {np.median(values):5.2f}%   "
              f"mean {values.mean():5.2f}%   90th pct {np.percentile(values, 90):5.2f}%   "
              f"within +-5% {int((values <= 5).sum()):2d}/{values.size}")

    lo, hi = _bootstrap_median_ci(diff)
    print(f"\n  paired difference in |err|: median {np.median(diff):+.2f} pp, "
          f"95% CI [{lo:+.2f}, {hi:+.2f}] pp")
    print(f"  runs better with '{name_b}': {int((diff < 0).sum())}, "
          f"worse: {int((diff > 0).sum())}")

    stat, p = wilcoxon(abs_b, abs_a)
    print(f"  Wilcoxon signed-rank: W={stat:.0f}, p={p:.3f}")
    verdict = ("no detectable difference" if p >= 0.05 else
               f"'{name_b}' is {'better' if np.median(diff) < 0 else 'worse'}")
    print(f"  => {verdict} at this sample size")

    print("\n  per level, median |err|")
    for level, grp in merged.groupby("level"):
        pa = grp["signed_err_pct_a"].abs().median()
        pb = grp["signed_err_pct_b"].abs().median()
        print(f"    L{int(level)}  n={len(grp)}  {name_a} {pa:5.1f}%   "
              f"{name_b} {pb:5.1f}%   diff {pb - pa:+5.1f} pp")

    n_a, n_b = merged["n_events_a"].sum(), merged["n_events_b"].sum()
    print(f"\n  events scored: {name_a} {n_a}, {name_b} {n_b} "
          f"({100.0 * (n_b - n_a) / n_a:+.1f}%)")


if __name__ == "__main__":
    main()
