"""Per-event volume regression, replacing the run-level aggregates in the Stage 1 report.

The question this answers is the one Neal keeps asking: if the model predicts each suck
and you add them up over a feed, how far off is the total? Everything before this has
been run-level correlation, which flatters the model because a run is a single point.

Two constraints make the numbers here honest rather than optimistic:

**No rig pressure in the features.** The vacuum line is testbed instrumentation; the
product has strain, IMU and a mic. TN-013 §7 regresses against pressure depth, which is
not a feature we will ever have. Only the strain ring and the cycle period are used
here. Event *timing* still comes from the pressure detector, which is a dependency worth
retiring separately, but no pressure *amplitude* enters the model.

**Leave-one-run-out, and leave-one-level-out.** Runs drift over a day and each
(level, rate) cell has one run, so scoring on held-out events from a run the model has
already seen would mostly measure interpolation within a drift state. Leaving out whole
runs is the weakest defensible protocol; leaving out whole vacuum levels tests the
extrapolation that the L1 endpoint error appears to be made of.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milkeaze.data.rig_session import (  # noqa: E402
    SENSOR_BOARD, discover_stems, fill_dropped_strain, open_capture,
)
from milkeaze.eval.baselines import (  # noqa: E402
    fit_ridge, regression_scores, strain_consensus_events,
)
from milkeaze.utils.logging import get_logger  # noqa: E402

log = get_logger("volume_regression")

STRAIN_CHANNELS = ["Radial_1", "Radial_2", "Radial_3", "Radial_4",
                   "Arc_2_in", "Arc_2_out", "Arc_4_in", "Arc_4_out"]

#: half-width of the central difference used for the per-event flow target, matching
#: TN-013 so the two are comparable
FLOW_HALFWIN_S = 2.5

#: the run is scored from the moment 5 g has landed to 2 g short of the final total, so
#: the fill transient at the start and the drain-down tail at the end are both excluded
SUPPORT_START_G = 5.0
SUPPORT_END_MARGIN_G = 2.0


@dataclass
class RunEvents:
    stem: str
    level: int
    cpm: float
    features: np.ndarray     # (n_events, n_features)
    flow: np.ndarray         # (n_events,) g/min, central difference
    period_s: np.ndarray     # (n_events,)
    true_total_g: float


#: Flow is fitted in log space. Amplitude-to-flow is a power law, so a least-squares fit
#: on linear flow is dominated by the high-vacuum events — it drives L1 predictions
#: negative and the integrated feed total with them. The offset keeps the transform
#: defined through the near-empty cycles at L1, where the true flow really is ~0.
LOG_OFFSET_G_MIN = 1.0


def _to_log_flow(flow: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(flow, 0.0) + LOG_OFFSET_G_MIN)


def _from_log_flow(log_flow: np.ndarray, smearing: float = 1.0) -> np.ndarray:
    """Invert the log fit, with Duan's smearing factor.

    Exponentiating a least-squares fit of ``log y`` estimates the *median* of y, not its
    mean, so it sits low by roughly ``exp(sigma**2 / 2)``. Per event that is a rounding
    error; integrated over the ~120 events in a feed it is a systematic shortfall, and
    the uncorrected fit runs 2.2 g/min light on every event. The smearing factor is the
    mean of ``exp(residual)`` on the training split, which needs no normality assumption.
    """
    return np.maximum(np.exp(log_flow) * smearing - LOG_OFFSET_G_MIN, 0.0)


def _smearing(residuals: np.ndarray) -> float:
    return float(np.mean(np.exp(np.asarray(residuals, dtype=np.float64))))


def _monotonic_mass(grams: np.ndarray) -> np.ndarray:
    """The scale reads backwards occasionally; mass into the cup only goes up."""
    return np.maximum.accumulate(np.asarray(grams, dtype=np.float64))


def _time_at_mass(t_s: np.ndarray, mass: np.ndarray, target_g: float) -> float:
    idx = int(np.searchsorted(mass, target_g))
    return float(t_s[min(idx, t_s.size - 1)])


#: Where per-event timing comes from. ``reference`` is the rig pressure detector, which
#: needs the vacuum line; ``strain`` is the strain ring alone, which is what the product
#: will actually have. Swapping between them is the whole point of the flag: if the
#: accuracy holds, the pipeline no longer depends on bench instrumentation.
REFERENCE, STRAIN = "reference", "strain"


def _event_times_s(root: Path, stem: str, source: str,
                   strain_t_s: np.ndarray, raw_block: np.ndarray) -> np.ndarray:
    if source == REFERENCE:
        events = pd.read_csv(root / f"{stem}_events.csv")
        ok = events["quality"].astype(str).str.strip() == "ok"
        return np.sort(events.loc[ok, "t_ms"].to_numpy(np.float64) / 1000.0)

    found = strain_consensus_events(strain_t_s * 1000.0,
                                    fill_dropped_strain(raw_block), STRAIN_CHANNELS)
    return np.sort(found.events["t_ms"].to_numpy(np.float64) / 1000.0)


def _extract(root: Path, stem: str, events_source: str = REFERENCE) -> RunEvents | None:
    capture = open_capture(root, stem)
    clock = capture.clocks[SENSOR_BOARD]

    scale = pd.read_csv(root / f"{stem}_sensor_scale.csv", usecols=["host_mono_s", "grams"])
    # unparsed serial lines arrive as NaN; two captures in this batch have them, and a
    # single NaN poisons the running maximum for the rest of the run
    scale = scale.dropna(subset=["host_mono_s", "grams"])
    scale_t = scale["host_mono_s"].to_numpy(np.float64)
    mass = _monotonic_mass(scale["grams"].to_numpy(np.float64))

    total = float(mass[-1])
    t_start = _time_at_mass(scale_t, mass, SUPPORT_START_G)
    t_end = _time_at_mass(scale_t, mass, total - SUPPORT_END_MARGIN_G)
    if not (t_end > t_start):
        log.warning("%s: empty support window, skipping", stem)
        return None

    strain = pd.read_csv(root / f"{stem}_sensor_strain.csv",
                         usecols=["scan_t_us", *STRAIN_CHANNELS])
    strain_t = clock.to_host_s(strain["scan_t_us"].to_numpy(np.float64))
    raw = strain[STRAIN_CHANNELS].to_numpy(np.float64)

    order = np.argsort(strain_t)
    strain_t, raw = strain_t[order], raw[order]
    unique = np.concatenate(([True], np.diff(strain_t) > 0))
    strain_t, raw = strain_t[unique], raw[unique]

    # amplitudes are read with NaN-aware quantiles, so dropped samples stay masked here;
    # the detector gets them interpolated instead
    block = np.where(raw == 0.0, np.nan, raw)

    ev_t = _event_times_s(root, stem, events_source, strain_t, raw)
    ev_t = ev_t[(ev_t >= t_start) & (ev_t <= t_end)]
    if ev_t.size < 20:
        log.warning("%s: only %d usable %s events, skipping", stem, ev_t.size, events_source)
        return None

    period = float(np.median(np.diff(ev_t)))
    lo = np.searchsorted(strain_t, ev_t - period / 2.0)
    hi = np.searchsorted(strain_t, ev_t + period / 2.0)

    rows, flows, periods = [], [], []
    gaps = np.diff(ev_t, prepend=ev_t[0] - period)
    for i in range(ev_t.size):
        seg = block[lo[i]:hi[i]]
        if seg.shape[0] < 8:
            continue
        with np.errstate(invalid="ignore"):
            hi_q = np.nanquantile(seg, 0.99, axis=0)
            lo_q = np.nanquantile(seg, 0.01, axis=0)
        amp = hi_q - lo_q
        if not np.all(np.isfinite(amp)) or np.any(amp <= 0):
            continue

        m0 = np.interp(ev_t[i] - FLOW_HALFWIN_S, scale_t, mass)
        m1 = np.interp(ev_t[i] + FLOW_HALFWIN_S, scale_t, mass)
        flow = (m1 - m0) / (2 * FLOW_HALFWIN_S / 60.0)
        if not np.isfinite(flow) or flow < 0:
            continue

        rows.append(np.concatenate([np.log(amp), [np.log(max(gaps[i], 1e-3))]]))
        flows.append(flow)
        periods.append(gaps[i])

    if len(rows) < 20:
        return None

    return RunEvents(
        stem=stem,
        level=int(capture.vacuum_level),
        cpm=float(capture.cycle_rate_cpm or np.nan),
        features=np.asarray(rows),
        flow=np.asarray(flows),
        period_s=np.asarray(periods),
        true_total_g=float(np.interp(t_end, scale_t, mass) - np.interp(t_start, scale_t, mass)),
    )


def _predicted_total_g(flow_g_min: np.ndarray, period_s: np.ndarray) -> float:
    return float(np.sum(flow_g_min * period_s / 60.0))


def _fold_report(name: str, runs: list[RunEvents], folds: list[list[int]],
                 alpha: float) -> None:
    """Fit on the complement of each fold, score events and per-run totals."""
    per_event_pred, per_event_true = [], []
    run_errors, single_errors, mean_errors = [], [], []

    for fold in folds:
        train = [r for i, r in enumerate(runs) if i not in set(fold)]
        X_tr = np.vstack([r.features for r in train])
        y_tr = np.concatenate([r.flow for r in train])

        log_tr = _to_log_flow(y_tr)
        model = fit_ridge(X_tr, log_tr, alpha=alpha)
        smear = _smearing(log_tr - model.predict(X_tr))

        # single-channel power law, TN-013 style: log flow on one log amplitude
        best = int(np.argmax([abs(np.corrcoef(X_tr[:, c], log_tr)[0, 1])
                              for c in range(len(STRAIN_CHANNELS))]))
        slope, intercept = np.polyfit(X_tr[:, best], log_tr, 1)
        smear_single = _smearing(log_tr - (intercept + slope * X_tr[:, best]))

        for idx in fold:
            r = runs[idx]
            pred = _from_log_flow(model.predict(r.features), smear)
            per_event_pred.append(pred)
            per_event_true.append(r.flow)

            total = _predicted_total_g(pred, r.period_s)
            run_errors.append(100.0 * (total - r.true_total_g) / r.true_total_g)

            single = _from_log_flow(intercept + slope * r.features[:, best], smear_single)
            single_errors.append(100.0 * (_predicted_total_g(single, r.period_s)
                                          - r.true_total_g) / r.true_total_g)

            flat = np.full(r.flow.shape, float(y_tr.mean()))
            mean_errors.append(100.0 * (_predicted_total_g(flat, r.period_s)
                                        - r.true_total_g) / r.true_total_g)

    pred = np.concatenate(per_event_pred)
    true = np.concatenate(per_event_true)
    scores = regression_scores(pred, true)

    print(f"\n--- {name} ({len(folds)} folds, {true.size} held-out events) ---")
    print(f"  per-event flow:  {scores.summary()}  (g/min)")
    print("  per-feed total error:")
    for label, errs in (("ridge, 8 channels + rate", run_errors),
                        ("single-channel power law", single_errors),
                        ("predict the training mean", mean_errors)):
        signed = np.asarray(errs)
        a = np.abs(signed)
        within5 = 100.0 * float(np.mean(a <= 5.0))
        print(f"    {label:26}  median |{np.median(a):4.1f}%|  SD {signed.std(ddof=1):5.1f}%"
              f"  90th pct {np.percentile(a, 90):5.1f}%  worst {a.max():5.1f}%"
              f"  bias {signed.mean():+5.1f}%  within +-5%: {within5:.0f}%")
    return np.asarray(run_errors)


def _learning_curve(runs: list[RunEvents], alpha: float, repeats: int = 40,
                    seed: int = 0) -> None:
    """How per-feed error falls as training runs are added.

    This is what turns "can we reach +-5% in the lab" into a measurement rather than an
    opinion. If the curve is still falling at 34 runs then more of the same collection
    helps; if it has flattened, the limit is the rig and the mount and no amount of
    further lab data will move it.

    Training runs are drawn stratified by vacuum level, because an unstratified draw that
    happens to contain no L1 run has to extrapolate below everything it has seen and the
    resulting error says more about the draw than about the sample size.
    """
    rng = np.random.default_rng(seed)
    levels = sorted({r.level for r in runs})
    by_level = {lv: [i for i, r in enumerate(runs) if r.level == lv] for lv in levels}
    sizes = [6, 9, 12, 15, 18, 21, 24, 27, 30]

    print("\n--- learning curve: per-feed error against number of training runs ---")
    print("  train runs   median |%|   within +-5%   (median over "
          f"{repeats} stratified draws)")
    curve = []
    for n_train in sizes:
        med_errs, within = [], []
        for _ in range(repeats):
            # take a proportional share from each level, at least one where possible
            train_idx: list[int] = []
            for lv in levels:
                pool = by_level[lv]
                take = max(1, round(n_train * len(pool) / len(runs)))
                take = min(take, len(pool) - 1)  # always leave one out to test on
                train_idx += list(rng.choice(pool, size=take, replace=False))
            test_idx = [i for i in range(len(runs)) if i not in set(train_idx)]
            if len(train_idx) < 4 or not test_idx:
                continue

            X_tr = np.vstack([runs[i].features for i in train_idx])
            y_tr = np.concatenate([runs[i].flow for i in train_idx])
            log_tr = _to_log_flow(y_tr)
            model = fit_ridge(X_tr, log_tr, alpha=alpha)
            smear = _smearing(log_tr - model.predict(X_tr))

            errs = []
            for i in test_idx:
                r = runs[i]
                pred = _from_log_flow(model.predict(r.features), smear)
                total = _predicted_total_g(pred, r.period_s)
                errs.append(100.0 * abs(total - r.true_total_g) / r.true_total_g)
            med_errs.append(float(np.median(errs)))
            within.append(100.0 * float(np.mean(np.asarray(errs) <= 5.0)))

        med = float(np.median(med_errs))
        curve.append((n_train, med))
        print(f"  {n_train:10d}   {med:9.1f}%   {float(np.median(within)):10.0f}%")

    n = np.array([c[0] for c in curve], dtype=float)
    e = np.array([c[1] for c in curve], dtype=float)

    slope, intercept = np.polyfit(np.log(n), np.log(e), 1)
    print(f"\n  power-law fit: error ~ {np.exp(intercept):.1f} * n^({slope:+.2f})")
    for target in (100, 300, 1000):
        print(f"    extrapolated median error at {target:5d} runs: "
              f"{np.exp(intercept) * target ** slope:5.1f}%")

    # The more useful shape: error^2 = floor^2 + (estimation variance)/n. The floor is the
    # part no amount of further collection touches, which is what "best case in the lab
    # with this rig and this feature set" means.
    A = np.column_stack([np.ones_like(n), 1.0 / n])
    coeffs, *_ = np.linalg.lstsq(A, e ** 2, rcond=None)
    floor_sq, var_term = float(coeffs[0]), float(coeffs[1])
    print("\n  variance decomposition, error^2 = floor^2 + b^2/n:")
    if floor_sq <= 0:
        print("    floor fits to <=0, i.e. the curve is still sample-limited at 34 runs")
    else:
        floor = np.sqrt(floor_sq)
        print(f"    irreducible floor: {floor:.1f}% per feed")
        print(f"    sample-limited term at n=30: {np.sqrt(max(var_term, 0) / 30):.1f}%")
        print(f"    => of the {e[-1]:.1f}% at 30 runs, {floor:.1f}% does not respond to "
              "more runs of this kind")
        if floor > 5.0:
            print("    the floor is above 5%, so more lab runs alone cannot reach +-5%;")
            print("    the gain has to come from features/model, ground truth, or the rig")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="new_dataset/Stage1_Sweeps_20260816/data")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--learning-curve", action="store_true",
                        help="measure how per-feed error scales with training runs")
    parser.add_argument("--events", choices=(REFERENCE, STRAIN), default=REFERENCE,
                        help="source of per-event timing; 'strain' needs no vacuum line")
    parser.add_argument("--dump-errors", type=Path, default=None,
                        help="write per-run leave-one-run-out errors here, for paired "
                             "comparison between two event sources on the same runs")
    args = parser.parse_args()

    root = Path(args.root)
    runs: list[RunEvents] = []
    for stem in discover_stems(root):
        extracted = _extract(root, stem, args.events)
        if extracted is not None:
            runs.append(extracted)
        print(f"  read {stem}", file=sys.stderr)

    n_events = sum(r.flow.size for r in runs)
    print(f"\nevent timing from: {args.events}")
    print(f"{len(runs)} runs, {n_events} events, "
          f"{runs[0].features.shape[1]} features (8 log-amplitudes + log period)")
    print(f"feed totals span {min(r.true_total_g for r in runs):.1f} to "
          f"{max(r.true_total_g for r in runs):.1f} g")

    loro = _fold_report("leave one run out", runs, [[i] for i in range(len(runs))], args.alpha)

    levels = sorted({r.level for r in runs})
    by_level = [[i for i, r in enumerate(runs) if r.level == lv] for lv in levels]
    _fold_report("leave one vacuum level out", runs, by_level, args.alpha)

    print("\n--- leave-one-run-out per-feed error, by vacuum level ---")
    print("  level   n runs   median |%|   signed mean %")
    for lv in levels:
        idx = [i for i, r in enumerate(runs) if r.level == lv]
        errs = loro[idx]
        print(f"  L{lv}      {len(idx):5d}   {np.median(np.abs(errs)):9.1f}%   {errs.mean():+13.1f}%")

    if args.dump_errors is not None:
        args.dump_errors.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "stem": [r.stem for r in runs],
            "level": [r.level for r in runs],
            "events": args.events,
            "signed_err_pct": loro,
            "n_events": [r.flow.size for r in runs],
        }).to_csv(args.dump_errors, index=False)
        print(f"\nwrote {args.dump_errors}")

    if args.learning_curve:
        _learning_curve(runs, args.alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
