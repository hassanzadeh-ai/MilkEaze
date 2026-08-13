"""Derive per-event suck labels from rig pressure and write them beside each capture.

    python scripts/derive_events.py --root new_dataset/20260718
    python scripts/derive_events.py --root new_dataset/20260718 --dry-run

Writes ``<stem>_events.csv`` plus ``<stem>_events_detector.json`` (the pinned
parameters) for every capture found. ``--dry-run`` reports what it would detect
without writing, which is the useful mode when tuning the band.

Labels are suck only — the pump rig has no swallow mechanism.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milkeaze.data.pressure_events import DetectorConfig, detect_suck_events, write_events
from milkeaze.data.rig_session import discover_stems, load_pressure, open_capture
from milkeaze.utils.logging import get_logger

log = get_logger("derive_events")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="directory of dual-board captures")
    ap.add_argument("--stem", default=None, help="process a single capture stem")
    ap.add_argument("--band-cpm", type=float, nargs=2, default=(18.0, 90.0),
                    metavar=("LO", "HI"), help="plausible cycle-rate band (default 18 90)")
    ap.add_argument("--convention", choices=("peak", "onset"), default="peak",
                    help="what t_ms marks (default: peak suction)")
    ap.add_argument("--prominence-fraction", type=float, default=0.35,
                    help="keep cycles at least this deep vs the median cycle")
    ap.add_argument("--dry-run", action="store_true", help="report without writing files")
    args = ap.parse_args()

    root = Path(args.root)
    stems = [args.stem] if args.stem else discover_stems(root)
    if not stems:
        log.error("no captures found under %s", root)
        return 1

    config = DetectorConfig(
        band_cpm=(args.band_cpm[0], args.band_cpm[1]),
        convention=args.convention,
        prominence_fraction=args.prominence_fraction,
    )

    print(f"{'capture':<34} {'vac':>4} {'set':>5} {'events':>7} "
          f"{'cpm':>7} {'CV':>6} {'spec':>7}")
    failures = 0
    for stem in stems:
        try:
            capture = open_capture(root, stem)
            t_ms, psi = load_pressure(capture)
            result = detect_suck_events(t_ms, psi, config)
        except (FileNotFoundError, ValueError) as exc:
            log.error("%s: %s", stem, exc)
            failures += 1
            continue

        set_rate = capture.cycle_rate_cpm
        print(f"{stem:<34} {str(capture.vacuum_level):>4} "
              f"{'' if set_rate is None else f'{set_rate:.0f}':>5} "
              f"{result.n_events:>7} {result.cycle_rate_cpm:>7.1f} "
              f"{result.cycle_rate_cv:>6.2f} {result.spectral_rate_cpm:>7.1f}")

        if not args.dry_run:
            write_events(result, root, filename=f"{stem}_events.csv")

    if args.dry_run:
        print("\n(dry run: nothing written)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
