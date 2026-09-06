"""Grade the sidecars of every capture in a dataset directory.

    python scripts/validate_dataset.py --root new_dataset/20260718
    python scripts/validate_dataset.py --root new_dataset/20260718 --strict

Errors mean the capture cannot be used as-is; warnings are known, agreed gaps that
should not silently persist into the large collection. ``--strict`` exits non-zero on any
error. ``--frozen`` additionally promotes the promised-but-missing sidecar fields to
errors, which is how this runs as a collection gate once the schema is frozen — run the
two together to see what the freeze will reject.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milkeaze.data.schema import PENDING_FIELDS, validate_dataset
from milkeaze.utils.logging import get_logger

log = get_logger("validate_dataset")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="directory of dual-board captures")
    ap.add_argument("--strict", action="store_true", help="exit non-zero on any error")
    ap.add_argument("--frozen", action="store_true",
                    help=f"treat the promised fields as required: {', '.join(PENDING_FIELDS)}")
    ap.add_argument("--quiet", action="store_true", help="summary lines only")
    args = ap.parse_args()

    try:
        reports = validate_dataset(Path(args.root), frozen=args.frozen)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1

    n_errors = 0
    for report in reports:
        print(f"\n{report.summary()}")
        n_errors += len(report.errors)
        if args.quiet:
            continue
        for issue in report.issues:
            print(f"    {issue}")

    # the same warning on every capture is a dataset-wide gap, not five separate ones
    recurring: dict[str, int] = {}
    for report in reports:
        for issue in report.warnings:
            recurring[issue.field] = recurring.get(issue.field, 0) + 1
    dataset_wide = sorted(f for f, n in recurring.items() if n == len(reports))
    if dataset_wide and not args.quiet:
        print(f"\nAffecting all {len(reports)} captures: {', '.join(dataset_wide)}")

    mode = " (frozen schema)" if args.frozen else ""
    print(f"\n{len(reports)} capture(s), {n_errors} error(s){mode}")
    return 1 if (args.strict and n_errors) else 0


if __name__ == "__main__":
    raise SystemExit(main())
