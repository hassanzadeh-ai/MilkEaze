"""Project logging. Fail-loud philosophy: warnings for recoverable data issues,
exceptions for contract violations (raised at the call site, not here)."""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "milkeaze") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        root = logging.getLogger("milkeaze")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        _CONFIGURED = True
    return logging.getLogger(name)
