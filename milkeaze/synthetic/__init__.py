"""Synthetic session generator for synthetic-first pretraining.

Real labeled data is scarce/expensive and per-event labels don't exist yet, so we
pretrain the backbone on physiologically-plausible synthetic sessions and later
fine-tune on real sessions. This is a coarse generator — good enough to pretrain
shapes/rhythm, not a physiological model of the breast.
"""

from .generator import SyntheticSession, generate_session, generate_dataset
from .write_session import write_session

__all__ = ["SyntheticSession", "generate_session", "generate_dataset", "write_session"]
