"""Inference: run the model over a session and aggregate to session-level outputs."""

from .session import SessionResult, aggregate_session
from .predict import predict_session

__all__ = ["SessionResult", "aggregate_session", "predict_session"]
