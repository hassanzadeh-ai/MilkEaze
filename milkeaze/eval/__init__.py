"""Evaluation layer: deciding whether a number is any good.

Kept separate from :mod:`milkeaze.training` on purpose. Nothing here imports torch, so
baselines, event matching and confound probes run against raw detector output and
processed sessions on any machine that can load the data — including ones where the
training stack cannot be installed.

Three questions, one module each:

:mod:`events`
    Do the predicted events line up with the reference ones? Counts and timing, matched
    one-to-one, because the product claim is a count.
:mod:`windows`
    Does a per-window score mean anything on this dataset? Usually not on rig data,
    where a constant "suck" answer already scores ~99%.
:mod:`baselines`
    Can something without learned parameters do the same job? Ridge for volume, a
    band-pass and a trough finder for rate.
:mod:`confound`
    Can a trivial classifier recover which capture a window came from — i.e. is a good
    validation score just session recognition?
:mod:`fill_response`
    How does strain gain move with reservoir fill, measured within a run where vacuum is
    fixed and the confound therefore does not apply?
"""
from __future__ import annotations

from .baselines import (
    STRAIN_POLARITY, RegressionScores, RidgeRegressor, fit_ridge, in_band_power_fraction,
    mean_predictor_scores, most_rhythmic_channel, regression_scores,
    strain_consensus_events, strain_consensus_signal, strain_event_baseline,
    strain_event_candidates, strain_rate_cpm,
)
from .confound import ConfoundProbe, condition_correlations, nearest_centroid_cv, spearman
from .events import EventMatch, match_events, rate_cpm, tolerance_from_period
from .fill_response import (
    FillResponse, fill_response, per_channel_fill_response, split_half_consistency,
    window_amplitudes,
)
from .windows import (
    LabelGranularity, class_balance, label_granularity, majority_class_rate, skill_score,
)

__all__ = [
    "STRAIN_POLARITY",
    "ConfoundProbe",
    "EventMatch",
    "FillResponse",
    "LabelGranularity",
    "RegressionScores",
    "RidgeRegressor",
    "class_balance",
    "condition_correlations",
    "fill_response",
    "fit_ridge",
    "in_band_power_fraction",
    "label_granularity",
    "majority_class_rate",
    "match_events",
    "mean_predictor_scores",
    "most_rhythmic_channel",
    "nearest_centroid_cv",
    "per_channel_fill_response",
    "rate_cpm",
    "regression_scores",
    "skill_score",
    "spearman",
    "split_half_consistency",
    "strain_consensus_events",
    "strain_consensus_signal",
    "strain_event_baseline",
    "strain_event_candidates",
    "strain_rate_cpm",
    "tolerance_from_period",
    "window_amplitudes",
]
