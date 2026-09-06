"""How much of a capture's identity is written into its features.

Steve's Jul 30 point was that vacuum level and fill state move together across the
Jul 18 sweep, so cross-level comparisons are confounded. That is a statement about the
data, and it can be measured rather than assumed: if a trivial classifier can tell which
capture a window came from, then any model trained across those captures can reach the
same shortcut, and a good validation score may only prove it recognised the session.

The probe is deliberately weak — a nearest-centroid classifier on standardised features,
cross-validated at the *window* level. A weak learner succeeding is the strong result:
it means the separation is coarse and sitting in the marginal feature distributions,
exactly where a network will find it first.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConfoundProbe:
    """Cross-validated separability of a grouping variable from features alone."""

    accuracy: float
    chance: float          # majority-class rate, the floor
    n_samples: int
    n_groups: int
    per_group_recall: np.ndarray

    @property
    def skill(self) -> float:
        """Accuracy rescaled so chance is 0 and perfect separation is 1."""
        denom = 1.0 - self.chance
        return (self.accuracy - self.chance) / denom if denom > 0 else float("nan")

    def summary(self) -> str:
        return (f"{self.n_groups} groups, {self.n_samples} windows: "
                f"accuracy {self.accuracy:.3f} vs chance {self.chance:.3f} "
                f"(skill {self.skill:.3f})")


def _stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Fold assignment that keeps each group's share roughly constant per fold."""
    rng = np.random.default_rng(seed)
    fold_of = np.empty(y.size, dtype=np.int64)
    for group in np.unique(y):
        idx = np.flatnonzero(y == group)
        rng.shuffle(idx)
        fold_of[idx] = np.arange(idx.size) % n_folds
    return [np.flatnonzero(fold_of == f) for f in range(n_folds)]


def nearest_centroid_cv(X, y, n_folds: int = 5, seed: int = 0) -> ConfoundProbe:
    """Cross-validated nearest-centroid accuracy for predicting group from features."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features), got shape {X.shape}")
    if X.shape[0] != y.size:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.size}")

    groups = np.unique(y)
    if groups.size < 2:
        raise ValueError("need at least 2 groups to probe separability")
    n_folds = int(min(n_folds, np.bincount(np.searchsorted(groups, y)).min()))
    if n_folds < 2:
        raise ValueError("at least one group has too few samples to cross-validate")

    predicted = np.empty(y.size, dtype=y.dtype)
    for test_idx in _stratified_folds(y, n_folds, seed):
        train_mask = np.ones(y.size, dtype=bool)
        train_mask[test_idx] = False

        train_X, train_y = X[train_mask], y[train_mask]
        mean = train_X.mean(axis=0)
        scale = train_X.std(axis=0)
        scale = np.where(scale <= 0, 1.0, scale)

        Z_train = (train_X - mean) / scale
        centroids = np.stack([Z_train[train_y == g].mean(axis=0) for g in groups])

        Z_test = (X[test_idx] - mean) / scale
        dist = ((Z_test[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        predicted[test_idx] = groups[np.argmin(dist, axis=1)]

    correct = predicted == y
    counts = np.array([(y == g).sum() for g in groups], dtype=np.float64)
    recall = np.array([correct[y == g].mean() for g in groups], dtype=np.float64)

    return ConfoundProbe(
        accuracy=float(correct.mean()),
        chance=float(counts.max() / counts.sum()),
        n_samples=int(y.size),
        n_groups=int(groups.size),
        per_group_recall=recall,
    )


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks, so Spearman handles the ties that quantised features produce."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size, dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def spearman(x, y) -> float:
    """Rank correlation, for monotone-but-not-linear feature responses."""
    a, b = _rank(np.asarray(x, dtype=np.float64).ravel()), _rank(np.asarray(y, dtype=np.float64).ravel())
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    return float((a @ b) / denom) if denom > 0 else 0.0


def condition_correlations(X, condition) -> np.ndarray:
    """Per-feature rank correlation with an ordinal condition (e.g. vacuum level)."""
    X = np.asarray(X, dtype=np.float64)
    condition = np.asarray(condition, dtype=np.float64).ravel()
    if X.shape[0] != condition.size:
        raise ValueError(f"X has {X.shape[0]} rows but condition has {condition.size}")
    return np.array([spearman(X[:, j], condition) for j in range(X.shape[1])])
