import numpy as np
import pytest

from milkeaze.eval.confound import condition_correlations, nearest_centroid_cv, spearman


def test_separable_groups_are_recovered_far_above_chance():
    """This is the shape of the real result: captures are trivially distinguishable."""
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(offset, 0.3, size=(80, 4))
                        for offset in (0.0, 3.0, 6.0)])
    y = np.repeat(["a", "b", "c"], 80)

    probe = nearest_centroid_cv(X, y)
    assert probe.accuracy > 0.95
    assert probe.chance == pytest.approx(1 / 3)
    assert probe.skill > 0.9
    assert probe.per_group_recall.min() > 0.9


def test_inseparable_groups_score_near_chance():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4))
    y = np.repeat([0, 1, 2], 100)

    probe = nearest_centroid_cv(X, y)
    assert probe.accuracy < 0.55
    assert probe.skill < 0.3


def test_scale_free_because_features_are_standardised_per_fold():
    rng = np.random.default_rng(2)
    X = np.concatenate([rng.normal(offset, 0.3, size=(60, 3)) for offset in (0.0, 3.0)])
    y = np.repeat([0, 1], 60)

    inflated = X * np.array([1.0, 1e6, 1e-6])
    assert nearest_centroid_cv(inflated, y).accuracy == pytest.approx(
        nearest_centroid_cv(X, y).accuracy, abs=0.05
    )


def test_a_single_group_cannot_be_probed():
    with pytest.raises(ValueError, match="at least 2 groups"):
        nearest_centroid_cv(np.zeros((10, 2)), np.zeros(10))


def test_too_few_samples_per_group_raises():
    with pytest.raises(ValueError, match="cross-validate"):
        nearest_centroid_cv(np.zeros((3, 2)), np.array([0, 1, 1]))


def test_spearman_is_monotone_not_linear():
    x = np.arange(20, dtype=float)
    assert spearman(x, x ** 3) == pytest.approx(1.0)
    assert spearman(x, -np.exp(x / 5)) == pytest.approx(-1.0)
    assert spearman(x, np.zeros(20)) == 0.0


def test_spearman_handles_the_ties_quantised_features_produce():
    x = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    assert spearman(x, x) == pytest.approx(1.0)


def test_condition_correlations_are_per_feature():
    rng = np.random.default_rng(3)
    condition = np.repeat([1, 2, 3, 4, 5], 20)
    tracking = condition + rng.normal(0, 0.1, condition.size)
    noise = rng.normal(size=condition.size)

    rho = condition_correlations(np.column_stack([tracking, noise]), condition)
    assert rho.shape == (2,)
    assert rho[0] > 0.95
    assert abs(rho[1]) < 0.3
