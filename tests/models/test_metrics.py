"""Hand-written evaluation metrics (spec §0 rule 6): known values, then library cross-checks."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats
from sklearn.metrics import brier_score_loss, cohen_kappa_score, f1_score

from patchpulse.models.metrics import (
    bootstrap_ci,
    brier_score,
    cohen_kappa,
    expected_calibration_error,
    macro_f1,
    micro_f1,
    paired_bootstrap_ci,
    prf,
)


def kappa_table() -> tuple[list[bool], list[bool]]:
    """Cohen's textbook 2x2 table over 50 items: both yes 20, A yes / B no 5, A no / B yes 10,
    both no 15. Agreement 0.7, chance 0.5, so kappa = 0.4."""
    a = [True] * 20 + [True] * 5 + [False] * 10 + [False] * 15
    b = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
    return a, b


def random_labels(seed: int, n: int = 300, k: int = 10) -> tuple[NDArray[np.bool_], ...]:
    rng = np.random.default_rng(seed)
    truth = rng.random((n, k)) < 0.2
    noisy = truth ^ (rng.random((n, k)) < 0.15)
    return truth, noisy


def test_kappa_matches_a_hand_computed_table() -> None:
    a, b = kappa_table()

    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_kappa_is_none_when_both_annotators_are_constant() -> None:
    assert cohen_kappa([False] * 10, [False] * 10) is None


def test_kappa_matches_scikit_learn() -> None:
    truth, noisy = random_labels(1)

    for aspect in range(truth.shape[1]):
        ours = cohen_kappa(truth[:, aspect], noisy[:, aspect])
        assert ours == pytest.approx(cohen_kappa_score(truth[:, aspect], noisy[:, aspect]))


def test_f1_with_no_positives_is_zero_and_flagged() -> None:
    truth = np.array([[True, False], [False, False]])
    predicted = np.array([[True, False], [False, False]])

    result = prf(truth, predicted)

    assert result.f1[0] == pytest.approx(1.0)
    assert result.f1[1] == 0.0
    assert list(result.empty) == [False, True]
    assert list(result.support) == [1, 0]


def test_f1_matches_scikit_learn() -> None:
    truth, noisy = random_labels(2)

    assert macro_f1(truth, noisy) == pytest.approx(
        f1_score(truth, noisy, average="macro", zero_division=0)
    )
    assert micro_f1(truth, noisy) == pytest.approx(
        f1_score(truth, noisy, average="micro", zero_division=0)
    )
    per_aspect = f1_score(truth, noisy, average=None, zero_division=0)
    assert prf(truth, noisy).f1 == pytest.approx(per_aspect)


def test_ece_matches_a_hand_computed_example() -> None:
    # Two bins: [0, 0.5) holds 0.1 and 0.4 (mean 0.25, no positives); [0.5, 1] holds 0.6 and 0.9
    # (mean 0.75, all positive). Each bin is off by 0.25 and holds half the items.
    probs = np.array([0.1, 0.4, 0.6, 0.9])
    truth = np.array([False, False, True, True])

    assert expected_calibration_error(truth, probs, bins=2) == pytest.approx(0.25)


def test_ece_of_a_perfectly_calibrated_bin_is_zero() -> None:
    probs = np.full(10, 0.3)
    truth = np.array([True] * 3 + [False] * 7)

    assert expected_calibration_error(truth, probs) == pytest.approx(0.0)


def test_brier_matches_scikit_learn() -> None:
    rng = np.random.default_rng(3)
    probs = rng.random(200)
    truth = rng.random(200) < probs

    assert brier_score(truth, probs) == pytest.approx(brier_score_loss(truth, probs))


def test_bootstrap_is_reproducible_with_a_seed() -> None:
    values = np.random.default_rng(4).random(100)

    def mean(indices: NDArray[np.intp]) -> float:
        return float(values[indices].mean())

    assert bootstrap_ci(mean, 100, seed=7) == bootstrap_ci(mean, 100, seed=7)


def test_bootstrap_matches_scipy_within_tolerance() -> None:
    values = np.random.default_rng(5).random(400)

    def mean(indices: NDArray[np.intp]) -> float:
        return float(values[indices].mean())

    low, high = bootstrap_ci(mean, len(values), samples=5000, seed=11)
    reference = stats.bootstrap(
        (values,), np.mean, n_resamples=5000, method="percentile", random_state=11
    ).confidence_interval

    assert low == pytest.approx(reference.low, abs=0.005)
    assert high == pytest.approx(reference.high, abs=0.005)


def test_paired_bootstrap_of_identical_models_contains_zero() -> None:
    truth, noisy = random_labels(6)

    def score(indices: NDArray[np.intp]) -> float:
        return macro_f1(truth[indices], noisy[indices])

    low, high = paired_bootstrap_ci(score, score, truth.shape[0], seed=3)

    assert low <= 0.0 <= high


def test_paired_bootstrap_separates_a_clearly_better_model() -> None:
    truth, noisy = random_labels(8)

    def perfect(indices: NDArray[np.intp]) -> float:
        return macro_f1(truth[indices], truth[indices])

    def worse(indices: NDArray[np.intp]) -> float:
        return macro_f1(truth[indices], noisy[indices])

    low, _ = paired_bootstrap_ci(perfect, worse, truth.shape[0], seed=3)

    assert low > 0.0
