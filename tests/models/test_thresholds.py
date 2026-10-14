"""Per-aspect decision thresholds (plan Task 11): tuned on gold dev, compared out of fold."""

from __future__ import annotations

import numpy as np
import pytest

from patchpulse.models.metrics import macro_f1, prf
from patchpulse.models.thresholds import (
    DEFAULT_CUT,
    cv_macro_f1,
    cv_predictions,
    tune_thresholds,
)


def best_f1_by_brute_force(truth: np.ndarray, prob: np.ndarray) -> float:
    """Every cut that changes a decision: each distinct probability, and above them all."""
    cuts = [*np.unique(prob), np.inf]
    return max(float(prf(truth[:, None], (prob >= cut)[:, None]).f1[0]) for cut in cuts)


def test_thresholds_maximise_f1_on_a_toy() -> None:
    rng = np.random.default_rng(11)
    truth = rng.random((80, 3)) < [0.5, 0.2, 0.05]
    prob = np.clip(truth * 0.3 + rng.random((80, 3)) * 0.7, 0, 1)

    cuts = tune_thresholds(truth, prob)

    assert cuts.shape == (3,)
    for aspect in range(3):
        tuned = prf(truth[:, [aspect]], (prob[:, aspect] >= cuts[aspect])[:, None]).f1[0]
        assert tuned == pytest.approx(best_f1_by_brute_force(truth[:, aspect], prob[:, aspect]))


def test_the_cut_sits_halfway_to_the_next_lower_score() -> None:
    truth = np.array([[False], [False], [True], [True]])
    prob = np.array([[0.1], [0.4], [0.35], [0.8]])  # best: positive from 0.35 up (F1 0.8)

    assert tune_thresholds(truth, prob)[0] == pytest.approx(0.225)


def test_an_aspect_without_positives_keeps_the_default_cut() -> None:
    truth = np.zeros((5, 2), dtype=bool)
    truth[0, 0] = True
    prob = np.linspace(0.1, 0.9, 10).reshape(5, 2)

    assert tune_thresholds(truth, prob)[1] == DEFAULT_CUT


def test_a_review_is_judged_by_cuts_tuned_without_it() -> None:
    rng = np.random.default_rng(3)
    truth = rng.random((60, 1)) < 0.4
    prob = rng.random((60, 1))  # noise: tuning in-sample would chase every label
    base = cv_predictions(truth, prob, folds=5, seed=1)

    for row in range(60):
        flipped = truth.copy()
        flipped[row, 0] = not flipped[row, 0]
        assert cv_predictions(flipped, prob, folds=5, seed=1)[row, 0] == base[row, 0]


def test_cv_macro_f1_is_not_flattered_by_its_own_tuning() -> None:
    rng = np.random.default_rng(5)
    truth = rng.random((150, 10)) < 0.2
    prob = rng.random((150, 10))
    in_sample = macro_f1(truth, prob >= tune_thresholds(truth, prob))

    assert cv_macro_f1(truth, prob, seed=1) < in_sample
    assert cv_macro_f1(truth, prob, seed=1) == cv_macro_f1(truth, prob, seed=1)


def test_cv_macro_f1_is_perfect_when_scores_separate_the_labels() -> None:
    rng = np.random.default_rng(8)
    truth = rng.random((100, 4)) < 0.3
    prob = np.where(truth, 0.6 + 0.4 * rng.random((100, 4)), 0.4 * rng.random((100, 4)))

    assert cv_macro_f1(truth, prob, seed=2) == 1.0


def test_mismatched_shapes_are_refused() -> None:
    with pytest.raises(ValueError, match="shape"):
        tune_thresholds(np.zeros((3, 2), dtype=bool), np.zeros((3, 3)))
