"""Per-aspect decision thresholds (plan Task 11, design D5-D6).

A student outputs a probability per aspect; an aspect is "on" when its probability reaches that
aspect's cut. Cuts are tuned on gold dev for F1. Comparing students by the F1 their cuts reach on
the same reviews they were tuned on would flatter them, so `cv_macro_f1` judges each dev review by
cuts tuned on the other folds only (champion rule 2).

NumPy only, so the jobs image can use it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from patchpulse.models.metrics import macro_f1

DEFAULT_CUT = 0.5  # for an aspect with no positives to tune on


def _matrices(y_true: ArrayLike, probs: ArrayLike) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    truth = np.asarray(y_true, dtype=bool)
    prob = np.asarray(probs, dtype=np.float64)
    if truth.shape != prob.shape or truth.ndim != 2:
        raise ValueError("labels and probabilities must have the same (reviews, aspects) shape")
    return truth, prob


def _best_cut(truth: NDArray[np.bool_], prob: NDArray[np.float64]) -> float:
    positives = int(truth.sum())
    if positives == 0:
        return DEFAULT_CUT
    order = np.argsort(-prob, kind="stable")
    ranked, hits = prob[order], truth[order]
    tp = np.cumsum(hits)
    fp = np.cumsum(~hits)
    f1 = 2 * tp / (tp + fp + positives)  # 2tp / (2tp + fp + fn), with fn = positives - tp
    # A cut can't separate equal scores: only the last of each run of ties is a real choice.
    last_of_run = np.append(ranked[1:] != ranked[:-1], True)
    best = int(np.argmax(np.where(last_of_run, f1, -1.0)))  # the highest cut among equal F1s
    lowest_positive = float(ranked[best])
    below = float(ranked[best + 1]) if best + 1 < ranked.size else 0.0
    cut = (lowest_positive + below) / 2  # halfway to the next score: the widest margin
    return cut if cut > below else lowest_positive


def tune_thresholds(y_true: ArrayLike, probs: ArrayLike) -> NDArray[np.float64]:
    """The F1-maximising cut for each aspect (predict "on" when probability >= cut)."""
    truth, prob = _matrices(y_true, probs)
    return np.array([_best_cut(truth[:, a], prob[:, a]) for a in range(truth.shape[1])])


def cv_predictions(
    y_true: ArrayLike, probs: ArrayLike, *, folds: int = 5, seed: int
) -> NDArray[np.bool_]:
    """Each review's labels from cuts tuned on the other folds: out-of-fold decisions."""
    truth, prob = _matrices(y_true, probs)
    if not 2 <= folds <= truth.shape[0]:
        raise ValueError(f"cross-validation needs 2 to {truth.shape[0]} folds, not {folds}")
    fold_of = np.random.default_rng(seed).permutation(truth.shape[0]) % folds
    predicted = np.zeros(truth.shape, dtype=bool)
    for fold in range(folds):
        held = fold_of == fold
        predicted[held] = prob[held] >= tune_thresholds(truth[~held], prob[~held])
    return predicted


def cv_macro_f1(y_true: ArrayLike, probs: ArrayLike, *, folds: int = 5, seed: int) -> float:
    """Macro-F1 of the out-of-fold decisions, pooled over all of dev."""
    return macro_f1(y_true, cv_predictions(y_true, probs, folds=folds, seed=seed))
