"""Evaluation metrics, written by hand (spec §0 rule 6) and cross-checked against scikit-learn and
SciPy in the tests.

Labels are boolean matrices: one row per review, one column per aspect. Probabilities have the same
shape. A metric that is undefined says so (`None`, or a flag) instead of returning a number that
looks meaningful.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

IndexMetric = Callable[[NDArray[np.intp]], float]


def _bool(values: ArrayLike) -> NDArray[np.bool_]:
    return np.asarray(values, dtype=bool)


def cohen_kappa(a: Sequence[bool] | ArrayLike, b: Sequence[bool] | ArrayLike) -> float | None:
    """Cohen's kappa for two annotators' yes/no labels on the same items.

    kappa = (p_o - p_e) / (1 - p_e): observed agreement corrected for the agreement two
    annotators with the same yes-rates would reach by chance. Undefined (None) when chance
    agreement is 1, i.e. both annotators gave the same single answer to everything.
    """
    x, y = _bool(a), _bool(b)
    if x.shape != y.shape or x.ndim != 1 or x.size == 0:
        raise ValueError("cohen_kappa needs two equal-length, non-empty 1-D label sequences")
    observed = float(np.mean(x == y))
    p_x, p_y = float(x.mean()), float(y.mean())
    expected = p_x * p_y + (1 - p_x) * (1 - p_y)
    if expected == 1.0:
        return None
    return (observed - expected) / (1 - expected)


@dataclass(frozen=True)
class PRF:
    """Per-aspect precision, recall and F1. `empty` flags aspects with no positives in the truth
    or in the predictions, whose F1 of 0 says nothing about the model."""

    precision: NDArray[np.float64]
    recall: NDArray[np.float64]
    f1: NDArray[np.float64]
    support: NDArray[np.int64]
    empty: NDArray[np.bool_]


def _counts(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[NDArray[np.int64], ...]:
    truth, predicted = _bool(y_true), _bool(y_pred)
    if truth.shape != predicted.shape or truth.ndim != 2:
        raise ValueError("labels must be two matrices of the same (reviews, aspects) shape")
    tp = (truth & predicted).sum(axis=0).astype(np.int64)
    fp = (~truth & predicted).sum(axis=0).astype(np.int64)
    fn = (truth & ~predicted).sum(axis=0).astype(np.int64)
    return tp, fp, fn


def _ratio(numerator: NDArray[np.int64], denominator: NDArray[np.int64]) -> NDArray[np.float64]:
    out = np.zeros(numerator.shape, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def prf(y_true: ArrayLike, y_pred: ArrayLike) -> PRF:
    tp, fp, fn = _counts(y_true, y_pred)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    return PRF(
        precision=precision,
        recall=recall,
        f1=f1,
        support=tp + fn,
        empty=(tp + fp == 0) | (tp + fn == 0),
    )


def macro_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """The unweighted mean of the per-aspect F1: every aspect counts the same, rare ones too."""
    return float(prf(y_true, y_pred).f1.mean())


def micro_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """F1 over every (review, aspect) decision pooled: dominated by the common aspects."""
    tp, fp, fn = (int(count.sum()) for count in _counts(y_true, y_pred))
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def expected_calibration_error(y_true: ArrayLike, prob: ArrayLike, *, bins: int = 10) -> float:
    """How far predicted probabilities sit from observed frequencies (0 is perfectly calibrated).

    Probabilities go into `bins` equal-width bins over [0, 1] (1.0 falls in the last). Each bin
    contributes |share of positives - mean probability|, weighted by its share of the items.
    Matrices are pooled over every (review, aspect) pair.
    """
    truth = _bool(y_true).ravel()
    p = np.asarray(prob, dtype=np.float64).ravel()
    if truth.shape != p.shape or p.size == 0:
        raise ValueError("labels and probabilities must have the same, non-zero size")
    which = np.minimum((p * bins).astype(np.intp), bins - 1)
    total = 0.0
    for b in range(bins):
        members = which == b
        count = int(members.sum())
        if count:
            total += count / p.size * abs(float(truth[members].mean()) - float(p[members].mean()))
    return total


def brier_score(y_true: ArrayLike, prob: ArrayLike) -> float:
    """Mean squared distance between the probability and the 0/1 outcome (lower is better)."""
    truth = _bool(y_true).ravel().astype(np.float64)
    p = np.asarray(prob, dtype=np.float64).ravel()
    if truth.shape != p.shape or p.size == 0:
        raise ValueError("labels and probabilities must have the same, non-zero size")
    return float(np.mean((p - truth) ** 2))


def _resamples(n_items: int, samples: int, seed: int) -> NDArray[np.intp]:
    if n_items <= 0 or samples <= 0:
        raise ValueError("bootstrap needs at least one item and one resample")
    return np.random.default_rng(seed).integers(0, n_items, size=(samples, n_items))


def _percentiles(values: NDArray[np.float64], alpha: float) -> tuple[float, float]:
    low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def bootstrap_ci(
    metric: IndexMetric, n_items: int, *, samples: int = 2000, seed: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap interval for `metric`, resampling items (reviews) with replacement.

    `metric` receives the resampled row indices, so it can score any aligned arrays.
    """
    draws = _resamples(n_items, samples, seed)
    return _percentiles(np.array([metric(indices) for indices in draws]), alpha)


def paired_bootstrap_ci(
    metric_a: IndexMetric,
    metric_b: IndexMetric,
    n_items: int,
    *,
    samples: int = 2000,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Interval for metric_a - metric_b, both scored on the same resample each time, so the
    comparison isn't drowned by which reviews happened to be drawn. Containing 0 means a tie."""
    draws = _resamples(n_items, samples, seed)
    return _percentiles(np.array([metric_a(ix) - metric_b(ix) for ix in draws]), alpha)
