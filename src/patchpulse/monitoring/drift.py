"""The weekly drift statistics (Phase 7 design D2): the week's reviews against the champion's
training data (`models/reference.json`).

- Language mix and text length: population stability index (PSI) over fixed categories.
- Embeddings: a domain classifier's ROC AUC. Logistic regression, out of fold, tells the
  reference's e5 vectors from the week's; 0.5 means it can't tell them apart.
- Predictions: each aspect's predicted share with a bootstrap interval, and the largest move.
- Confidence: the share of review-aspect probabilities within 0.1 of their threshold.

These are Evidently's default methods (PSI; the domain classifier for embeddings), written on scipy
and scikit-learn so no review text is ever rendered (design D2). Numbers only.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.metrics import bootstrap_ci
from patchpulse.monitoring.alerts import AlertRule, Outcome
from patchpulse.monitoring.reference import Reference
from patchpulse.monitoring.store import MetricRow

LANGUAGE_TOP = 8  # reference.LANGUAGE_TOP is the same
LENGTH_EDGES = (1.0, 1.5, 2.0, 2.5, 3.0, 3.3)  # log10 of characters
PSI_FLOOR = 1e-4
UNCERTAIN_BAND = 0.1
DOMAIN_SAMPLE = 1000
DOMAIN_FOLDS = 5
DOMAIN_MIN = 50  # fewer on either side gives no AUC
DRIFT_SEED = 20261117
MIN_WEEK_REVIEWS = 200  # a smaller week isn't measured (calibration and the nightly alike)
OTHER = "other"

# Each drift rule of config/alerts.yaml and the statistic it watches.
DRIFT_RULES = {
    "drift_language": "language_psi",
    "drift_length": "length_psi",
    "drift_embedding": "embedding_auc",
    "drift_aspects": "aspect_max_change",
    "drift_confidence": "uncertain_change",
}


@dataclass(frozen=True)
class WeekSample:
    review_ids: list[int]
    languages: list[str]
    lengths: list[int]  # characters
    labels: NDArray[np.bool_]  # (reviews, aspects), the champion's
    probs: NDArray[np.float64]  # (reviews, aspects)


class VectorSource(Protocol):
    def vectors(self, ids: Sequence[int]) -> NDArray[np.float32]:
        """The e5 vectors of the ids found, one row each."""
        ...


def psi(reference: Mapping[str, float], current: Mapping[str, float]) -> float:
    total = 0.0
    for key in set(reference) | set(current):
        ref = max(reference.get(key, 0.0), PSI_FLOOR)
        cur = max(current.get(key, 0.0), PSI_FLOOR)
        total += (cur - ref) * math.log(cur / ref)
    return total


def _shares(counts: Counter[str], total: int) -> dict[str, float]:
    return {key: count / total for key, count in sorted(counts.items())} if total else {}


def language_shares(languages: Sequence[str], top: Sequence[str]) -> dict[str, float]:
    keep = set(top)
    counts = Counter(lang if lang in keep else OTHER for lang in languages)
    return _shares(counts, len(languages))


def _length_bin(length: int) -> str:
    position = int(np.searchsorted(LENGTH_EDGES, math.log10(max(length, 1)), side="right"))
    return f"bin{position}"


def length_shares(lengths: Sequence[int]) -> dict[str, float]:
    return _shares(Counter(_length_bin(n) for n in lengths), len(lengths))


def aspect_shares(labels: NDArray[np.bool_]) -> dict[str, float]:
    means = labels.mean(axis=0) if len(labels) else np.zeros(len(ASPECTS))
    return {code: float(share) for code, share in zip(ASPECTS, means, strict=True)}


def uncertain_share(probs: NDArray[np.float64], thresholds: NDArray[np.float64]) -> float:
    if not probs.size:
        return float("nan")
    return float(np.mean(np.abs(probs - thresholds[np.newaxis, :]) <= UNCERTAIN_BAND))


def domain_auc(
    reference: NDArray[np.float32], current: NDArray[np.float32], *, seed: int = DRIFT_SEED
) -> float:
    if len(reference) < DOMAIN_MIN or len(current) < DOMAIN_MIN:
        return float("nan")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    features = np.vstack([reference, current])
    target = np.r_[np.zeros(len(reference)), np.ones(len(current))]
    folds = StratifiedKFold(n_splits=DOMAIN_FOLDS, shuffle=True, random_state=seed)
    probs = cross_val_predict(
        LogisticRegression(max_iter=1000), features, target, cv=folds, method="predict_proba"
    )[:, 1]
    return float(roc_auc_score(target, probs))


def _sample(ids: Sequence[int], k: int, seed: int) -> list[int]:
    if len(ids) <= k:
        return list(ids)
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(np.asarray(ids), size=k, replace=False))


def drift_metrics(
    reference: Reference,
    week: WeekSample,
    vectors: VectorSource,
    *,
    window: tuple[date, date],
    scope: str = "week",
) -> list[MetricRow]:
    n = len(week.review_ids)

    def row(
        metric: str, value: float, lo: float | None = None, hi: float | None = None, m: int = n
    ) -> MetricRow:
        clean = None if math.isnan(value) else value
        return MetricRow("drift", metric, scope, window[0], window[1], clean, lo, hi, m)

    rows = [
        row(
            "language_psi",
            psi(
                reference.language_shares, language_shares(week.languages, reference.languages_top)
            ),
        ),
        row("length_psi", psi(reference.length_shares, length_shares(week.lengths))),
    ]
    reference_vectors = vectors.vectors(reference.sample_ids)
    week_vectors = vectors.vectors(_sample(week.review_ids, DOMAIN_SAMPLE, DRIFT_SEED))
    auc = domain_auc(reference_vectors, week_vectors)
    rows.append(row("embedding_auc", auc, m=min(len(reference_vectors), len(week_vectors))))

    shares = aspect_shares(week.labels)
    moves = []
    for k, code in enumerate(ASPECTS):
        column = week.labels[:, k]

        def share(ix: NDArray[np.intp], c: NDArray[np.bool_] = column) -> float:
            return float(c[ix].mean())

        lo, hi = bootstrap_ci(share, n, seed=DRIFT_SEED) if n else (None, None)
        rows.append(row(f"aspect_share.{code}", shares[code], lo, hi))
        moves.append(abs(shares[code] - reference.aspect_shares[code]) * 100)
    rows.append(row("aspect_max_change", max(moves) if n else float("nan")))
    thresholds = np.array([reference.thresholds[code] for code in ASPECTS])
    change = (uncertain_share(week.probs, thresholds) - reference.uncertain_share) * 100
    rows.append(row("uncertain_change", change))
    return rows


def drift_outcomes(rows: Sequence[MetricRow], rules: Mapping[str, AlertRule]) -> list[Outcome]:
    """One outcome per drift rule: breached above its threshold; None while the threshold or the
    week's value is missing (design D2: thresholds come from the calibration)."""
    by_metric = {row.metric: row for row in rows}
    outcomes = []
    for rule_id, metric in DRIFT_RULES.items():
        threshold = rules[rule_id].threshold
        found = by_metric.get(metric)
        value = None if found is None else found.value
        if threshold is None or value is None or found is None:
            outcomes.append(Outcome(rule_id, None, f"{metric}: not measured"))
            continue
        body = (
            f"{metric} {value:.3f} (threshold {threshold:.3f}), "
            f"week {found.window_start} to {found.window_end}, n {found.n}"
        )
        outcomes.append(Outcome(rule_id, value > threshold, body))
    return outcomes
