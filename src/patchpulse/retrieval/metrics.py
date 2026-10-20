"""Retrieval metrics for Phase 3's eval (design D7, plan Task 12), hand-written (spec §0 rule 6).

Each works on one query: its ranked review ids (a system's run) and the grades of its judged pool
pairs, 0 to 2. A review the judges never saw counts as 0. Relevant means grade 1 or more. Recall is
pooled: it counts the relevant reviews any compared system found, not every relevant review in the
index.

A query with no relevant review in its pool has no recall, NDCG or MRR (None): it's left out of the
means and counted, never scored 0 or 1.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

Grades = Mapping[int, int]  # review id -> grade, for one query's judged pool pairs


def _check(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be at least 1, not {k}")


def recall_at(ranked: Sequence[int], grades: Grades, k: int) -> float | None:
    """Relevant reviews in the top k over the query's relevant judged reviews."""
    _check(k)
    relevant = {review for review, grade in grades.items() if grade >= 1}
    if not relevant:
        return None
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def _dcg(gains: Sequence[int]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at(ranked: Sequence[int], grades: Grades, k: int) -> float | None:
    """Linear gain (the grade) with a log2 discount, over the ideal ranking of every judged pair."""
    _check(k)
    ideal = _dcg(sorted(grades.values(), reverse=True)[:k])
    if ideal == 0.0:
        return None
    return _dcg([grades.get(review, 0) for review in ranked[:k]]) / ideal


def mrr_at(ranked: Sequence[int], grades: Grades, k: int) -> float | None:
    """1 over the rank of the first relevant review in the top k; 0 if none is there."""
    _check(k)
    if not any(grade >= 1 for grade in grades.values()):
        return None
    for rank, review in enumerate(ranked[:k], start=1):
        if grades.get(review, 0) >= 1:
            return 1.0 / rank
    return 0.0


def mean_over_queries(values: Sequence[float | None]) -> tuple[float | None, int, int]:
    """(mean, queries counted, queries left out for having no relevant review)."""
    scored = [value for value in values if value is not None]
    mean = sum(scored) / len(scored) if scored else None
    return mean, len(scored), len(values) - len(scored)
