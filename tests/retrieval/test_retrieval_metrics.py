"""Retrieval metrics (plan Task 12, design D7): hand-written over one query's ranked review ids and
its graded pool pairs (unjudged = 0), checked against hand-worked rankings and scikit-learn."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from patchpulse.retrieval.metrics import mean_over_queries, mrr_at, ndcg_at, recall_at

# One query's judged pool: 1 and 2 are relevant (2), 3 partly (1), 4 and 5 not (0).
GRADES = {1: 2, 2: 2, 3: 1, 4: 0, 5: 0}


def test_metrics_match_hand_worked_rankings() -> None:
    ranked = [4, 1, 9, 3]  # 9 was never judged: it counts as 0

    # Recall@k: relevant (grade >= 1) in the top k over the 3 relevant judged.
    assert recall_at(ranked, GRADES, 2) == pytest.approx(1 / 3)
    assert recall_at(ranked, GRADES, 4) == pytest.approx(2 / 3)
    # MRR@k: the first relevant is at rank 2.
    assert mrr_at(ranked, GRADES, 10) == pytest.approx(0.5)
    assert mrr_at(ranked, GRADES, 1) == 0.0
    # NDCG@4: DCG = 0/log2(2) + 2/log2(3) + 0/log2(4) + 1/log2(5);
    # ideal = 2/log2(2) + 2/log2(3) + 1/log2(4) + 0.
    dcg = 2 / math.log2(3) + 1 / math.log2(5)
    ideal = 2 + 2 / math.log2(3) + 1 / 2
    assert ndcg_at(ranked, GRADES, 4) == pytest.approx(dcg / ideal)
    assert ndcg_at([1, 2, 3], GRADES, 3) == pytest.approx(1.0)
    assert ndcg_at([], GRADES, 10) == 0.0  # nothing found


def test_ndcg_matches_scikit_learn() -> None:
    rng = np.random.default_rng(7)
    for _ in range(50):
        judged = rng.integers(0, 3, size=30)
        grades = {review: int(grade) for review, grade in enumerate(judged)}
        if not any(grades.values()):
            continue
        ranked = [int(r) for r in rng.permutation(30)[:20]]
        # scikit-learn scores every judged review; ours ranks the top 20, the rest below them.
        scores = np.full(30, -1.0)
        scores[ranked] = np.arange(20, 0, -1)
        for k in (10, 20):
            expected = ndcg_score([judged], [scores], k=k)
            assert ndcg_at(ranked, grades, k) == pytest.approx(expected)


def test_queries_without_relevant_reviews_are_counted_not_averaged() -> None:
    nothing = {7: 0, 8: 0}
    assert recall_at([7, 8], nothing, 10) is None
    assert ndcg_at([7, 8], nothing, 10) is None
    assert mrr_at([7, 8], nothing, 10) is None

    mean, counted, left_out = mean_over_queries([0.5, None, 1.0, None])
    assert mean == pytest.approx(0.75)
    assert (counted, left_out) == (2, 2)
    with pytest.raises(ValueError, match="k must be"):
        recall_at([1], GRADES, 0)
