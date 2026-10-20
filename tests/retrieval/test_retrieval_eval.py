"""The retrieval eval (plan Task 12, design D7): every system's metrics per query, intervals over
queries, the embedding model's choice by rules fixed before the results, ANN recall, and a report
that holds numbers only."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from patchpulse.retrieval import evaluate
from patchpulse.retrieval.agreement import relevance_agreement
from patchpulse.retrieval.benchmark import EmbedderSpeed
from patchpulse.retrieval.evaluate import (
    Report,
    ann_recall,
    choose_embedder,
    evaluate_runs,
    paired,
    render_report,
    score_run,
    summarize,
)
from patchpulse.retrieval.queries import Query
from patchpulse.retrieval.search import Hit

GRADES = {1: 2, 2: 2, 3: 1, 4: 0, 5: 0}


def speed(name: str, passages_per_s: float) -> EmbedderSpeed:
    return EmbedderSpeed(name, passages_per_s, 5.0, 50.0, 100.0)


SPEEDS = {
    "slow-best": speed("slow-best", 1.0),  # 300 passages in the 5-minute budget: under the floor
    "a": speed("a", 5.0),
    "b": speed("b", 15.0),  # three times cheaper than a
}


def test_choice_rule_floor_then_ndcg_then_tie_to_cheaper() -> None:
    rng = np.random.default_rng(1)
    base = rng.uniform(0.3, 0.7, size=100)

    # The floor comes first: the best NDCG under the floor is out. Then the best NDCG wins.
    clear = {"slow-best": list(base + 0.2), "a": list(base + 0.1), "b": list(base)}
    choice = choose_embedder(clear, SPEEDS, seed=1)
    assert choice.name == "a"
    assert "best hybrid NDCG@10" in choice.reason
    rows = {row["name"]: row for row in choice.table}
    assert rows["slow-best"]["passes_floor"] is False
    assert rows["a"]["passes_floor"] is True

    # A tie (the paired interval contains 0) goes to the cheaper model, even if a is ahead.
    noise = rng.normal(0, 0.05, size=100)
    noise = noise - noise.mean() + 0.002
    tied: dict[str, list[float | None]] = {
        "slow-best": list(base),
        "a": list(base + noise),
        "b": list(base),
    }
    tied["a"][0] = tied["b"][0] = None  # a query without relevant reviews is skipped by both
    choice = choose_embedder(tied, SPEEDS, seed=1)
    assert choice.name == "b"
    assert "tie" in choice.reason

    with pytest.raises(ValueError, match="no candidate passes the speed floor"):
        choose_embedder(clear, {"slow-best": SPEEDS["slow-best"]}, seed=1)


def test_score_run_gives_each_metric_per_query() -> None:
    run = pd.DataFrame(
        [("q1", 1, 4), ("q1", 2, 1), ("q1", 3, 9), ("q1", 4, 3), ("q2", 1, 7)],
        columns=["query_id", "rank", "review_id"],
    )
    grades = {"q1": GRADES, "q2": {7: 0, 8: 0}}

    values = score_run(run, grades, ["q1", "q2", "q3"])

    assert list(values) == ["Recall@10", "Recall@20", "NDCG@10", "MRR@10"]
    assert values["Recall@10"][0] == pytest.approx(2 / 3)
    assert values["MRR@10"] == [0.5, None, None]  # q2 has no relevant review; q3 no judgments


def test_summaries_and_paired_intervals_skip_queries_without_relevant_reviews() -> None:
    summary = summarize([0.5, None, 1.0, 0.0], seed=3)
    assert summary.mean == pytest.approx(0.5)
    assert (summary.counted, summary.left_out) == (3, 1)
    assert summary.low is not None
    assert summary.high is not None
    assert summary.low <= 0.5 <= summary.high

    difference = paired([0.9, None, 0.8, 0.7], [0.5, None, 0.4, 0.3], seed=3)
    assert difference.mean == pytest.approx(0.4)
    assert difference.n == 3
    assert difference.low == pytest.approx(0.4)
    assert difference.high == pytest.approx(0.4)


def test_ann_recall_is_the_overlap_with_exact_search(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_search(*args: Any, mode: str, k: int, exact: bool, **kwargs: Any) -> list[Hit]:
        assert mode == "dense"
        assert k == 10
        ids = range(10) if exact else range(2, 12)  # 8 of the exact top 10 found
        return [Hit(review_id, 1.0, "") for review_id in ids]

    monkeypatch.setattr(evaluate, "search", fake_search)
    queries = [Query(f"q{n}", 10, "english", "aspect", f"query {n}") for n in range(3)]

    class NoEmbedder:  # the search is faked: nothing is embedded
        def embed_queries(self, texts: list[str], *, batch: int = 32) -> Any:
            raise AssertionError("not called")

    assert ann_recall(object(), "eval_x", NoEmbedder(), queries, k=10) == pytest.approx(0.8)


def tiny_eval() -> Report:
    queries = [
        Query("q1", 10, "english", "aspect", "secret query one"),
        Query("q2", 10, "french", "event", "secret query two"),
        Query("q3", 10, "english", "crosslingual", "secret query three"),
    ]
    grades = {"q1": {101: 2, 102: 1}, "q2": {201: 2, 202: 0}, "q3": {301: 1}}
    run = pd.DataFrame(
        [("q1", 1, 101), ("q1", 2, 102), ("q2", 1, 202), ("q2", 2, 201), ("q3", 1, 301)],
        columns=["query_id", "rank", "review_id"],
    )
    systems = ("bm25", "dense:a", "hybrid:a", "dense:b", "hybrid:b")
    runs = {(system, variant): run for system in systems for variant in ("base", "filtered")}
    evaluation = evaluate_runs(runs, grades, queries, seed=5)
    hybrid = {name: evaluation.per_query[(f"hybrid:{name}", "base")]["NDCG@10"] for name in "ab"}
    audit = [(f"q{i % 3 + 1}", 1000 + i) for i in range(30)]
    claude = {pair: i % 3 for i, pair in enumerate(audit)}
    seif = {pair: (i + 1) % 3 if i < 12 else i % 3 for i, pair in enumerate(audit)}
    agreement = relevance_agreement(claude, seif, audit, judges=dict.fromkeys(audit, "m"))
    return Report(
        evaluation=evaluation,
        choice=choose_embedder(hybrid, {"a": SPEEDS["a"], "b": SPEEDS["b"]}, seed=5),
        choice_without_events=choose_embedder(hybrid, {"a": SPEEDS["a"], "b": SPEEDS["b"]}, seed=5),
        ann_recall={"a": 0.97, "b": 0.99},
        agreement=agreement,
        agreement_by_kind={"aspect": (20, 0.75), "event": (10, 0.47)},
        judged_by={"claude-sonnet-5-5": 4, "claude-opus-5-5": 1},
        guidelines={"v2": 3, "v3": 2},
        accepted_failed_agreement=True,
    )


def test_report_holds_numbers_only_and_discloses_a_failed_agreement() -> None:
    text = render_report(tiny_eval())

    assert "Recall@10" in text
    assert "NDCG@10" in text
    assert "MRR@10" in text
    assert "hybrid:a" in text
    assert "bm25" in text
    assert "failed" in text
    assert "accepted" in text
    assert "0.47" in text  # the event queries' agreement
    assert "ANN recall@10" in text
    assert "0.97" in text
    assert "secret query" not in text  # no query text
    for review in ("101", "102", "201", "202", "301"):
        assert review not in text  # no review ids


def test_recall_has_a_ceiling_when_a_query_has_more_relevant_reviews_than_k() -> None:
    # 40 relevant reviews: ten results can hold at most 10 of them, so Recall@10 is at most 0.25.
    queries = [Query("q1", 10, "english", "aspect", "a"), Query("q2", 10, "english", "aspect", "b")]
    grades = {"q1": dict.fromkeys(range(40), 1), "q2": {1: 2, 2: 0}}
    run = pd.DataFrame([("q1", 1, 0), ("q2", 1, 1)], columns=["query_id", "rank", "review_id"])

    evaluation = evaluate_runs({("bm25", "base"): run}, grades, queries, seed=1)

    assert evaluation.recall_ceiling[10] == pytest.approx((10 / 40 + 1) / 2)
    assert evaluation.recall_ceiling[20] == pytest.approx((20 / 40 + 1) / 2)
    assert evaluation.relevant_per_query == pytest.approx(20.5)  # the median
    assert "best possible Recall@10" in render_report(replace(tiny_eval(), evaluation=evaluation))
