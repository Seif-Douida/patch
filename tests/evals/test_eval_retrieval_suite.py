"""The retrieval suite (Phase 6 plan Task 9, design D11): the judged reviews as a small pool
collection, the queries' recorded vectors, and NDCG@10 and Recall@20 of hybrid search on it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from qdrant_client import QdrantClient
from tiny_encoder import write_tiny_embedder

from patchpulse.evals.retrieval_suite import (
    RecordedQueryEmbedder,
    export_pool,
    export_query_vectors,
    grades_of,
    load_pool,
    run_retrieval,
)
from patchpulse.models.snapshot import Snapshot
from patchpulse.retrieval.embed import EmbeddingCache, embed_index_set
from patchpulse.retrieval.embedders import CANDIDATES, OnnxEmbedder
from patchpulse.retrieval.queries import Query
from patchpulse.retrieval.rules import INDEX_COLUMNS

DIM = 32
SPEC = replace(CANDIDATES["e5-small"], name="tiny-pool", dims=DIM, max_tokens=32)
WORDS = ("good", "bad", "game", "fps", "drops", "crashes", "great", "story", "server", "down")
GAME, OTHER = 10, 20
UNIQUE = {7: "zebra", 8: "walrus"}


def text_of(rid: int) -> str:
    return UNIQUE.get(rid) or " ".join(WORDS[(rid * k) % len(WORDS)] for k in (1, 3, 7))


def corpus() -> Snapshot:
    ids = list(range(1, 21))
    index = pd.DataFrame(
        {
            "review_id": ids,
            "appid": [GAME if rid <= 15 else OTHER for rid in ids],
            "day": [20261001 + rid % 10 for rid in ids],
            "created_ts": [1_790_000_000 + rid for rid in ids],
            "language": ["english"] * 20,
            "voted_up": [rid % 2 == 0 for rid in ids],
            "playtime_bucket": ["2h_10h"] * 20,
            "first_seen_run_id": [1] * 20,
            "version": [pd.Timestamp("2026-10-01")] * 20,
            "aspects": ["performance"] * 20,
            "aspect_model_version": ["s0-1"] * 20,
            "fingerprint": [f"{rid:064x}" for rid in ids],
            "rules_version": ["r1"] * 20,
        },
        columns=list(INDEX_COLUMNS),
    )
    reviews = pd.DataFrame({"review_id": ids, "text": [f"[b]{text_of(rid)}[/b]" for rid in ids]})
    frame = pd.DataFrame({"appid": [GAME, OTHER]})
    return Snapshot(reviews=reviews, games=frame, patches=frame, index=index)


QUERIES = [
    Query("q1", GAME, "english", "aspect", "zebra"),
    Query("q2", GAME, "english", "aspect", "walrus"),
    Query("q3", GAME, "english", "aspect", "fps drops"),
]
# Every review judged; q3's only relevant review belongs to another game, so the game filter
# keeps it out of q3's ranking.
JUDGMENTS = pd.DataFrame(
    [("q1", rid, 2 if rid == 7 else 0) for rid in range(1, 21)]
    + [("q2", rid, 2 if rid == 8 else 0) for rid in range(1, 21)]
    + [("q3", rid, 1 if rid == 18 else 0) for rid in range(1, 21)]
    + [("q3", 99, 2)],  # a judged review the snapshot no longer has
    columns=["query_id", "review_id", "grade"],
)


@pytest.fixture(scope="module")
def embedded(tmp_path_factory: pytest.TempPathFactory) -> tuple[OnnxEmbedder, EmbeddingCache]:
    root = tmp_path_factory.mktemp("pool")
    write_tiny_embedder(root / "tiny", SPEC, dim=DIM)
    embedder = OnnxEmbedder(root / "tiny")
    cache = EmbeddingCache.for_embedder(root, embedder)
    embed_index_set(corpus(), embedder, cache)
    return embedder, cache


def test_an_unknown_query_text_is_refused_by_the_recorded_embedder(
    embedded: tuple[OnnxEmbedder, EmbeddingCache],
) -> None:
    embedder, _ = embedded
    vectors = export_query_vectors(QUERIES, embedder)
    recorded = RecordedQueryEmbedder(vectors)

    found = recorded.embed_queries(["zebra"])

    assert set(vectors) == {"zebra", "walrus", "fps drops"}
    assert found.shape == (1, DIM)
    assert found.dtype == np.float32
    np.testing.assert_allclose(found[0], embedder.embed_queries(["zebra"])[0], rtol=1e-6)
    with pytest.raises(ValueError, match="no recorded vector"):
        recorded.embed_queries(["unicorn"])


def test_the_pool_is_the_judged_reviews_with_a_vector(
    embedded: tuple[OnnxEmbedder, EmbeddingCache],
) -> None:
    _, cache = embedded

    pool = export_pool(corpus(), JUDGMENTS, cache)

    assert sorted(pool["review_id"]) == list(range(1, 21))
    assert pool.attrs["skipped"] == 1
    assert pool.loc[pool["review_id"] == 7, "text"].item() == "zebra"  # cleaned
    assert len(pool["vector"].iloc[0]) == DIM


@pytest.mark.qdrant
def test_pool_search_scores_a_tiny_pool(
    qdrant: QdrantClient,
    new_collection: Callable[[], str],
    embedded: tuple[OnnxEmbedder, EmbeddingCache],
) -> None:
    embedder, cache = embedded
    name = new_collection()
    pool = export_pool(corpus(), JUDGMENTS, cache)
    recorded = RecordedQueryEmbedder(export_query_vectors(QUERIES, embedder))

    load_pool(qdrant, pool, collection=name)
    scores = run_retrieval(qdrant, QUERIES, grades_of(JUDGMENTS), recorded, collection=name)

    assert scores.per_query["q1"] == (1.0, 1.0)
    assert scores.per_query["q2"] == (1.0, 1.0)
    assert scores.per_query["q3"] == (0.0, 0.0)
    assert scores.ndcg_at_10 == pytest.approx(2 / 3)
    assert scores.recall_at_20 == pytest.approx(2 / 3)


def test_grades_come_from_the_judgments() -> None:
    """Grades by query and review, from the judgments frame the bundle carries."""
    grades = grades_of(JUDGMENTS)

    assert grades["q1"][7] == 2
    assert grades["q3"] == {**dict.fromkeys(range(1, 21), 0), 18: 1, 99: 2}
