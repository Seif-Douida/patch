"""Running every system over the frozen queries and pooling their top 20s (plan Task 9, design
D7): each system as `base` (the game only) and `filtered` (plus the query's own filters)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from qdrant_client import QdrantClient
from tiny_encoder import write_tiny_embedder

from patchpulse.models.snapshot import Snapshot
from patchpulse.retrieval.embed import EmbeddingCache, embed_index_set
from patchpulse.retrieval.embedders import CANDIDATES, OnnxEmbedder
from patchpulse.retrieval.load import load_collection
from patchpulse.retrieval.qdrant import ensure_collection
from patchpulse.retrieval.queries import COLUMNS, QueriesNotFrozenError, freeze_queries
from patchpulse.retrieval.rules import INDEX_COLUMNS
from patchpulse.retrieval.runs import (
    SYSTEMS,
    VARIANTS,
    pool,
    read_runs,
    run_systems,
    systems,
    write_runs,
)


def run(rows: list[tuple[str, int, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["query_id", "rank", "review_id", "score"])


def test_systems_are_bm25_then_dense_and_hybrid_per_candidate() -> None:
    assert systems(["a", "b"]) == ["bm25", "dense:a", "hybrid:a", "dense:b", "hybrid:b"]
    assert tuple(systems(list(CANDIDATES))) == SYSTEMS
    assert VARIANTS == ("base", "filtered")


def test_pool_is_the_deduplicated_union_of_top_20s() -> None:
    deep = [("q1", rank, 1000 + rank, 1.0 / rank) for rank in range(1, 26)]  # ranks past 20
    runs = {
        ("bm25", "base"): run([("q1", 1, 5, 9.0), ("q1", 2, 3, 8.0), ("q2", 1, 7, 1.0)]),
        ("dense:a", "base"): run([("q1", 1, 3, 0.9), ("q1", 2, 4, 0.8)]),  # 3 found twice
        ("dense:a", "filtered"): run(deep),
    }

    pairs = pool(runs, depth=20)

    assert list(pairs.columns) == ["query_id", "review_id"]
    q1 = pairs.loc[pairs["query_id"] == "q1", "review_id"].tolist()
    assert q1 == sorted({3, 4, 5} | {1000 + rank for rank in range(1, 21)})
    assert pairs.loc[pairs["query_id"] == "q2", "review_id"].tolist() == [7]
    assert pairs.equals(pairs.sort_values(["query_id", "review_id"]).reset_index(drop=True))
    assert not pairs.duplicated().any()


def test_runs_round_trip_through_csv_with_ids_and_scores_only(tmp_path: Path) -> None:
    runs = {
        ("bm25", "base"): run([("q1", 1, 5, 9.25), ("q1", 2, 3, 8.5)]),
        ("hybrid:e5-small", "filtered"): run([("q1", 1, 3, 0.5)]),
    }

    paths = write_runs(runs, tmp_path)

    # No colon in a file name (Windows); the columns are ids, ranks and scores, nothing else.
    assert sorted(p.name for p in paths) == ["bm25-base.csv", "hybrid_e5-small-filtered.csv"]
    assert (tmp_path / "bm25-base.csv").read_text(encoding="utf-8").splitlines()[0] == (
        "query_id,rank,review_id,score"
    )
    back = read_runs(tmp_path, ["bm25", "hybrid:e5-small"])
    assert set(back) == set(runs)
    for key, frame in runs.items():
        pd.testing.assert_frame_equal(back[key], frame, check_dtype=False)


# The toy corpus on Qdrant: 30 reviews of one game and 10 of another, in English.
DIM = 32
GAME, OTHER = 10, 20
WORDS = ("good", "bad", "game", "fps", "drops", "crashes", "great", "story", "server", "down")


def corpus() -> Snapshot:
    ids = list(range(1, 41))
    index = pd.DataFrame(
        {
            "review_id": ids,
            "appid": [GAME if rid <= 30 else OTHER for rid in ids],
            "day": [20261001 + rid % 10 for rid in ids],
            "created_ts": [1_790_000_000 + rid for rid in ids],
            "language": ["english"] * 40,
            "voted_up": [rid % 2 == 0 for rid in ids],
            "playtime_bucket": ["2h_10h"] * 40,
            "first_seen_run_id": [1] * 40,
            "version": [pd.Timestamp("2026-10-01")] * 40,
            "aspects": ["performance" if rid % 3 == 0 else "" for rid in ids],
            "aspect_model_version": ["s0-1"] * 40,
            "fingerprint": [f"{rid:064x}" for rid in ids],
            "rules_version": ["r1"] * 40,
        },
        columns=list(INDEX_COLUMNS),
    )
    texts = [" ".join(WORDS[(rid * k) % len(WORDS)] for k in (1, 3, 7)) for rid in ids]
    reviews = pd.DataFrame({"review_id": ids, "text": texts})
    frame = pd.DataFrame({"appid": [GAME, OTHER]})
    return Snapshot(reviews=reviews, games=frame, patches=frame, index=index)


def write_queries(path: Path) -> Path:
    rows = [
        ["t1", str(GAME), "english", "aspect", "fps drops", "", "", "", ""],
        ["t2", str(GAME), "english", "event", "server down", "20261003", "20261006", "true", ""],
    ]
    pd.DataFrame(rows, columns=list(COLUMNS)).to_csv(path, index=False, lineterminator="\n")
    return path


@pytest.fixture(scope="module")
def toy_collections(
    qdrant: QdrantClient, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[dict[str, OnnxEmbedder], dict[str, str]]]:
    root = tmp_path_factory.mktemp("runs")
    snapshot = corpus()
    texts = dict(zip(snapshot.reviews["review_id"], snapshot.reviews["text"], strict=True))
    embedders: dict[str, OnnxEmbedder] = {}
    collections: dict[str, str] = {}
    for name, mix in (("tiny-a", False), ("tiny-b", True)):
        spec = replace(CANDIDATES["e5-small"], name=name, dims=DIM, max_tokens=32)
        write_tiny_embedder(root / name, spec, dim=DIM, mix=mix)
        embedders[name] = OnnxEmbedder(root / name)
        cache = EmbeddingCache.for_embedder(root, embedders[name])
        embed_index_set(snapshot, embedders[name], cache)
        collections[name] = f"pp_test_runs_{name}_{root.name[-6:]}"
        ensure_collection(qdrant, collections[name], DIM)
        load_collection(qdrant, collections[name], snapshot.index, texts, cache, batch=16)
    yield embedders, collections
    for name in collections.values():
        qdrant.delete_collection(name)


@pytest.mark.qdrant
def test_every_system_runs_on_a_toy_corpus(
    qdrant: QdrantClient,
    toy_collections: tuple[dict[str, OnnxEmbedder], dict[str, str]],
    tmp_path: Path,
) -> None:
    embedders, collections = toy_collections
    queries = write_queries(tmp_path / "queries.csv")

    with pytest.raises(QueriesNotFrozenError):  # nothing runs on queries that aren't frozen
        run_systems(qdrant, queries, embedders, collection=collections.__getitem__,
                    games={GAME, OTHER})  # fmt: skip
    freeze_queries(queries, games={GAME, OTHER})
    runs = run_systems(
        qdrant, queries, embedders, collection=collections.__getitem__, games={GAME, OTHER}
    )

    rows = corpus().index.set_index("review_id")
    assert set(runs) == {(s, v) for s in systems(["tiny-a", "tiny-b"]) for v in VARIANTS}
    for (system, variant), frame in runs.items():
        assert list(frame.columns) == ["query_id", "rank", "review_id", "score"]
        for query_id, hits in frame.groupby("query_id"):
            assert hits["rank"].tolist() == list(range(1, len(hits) + 1)), (system, query_id)
            assert len(hits) <= 20
            assert all(rows.loc[rid, "appid"] == GAME for rid in hits["review_id"])
        assert set(frame["query_id"]) == {"t1", "t2"}, (system, variant)
        # t1 has no extra filters: its filtered run is its base run.
        base_t1 = runs[(system, "base")].query("query_id == 't1'").reset_index(drop=True)
        assert frame.query("query_id == 't1'").reset_index(drop=True).equals(base_t1)
        t2 = frame.loc[frame["query_id"] == "t2", "review_id"]
        if variant == "filtered":
            assert all(rows.loc[rid, "voted_up"] for rid in t2)
            assert all(20261003 <= rows.loc[rid, "day"] <= 20261006 for rid in t2)
    # BM25 doesn't depend on the embedder: one run, from the first candidate's collection.
    assert len(runs[("bm25", "base")]) > 0
