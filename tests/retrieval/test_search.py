"""Loading and searching a collection (plan Task 7, design D4, D6) on the Qdrant container: a toy
corpus of 40 reviews in two games, embedded by a tiny fake embedder, loaded the way the eval's
collections are, then searched in each mode with and without filters."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from qdrant_client import QdrantClient
from tiny_encoder import write_tiny_embedder

from patchpulse.models.snapshot import Snapshot
from patchpulse.models.text import CLEANING_VERSION
from patchpulse.retrieval.embed import EmbeddingCache, embed_index_set
from patchpulse.retrieval.embedders import CANDIDATES, OnnxEmbedder
from patchpulse.retrieval.load import load_collection
from patchpulse.retrieval.qdrant import ensure_collection
from patchpulse.retrieval.rules import INDEX_COLUMNS
from patchpulse.retrieval.search import SearchFilters, search

pytestmark = pytest.mark.qdrant

DIM = 32
SPEC = replace(CANDIDATES["e5-small"], name="tiny-search", dims=DIM, max_tokens=32)
WORDS = ("good", "bad", "game", "fps", "drops", "crashes", "great", "story", "server", "down")
GAME, OTHER = 10, 20
OBVIOUS = 7  # the only review with the word "zebra"


def text_of(rid: int) -> str:
    if rid == OBVIOUS:
        return "zebra"
    return " ".join(WORDS[(rid * k) % len(WORDS)] for k in (1, 3, 7))


def corpus() -> Snapshot:
    ids = list(range(1, 41))
    aspects = ["performance", "stability_bugs|performance", "", "content"]
    index = pd.DataFrame(
        {
            "review_id": ids,
            "appid": [GAME if rid <= 30 else OTHER for rid in ids],
            "day": [20261001 + rid % 10 for rid in ids],
            "created_ts": [1_790_000_000 + rid for rid in ids],
            "language": ["english"] * 40,
            "voted_up": [rid % 2 == 0 for rid in ids],
            "playtime_bucket": ["2h_10h"] * 40,
            "first_seen_run_id": [1 + rid // 10 for rid in ids],
            "version": [pd.Timestamp("2026-10-01")] * 40,
            "aspects": [aspects[rid % 4] for rid in ids],
            "aspect_model_version": ["s0-1"] * 40,
            "fingerprint": [f"{rid:064x}" for rid in ids],
            "rules_version": ["r1"] * 40,
        },
        columns=list(INDEX_COLUMNS),
    )
    reviews = pd.DataFrame({"review_id": ids, "text": [f"[b]{text_of(rid)}[/b]" for rid in ids]})
    frame = pd.DataFrame({"appid": [GAME, OTHER]})
    return Snapshot(reviews=reviews, games=frame, patches=frame, index=index)


@pytest.fixture(scope="module")
def loaded(
    qdrant: QdrantClient, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[QdrantClient, str, OnnxEmbedder]]:
    root = tmp_path_factory.mktemp("search")
    write_tiny_embedder(root / "tiny", SPEC, dim=DIM)
    embedder = OnnxEmbedder(root / "tiny")
    snapshot = corpus()
    cache = EmbeddingCache.for_embedder(root, embedder)
    embed_index_set(snapshot, embedder, cache)
    name = f"pp_test_search_{root.name[-8:]}"
    ensure_collection(qdrant, name, DIM)
    texts = dict(zip(snapshot.reviews["review_id"], snapshot.reviews["text"], strict=True))
    load_collection(qdrant, name, snapshot.index, texts, cache, batch=16)
    yield qdrant, name, embedder
    qdrant.delete_collection(name)


def ids(hits: list[Any]) -> list[int]:
    return [hit.review_id for hit in hits]


@pytest.mark.parametrize("mode", ["hybrid", "dense", "bm25"])
def test_each_mode_finds_the_obvious_match(
    loaded: tuple[QdrantClient, str, OnnxEmbedder], mode: str
) -> None:
    client, name, embedder = loaded

    hits = search(client, name, embedder, "zebra", appid=GAME, language="english", mode=mode)

    assert hits[0].review_id == OBVIOUS
    assert hits[0].text == "zebra"  # the cleaned text, from the payload
    assert len(hits) <= 20
    assert all(hit.review_id <= 30 for hit in hits)  # the game is always a filter


def test_filters_restrict_by_game_window_thumbs_aspect_and_data_version(
    loaded: tuple[QdrantClient, str, OnnxEmbedder],
) -> None:
    client, name, embedder = loaded
    rows = {int(row["review_id"]): row for row in corpus().index.to_dict("records")}

    def found(filters: SearchFilters, appid: int = GAME) -> list[int]:
        return ids(
            search(
                client, name, embedder, "fps drops", appid=appid, language="english",
                filters=filters, k=40,
            )
        )  # fmt: skip

    window = found(SearchFilters(start_day=20261003, end_day=20261005))
    assert window
    assert all(20261003 <= int(rows[r]["day"]) <= 20261005 for r in window)
    thumbs = found(SearchFilters(voted_up=True))
    assert thumbs
    assert all(rows[r]["voted_up"] for r in thumbs)
    aspect = found(SearchFilters(aspect="performance"))
    assert aspect
    assert all("performance" in str(rows[r]["aspects"]) for r in aspect)
    as_of = found(SearchFilters(as_of_run_id=2))
    assert as_of
    assert all(int(rows[r]["first_seen_run_id"]) <= 2 for r in as_of)
    assert set(found(SearchFilters(), appid=OTHER)) <= set(range(31, 41))
    assert found(SearchFilters(review_language="french")) == []  # every toy review is English
    assert len(found(SearchFilters(review_language="english"))) == 30


def test_search_handles_odd_queries(
    loaded: tuple[QdrantClient, str, OnnxEmbedder], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, name, embedder = loaded
    calls: list[str] = []
    real = client.query_points

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append("query")
        return real(*args, **kwargs)

    monkeypatch.setattr(client, "query_points", counted)

    def run(query: str, **kwargs: Any) -> list[Any]:
        return search(client, name, embedder, query, appid=GAME, language="english", **kwargs)

    assert run("") == []
    assert run("   \n\t ") == []
    assert calls == []  # never sent to Qdrant
    for mode in ("hybrid", "dense", "bm25"):
        assert isinstance(run("the and of", mode=mode), list)  # stopwords only
        assert isinstance(run("lag " * 500, mode=mode), list)  # 2,000 characters
        assert isinstance(run('"crashes" AND -fps OR (story)', mode=mode), list)
    thai = search(client, name, embedder, "เกม", appid=GAME, language="thai")
    assert isinstance(thai, list)  # no Thai reviews in the game: still a valid answer
    assert run("fps", filters=SearchFilters(start_day=20200101, end_day=20200102)) == []
    with pytest.raises(ValueError, match="mode"):
        run("fps", mode="fuzzy")


def test_exact_and_approximate_agree_on_a_tiny_collection(
    loaded: tuple[QdrantClient, str, OnnxEmbedder],
) -> None:
    client, name, embedder = loaded

    for query in ("fps drops", "server down", "great story"):
        approximate = search(
            client, name, embedder, query, appid=GAME, language="english", mode="dense", k=10
        )
        exact = search(
            client, name, embedder, query, appid=GAME, language="english", mode="dense", k=10,
            exact=True,
        )  # fmt: skip
        assert ids(approximate) == ids(exact)


class FailsOnce:
    """A client whose `batch`-th upsert fails once, as a dropped connection would."""

    def __init__(self, client: QdrantClient, batch: int) -> None:
        self.client, self.batch = client, batch

    def upsert(self, *args: Any, **kwargs: Any) -> Any:
        self.batch -= 1
        if self.batch == 0:
            raise ConnectionError("connection reset")
        return self.client.upsert(*args, **kwargs)


def test_load_resumes_and_is_idempotent(
    qdrant: QdrantClient, new_collection: Callable[[], str], tmp_path: Path
) -> None:
    write_tiny_embedder(tmp_path / "tiny", SPEC, dim=DIM)
    embedder = OnnxEmbedder(tmp_path / "tiny")
    snapshot = corpus()
    cache = EmbeddingCache.for_embedder(tmp_path, embedder)
    embed_index_set(snapshot, embedder, cache)
    texts = dict(zip(snapshot.reviews["review_id"], snapshot.reviews["text"], strict=True))
    name, progress = new_collection(), tmp_path / "progress.json"
    ensure_collection(qdrant, name, DIM)

    with pytest.raises(ConnectionError):
        load_collection(
            FailsOnce(qdrant, 3),
            name,
            snapshot.index,
            texts,
            cache,
            batch=8,
            progress=progress,
        )

    assert qdrant.count(name, exact=True).count == 16  # two confirmed batches
    assert json.loads(progress.read_text(encoding="utf-8"))["confirmed_through"] == 16
    resumed = load_collection(
        qdrant, name, snapshot.index, texts, cache, batch=8, progress=progress
    )
    assert (resumed.upserted, resumed.skipped) == (24, 16)
    again = load_collection(qdrant, name, snapshot.index, texts, cache, batch=8, progress=progress)
    assert (again.upserted, again.skipped) == (0, 40)
    # Another index set (a new pull) never resumes from this one's progress.
    other = load_collection(
        qdrant, name, snapshot.index.iloc[:-1], texts, cache, batch=8, progress=progress
    )
    assert (other.upserted, other.skipped) == (39, 0)
    assert qdrant.count(name, exact=True).count == 40  # the same ids: no duplicates
    # Nor another encoder with the same pull: its vectors must all go up (the Phase 3 review).
    load_collection(qdrant, name, snapshot.index, texts, cache, batch=8, progress=progress)
    another = EmbeddingCache(
        tmp_path, name=SPEC.name, model_sha256="f" * 64, cleaning_version=CLEANING_VERSION, dims=DIM
    )
    shutil.copytree(cache.directory, another.directory)
    switched = load_collection(
        qdrant, name, snapshot.index, texts, another, batch=8, progress=progress
    )
    assert (switched.upserted, switched.skipped) == (40, 0)
