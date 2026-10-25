"""search_reviews, the model's review search (plan Task 4, design D3): Phase 3's hybrid search,
always within the game, the window and the bundle's data version; the model sees snippets, the
trace keeps the full text for V3 and V7."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from agent_fixtures import DATA_VERSION, GAME, PATCH_DAY, make_bundle
from numpy.typing import NDArray
from qdrant_client import QdrantClient, models
from tiny_encoder import write_tiny_embedder

from patchpulse.agent.search import MAX_HITS, SNIPPET_CHARS, ReviewSearch, open_pinned_search
from patchpulse.agent.tools import Tools, Trace
from patchpulse.retrieval.embedders import CANDIDATES
from patchpulse.retrieval.pin import pin_for, write_pin

WINDOW = (PATCH_DAY, PATCH_DAY + timedelta(days=6))
TINY = replace(CANDIDATES["e5-small"], name="tiny-index", dims=8, max_tokens=32)


class FakeEmbedder:
    def embed_queries(self, texts: list[str], *, batch: int = 32) -> NDArray[np.float32]:
        return np.ones((len(texts), 4), dtype=np.float32)


class FakeQdrant:
    """Answers every query with the given points (id, score, text); records the calls."""

    def __init__(self, hits: list[tuple[int, float, str]]) -> None:
        self.hits = hits
        self.calls: list[dict[str, Any]] = []

    def query_points(self, collection: str, **kwargs: Any) -> Any:
        self.calls.append({"collection": collection, **kwargs})
        points = [
            SimpleNamespace(id=rid, score=score, payload={"text": text})
            for rid, score, text in self.hits[: kwargs["limit"]]
        ]
        return SimpleNamespace(points=points)


def tools_with(client: FakeQdrant, trace: Trace | None = None) -> Tools:
    search = ReviewSearch(client, "reviews", FakeEmbedder())
    return Tools(make_bundle(), trace or Trace(), search=search)


def conditions(call: dict[str, Any]) -> dict[str, Any]:
    """The hybrid query's filter (the same on both prefetches), by field."""
    dense, sparse = call["prefetch"]
    assert dense.filter == sparse.filter
    return {c.key: c for c in dense.filter.must}


def test_the_filter_carries_the_game_window_and_data_version() -> None:
    client = FakeQdrant([(11, 0.5, "Crashes on load since the patch")])

    tools_with(client).search_reviews(
        GAME,
        "crashes on load",
        WINDOW,
        voted_up=False,
        aspect="stability_bugs",
        language="english",
        min_playtime=10,
    )

    (call,) = client.calls
    where = conditions(call)
    assert where["appid"].match == models.MatchValue(value=GAME)
    assert (where["day"].range.gte, where["day"].range.lte) == (20260910, 20260916)
    assert where["first_seen_run_id"].range.lte == DATA_VERSION
    assert where["voted_up"].match == models.MatchValue(value=False)
    assert where["aspects"].match == models.MatchValue(value="stability_bugs")
    assert where["language"].match == models.MatchValue(value="english")
    assert where["playtime_bucket"].match == models.MatchAny(
        any=["10h_50h", "50h_200h", "200h_plus"]
    )
    assert call["limit"] == 8


def test_snippets_are_cut_and_the_trace_keeps_the_full_text() -> None:
    long_text = " ".join(f"word{i}" for i in range(300))  # about 2,000 characters
    client = FakeQdrant([(11, 0.5, long_text), (12, 0.3333333, "Short and fine")])
    trace = Trace()

    result = tools_with(client, trace).search_reviews(GAME, "words", WINDOW)

    first, second = result["hits"]
    assert (first["review_id"], first["score"]) == (11, 0.5)
    assert len(first["snippet"]) <= SNIPPET_CHARS
    assert long_text.startswith(first["snippet"])  # a quote from the snippet is in the text
    assert (second["snippet"], second["score"]) == ("Short and fine", 0.3333)
    assert trace.retrieved == {11: long_text, 12: "Short and fine"}
    assert long_text not in json.dumps(trace.calls[0].result)  # the model sees the snippet only
    assert trace.calls[0].args == {
        "appid": GAME,
        "query": "words",
        "window": ["2026-09-10", "2026-09-16"],
        "voted_up": None,
        "aspect": None,
        "language": None,
        "min_playtime": None,
        "k": 8,
    }


def test_k_is_capped() -> None:
    client = FakeQdrant([(i, 1 / i, f"review {i}") for i in range(1, 40)])

    result = tools_with(client).search_reviews(GAME, "review", WINDOW, k=50)

    assert client.calls[0]["limit"] == MAX_HITS
    assert len(result["hits"]) == MAX_HITS


def test_search_refuses_what_it_cannot_filter() -> None:
    client = FakeQdrant([])
    tools = tools_with(client)
    untracked = 1086940

    errors = [
        tools.search_reviews(GAME, "crashes", WINDOW, min_playtime=5),
        tools.search_reviews(GAME, "crashes", WINDOW, aspect="graphics"),
        tools.search_reviews(untracked, "crashes", WINDOW),
        tools.search_reviews(GAME, "crashes", (WINDOW[1], WINDOW[0])),
        Tools(make_bundle(), Trace()).search_reviews(GAME, "crashes", WINDOW),
    ]

    assert [e["error"] for e in errors] == [
        "min_playtime must be 2, 10, 50 or 200 hours: the index stores playtime buckets",
        "unknown aspect 'graphics'",
        "game 1086940 isn't tracked",
        "the range ends before it starts",
        "review search isn't available in this run",
    ]
    assert client.calls == []


# The same search on the local Qdrant container, a tiny collection.

DIM = 32


@pytest.fixture(scope="module")
def collection(
    qdrant: QdrantClient, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[ReviewSearch]:
    from patchpulse.models.snapshot import Snapshot
    from patchpulse.retrieval.embed import EmbeddingCache, embed_index_set
    from patchpulse.retrieval.embedders import CANDIDATES, OnnxEmbedder
    from patchpulse.retrieval.load import load_collection
    from patchpulse.retrieval.qdrant import ensure_collection
    from patchpulse.retrieval.rules import INDEX_COLUMNS

    root = tmp_path_factory.mktemp("agent-search")
    spec = replace(CANDIDATES["e5-small"], name="tiny-agent", dims=DIM, max_tokens=32)
    write_tiny_embedder(root / "tiny", spec, dim=DIM)
    embedder = OnnxEmbedder(root / "tiny")
    rows = [  # (review id, day, first seen run, text)
        (1, PATCH_DAY, DATA_VERSION, "zebra crossing crashes the game"),
        (2, PATCH_DAY + timedelta(days=1), DATA_VERSION, "servers down again"),
        (3, PATCH_DAY + timedelta(days=2), DATA_VERSION + 1, "zebra zebra zebra"),  # tomorrow's
        (4, PATCH_DAY - timedelta(days=20), DATA_VERSION, "zebra before the patch"),
    ]
    ids = [r[0] for r in rows]
    index = pd.DataFrame(
        {
            "review_id": ids,
            "appid": [GAME] * len(rows),
            "day": [int(d.strftime("%Y%m%d")) for _, d, _, _ in rows],
            "created_ts": [1_790_000_000 + rid for rid in ids],
            "language": ["english"] * len(rows),
            "voted_up": [False] * len(rows),
            "playtime_bucket": ["2h_10h"] * len(rows),
            "first_seen_run_id": [run for _, _, run, _ in rows],
            "version": [pd.Timestamp("2026-09-20")] * len(rows),
            "aspects": [""] * len(rows),
            "aspect_model_version": ["s0-1"] * len(rows),
            "fingerprint": [f"{rid:064x}" for rid in ids],
            "rules_version": ["r1"] * len(rows),
        },
        columns=list(INDEX_COLUMNS),
    )
    reviews = pd.DataFrame({"review_id": ids, "text": [t for _, _, _, t in rows]})
    games = pd.DataFrame({"appid": [GAME]})
    snapshot = Snapshot(reviews=reviews, games=games, patches=games, index=index)
    cache = EmbeddingCache.for_embedder(root, embedder)
    embed_index_set(snapshot, embedder, cache)
    name = f"pp_test_agent_{root.name[-8:]}"
    ensure_collection(qdrant, name, DIM)
    load_collection(qdrant, name, index, dict(zip(ids, reviews["text"], strict=True)), cache)
    yield ReviewSearch(qdrant, name, embedder)
    qdrant.delete_collection(name)


@pytest.mark.qdrant
def test_search_reviews_finds_the_seeded_review(collection: ReviewSearch) -> None:
    trace = Trace()
    tools = Tools(make_bundle(), trace, search=collection)

    result = tools.search_reviews(GAME, "zebra", WINDOW)

    found = [h["review_id"] for h in result["hits"]]
    assert found[0] == 1
    assert 3 not in found  # first seen after the bundle's data version
    assert 4 not in found  # outside the window
    assert trace.retrieved[1] == "zebra crossing crashes the game"


def test_window_days_are_inclusive() -> None:
    client = FakeQdrant([])
    one_day = (date(2026, 9, 12), date(2026, 9, 12))

    tools_with(client).search_reviews(GAME, "x", one_day)

    day = conditions(client.calls[0])["day"].range
    assert (day.gte, day.lte) == (20260912, 20260912)


def test_the_limits_are_the_plans() -> None:
    assert (SNIPPET_CHARS, MAX_HITS) == (400, 10)


def test_pinned_search_opens_the_pinned_collection_with_the_pinned_encoder(tmp_path: Path) -> None:
    # The nightly reports, the past-patch command and /ask search the collection
    # config/retrieval.yaml pins, with the encoder it names, as the index stage checks.
    encoder = tmp_path / "encoder"
    write_tiny_embedder(encoder, TINY, dim=TINY.dims)
    pin = tmp_path / "retrieval.yaml"
    write_pin(pin_for(encoder, 1).model_copy(update={"collection": "reviews_v9"}), pin)

    search = open_pinned_search("http://localhost:6333", None, encoder, pin_path=pin)
    try:
        assert search.collection == "reviews_v9"
    finally:
        search.client.close()

    other = tmp_path / "other"
    write_tiny_embedder(other, replace(TINY, name="other-index"), dim=TINY.dims)
    with pytest.raises(ValueError, match="isn't the one"):
        open_pinned_search("http://localhost:6333", None, other, pin_path=pin)
