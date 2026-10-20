"""Qdrant groundwork (plan Task 1, design D2): the collection's settings, the alias, BM25 computed
by Qdrant with each review's language, and exact counts per day for reconciliation."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from patchpulse.retrieval.points import to_point
from patchpulse.retrieval.qdrant import (
    PAYLOAD_INDEXES,
    STEAM_BM25,
    CollectionMismatchError,
    bm25_options,
    ensure_collection,
    facet_counts,
    make_client,
    point_alias,
)

# Every Steam language code in the 12 Oct snapshot (1.03M reviews of the 12 games).
SNAPSHOT_LANGUAGES = {
    "english", "schinese", "russian", "brazilian", "spanish", "german", "french", "koreana",
    "polish", "japanese", "tchinese", "turkish", "latam", "ukrainian", "italian", "czech", "thai",
    "dutch", "swedish", "portuguese", "hungarian", "finnish", "danish", "vietnamese", "norwegian",
    "indonesian", "greek", "romanian", "bulgarian", "arabic", "malay",
}  # fmt: skip
# A core.review_index row, as the upload sends it.
ROW = {
    "review_id": 1, "appid": 1, "day": 20261020, "created_ts": 1_792_000_000,
    "language": "english", "voted_up": True, "playtime_bucket": None, "first_seen_run_id": 1,
    "version": "2026-10-20T00:00:00", "aspects": "", "aspect_model_version": None,
    "fingerprint": "a" * 64, "rules_version": "r1",
}  # fmt: skip
NO_SPACES = {"schinese", "tchinese", "japanese", "koreana", "thai"}


def bm25(text: str, language: str) -> models.Document:
    return models.Document(text=text, model="qdrant/bm25", options=bm25_options(language))


# Unit


def test_bm25_options_for_every_steam_language() -> None:
    assert set(STEAM_BM25) >= SNAPSHOT_LANGUAGES
    for code in [*STEAM_BM25, "klingon"]:
        options = bm25_options(code)
        # Every option is spelled out: Qdrant assumes English for anything left unset.
        assert options.tokenizer is not None
        assert options.stemmer is not None
        assert options.stopwords is not None
        models.Document.model_validate(bm25("x", code).model_dump())
    for code in NO_SPACES:
        assert bm25_options(code).tokenizer == models.TokenizerType.MULTILINGUAL
        assert isinstance(bm25_options(code).stemmer, models.DisabledStemmerParams)
    assert bm25_options("french").stemmer == models.SnowballParams(
        type=models.Snowball.SNOWBALL, language=models.SnowballLanguage.FRENCH
    )
    assert bm25_options("brazilian").stopwords == models.Language.PORTUGUESE
    unknown = bm25_options("klingon")
    assert unknown.tokenizer == models.TokenizerType.WORD
    assert isinstance(unknown.stemmer, models.DisabledStemmerParams)
    assert unknown.stopwords == models.StopwordsSet(languages=[], custom=[])


@pytest.mark.filterwarnings("ignore:Api key is used with an insecure connection")
def test_client_never_shows_the_key() -> None:
    client = make_client("http://127.0.0.1:9", "hunter2-key")  # nothing listens on port 9

    assert "hunter2" not in repr(client)
    with pytest.raises(Exception) as error:  # noqa: PT011  (whatever the transport raises)
        client.get_collections()
    assert "hunter2" not in str(error.value)
    assert "hunter2" not in repr(error.value)


# Against the Qdrant container


@pytest.mark.qdrant
def test_collection_matches_the_design(
    qdrant: QdrantClient, new_collection: Callable[[], str]
) -> None:
    name = new_collection()

    ensure_collection(qdrant, name, 384)
    ensure_collection(qdrant, name, 384)  # an existing, matching collection is left as it is

    info = qdrant.get_collection(name)
    params = info.config.params
    assert isinstance(params.vectors, dict)
    dense = params.vectors["dense"]
    assert (dense.size, dense.distance, dense.on_disk) == (384, models.Distance.COSINE, True)
    assert isinstance(info.config.quantization_config, models.ScalarQuantization)
    scalar = info.config.quantization_config.scalar
    assert (scalar.type, scalar.always_ram) == (models.ScalarType.INT8, True)
    assert params.sparse_vectors is not None
    assert params.sparse_vectors["bm25"].modifier == models.Modifier.IDF
    assert params.on_disk_payload is True
    assert (info.config.hnsw_config.m, info.config.hnsw_config.ef_construct) == (16, 100)
    # D9's fallback for a tight 1 GB (plan Task 17): the BM25 index and the HNSW graph on disk.
    sparse_index = params.sparse_vectors["bm25"].index
    assert sparse_index is not None
    assert sparse_index.on_disk is True
    assert info.config.hnsw_config.on_disk is True
    schema = {field: index.data_type for field, index in info.payload_schema.items()}
    assert schema == PAYLOAD_INDEXES


@pytest.mark.qdrant
def test_existing_collection_with_other_settings_is_refused(
    qdrant: QdrantClient, new_collection: Callable[[], str]
) -> None:
    name = new_collection()
    ensure_collection(qdrant, name, 384)

    with pytest.raises(CollectionMismatchError, match=r"dense.*384.*256"):
        ensure_collection(qdrant, name, 256)

    qdrant.delete_payload_index(name, "aspects", wait=True)
    with pytest.raises(CollectionMismatchError, match="aspects"):
        ensure_collection(qdrant, name, 384)
    other = new_collection()
    ensure_collection(qdrant, other, 384)
    qdrant.update_collection(other, hnsw_config=models.HnswConfigDiff(on_disk=False))
    with pytest.raises(CollectionMismatchError, match="hnsw"):
        ensure_collection(qdrant, other, 384)
    info = qdrant.get_collection(name)
    assert isinstance(info.config.params.vectors, dict)
    assert info.config.params.vectors["dense"].size == 384  # never altered
    assert "aspects" not in info.payload_schema


@pytest.mark.qdrant
def test_alias_switch_is_atomic(
    qdrant: QdrantClient, new_collection: Callable[[], str], monkeypatch: pytest.MonkeyPatch
) -> None:
    old, new, alias = new_collection(), new_collection(), f"alias_{new_collection()}"
    for name in (old, new):
        ensure_collection(qdrant, name, 4)
    point_alias(qdrant, alias, old)
    calls: list[Any] = []
    real = qdrant.update_collection_aliases

    def record(**kwargs: Any) -> Any:
        calls.append(kwargs["change_aliases_operations"])
        return real(**kwargs)

    monkeypatch.setattr(qdrant, "update_collection_aliases", record)

    point_alias(qdrant, alias, new)

    assert len(calls) == 1  # one request: Qdrant applies its operations together
    assert [type(op) for op in calls[0]] == [
        models.DeleteAliasOperation,
        models.CreateAliasOperation,
    ]
    targets = [a.collection_name for a in qdrant.get_aliases().aliases if a.alias_name == alias]
    assert targets == [new]


@pytest.mark.qdrant
def test_server_side_bm25_follows_the_language_options(
    qdrant: QdrantClient, new_collection: Callable[[], str]
) -> None:
    name = new_collection()
    ensure_collection(qdrant, name, 4)
    reviews = [
        (1, "french", "Les graphismes sont magnifiques et la musique aussi"),
        (2, "english", "The graphics are gorgeous and the soundtrack too"),
        (3, "schinese", "游戏更新后经常崩溃"),
    ]
    qdrant.upsert(
        name,
        [
            models.PointStruct(id=rid, vector={"bm25": bm25(text, language)}, payload={})
            for rid, language, text in reviews
        ],
        wait=True,
    )

    def found(text: str, language: str) -> list[int]:
        hits = qdrant.query_points(name, query=bm25(text, language), using="bm25", limit=5)
        return [int(hit.id) for hit in hits.points]

    assert found("magnifique", "french") == [1]  # another inflection, the same French stem
    assert found("magnifique", "english") == []  # stemmed as English, it matches nothing
    assert found("崩溃", "schinese") == [3]  # "crash", cut out of a sentence


@pytest.mark.qdrant
def test_a_grpc_client_uploads_with_server_side_bm25(
    qdrant: QdrantClient, new_collection: Callable[[], str]
) -> None:
    # The laptop's upload goes over gRPC (plan Task 17): a vector is 1.5 KB of float32 instead of
    # 8.4 KB of JSON text, and the upload is bound by the home connection.
    name = new_collection()
    url = os.environ.get("PP_TEST_QDRANT_URL", "http://localhost:6333")
    client = make_client(url, os.environ.get("PP_TEST_QDRANT_API_KEY") or None, grpc=True)
    try:
        ensure_collection(client, name, 4)
        client.upsert(
            name,
            [
                to_point({**ROW, "review_id": rid}, text, np.full(4, 0.5, dtype=np.float32))
                for rid, text in ((1, "the servers are down again"), (2, "great story"))
            ],
            wait=True,
        )
    finally:
        client.close()

    hits = qdrant.query_points(name, query=bm25("server", "english"), using="bm25", limit=5)
    assert [int(hit.id) for hit in hits.points] == [1]
    assert qdrant.count(name, exact=True).count == 2


@pytest.mark.qdrant
def test_every_language_option_is_accepted_by_qdrant(
    qdrant: QdrantClient, new_collection: Callable[[], str]
) -> None:
    name = new_collection()
    ensure_collection(qdrant, name, 4)
    codes = [*sorted(STEAM_BM25), "klingon"]

    qdrant.upsert(
        name,
        [
            models.PointStruct(id=i, vector={"bm25": bm25("crash after update", code)}, payload={})
            for i, code in enumerate(codes, start=1)
        ],
        wait=True,
    )

    assert qdrant.count(name, exact=True).count == len(codes)
    for code in codes:
        qdrant.query_points(name, query=bm25("crash", code), using="bm25", limit=1)


@pytest.mark.qdrant
def test_facet_counts_are_exact(qdrant: QdrantClient, new_collection: Callable[[], str]) -> None:
    name = new_collection()
    ensure_collection(qdrant, name, 4)
    expected: dict[int, int] = {}
    payloads: list[dict[str, int]] = []
    for i in range(1_200):  # more days than any game has in the index (about three years)
        day = 20230101 + i
        expected[day] = 1 + i % 3
        payloads += [{"appid": 10, "day": day}] * expected[day]
    payloads.append({"appid": 20, "day": 20230101})
    points = [
        models.PointStruct(id=i, vector={}, payload=payload)
        for i, payload in enumerate(payloads, start=1)
    ]
    for start in range(0, len(points), 1_000):
        qdrant.upsert(name, points[start : start + 1_000], wait=True)

    assert facet_counts(qdrant, name, 10) == expected
    assert facet_counts(qdrant, name, 20) == {20230101: 1}
    assert facet_counts(qdrant, name, 30) == {}
