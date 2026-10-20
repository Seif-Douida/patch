"""The nightly sync of a collection with core.review_index (plan Task 14, design D5).

`plan_sync` compares the target with ops.qdrant_point, the mirror of what the stage has written;
`apply_sync` writes the changes to Qdrant, and to the mirror after each batch Qdrant confirms,
within a time budget; `read_back` fills an empty mirror from the collection (the first night after
the laptop's upload).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pytest
from qdrant_client import QdrantClient
from review_index import create_index_tables, index_row, mirror, write_target, write_texts
from sqlalchemy import Engine
from tiny_encoder import write_tiny_embedder

from patchpulse.db import tables
from patchpulse.retrieval.embedders import CANDIDATES, OnnxEmbedder
from patchpulse.retrieval.qdrant import DENSE, ensure_collection
from patchpulse.retrieval.sync import (
    BATCH,
    PointWriter,
    SyncStats,
    apply_sync,
    plan_sync,
    read_back,
)

pytestmark = pytest.mark.integration

DIM = 8
SPEC = replace(CANDIDATES["e5-small"], name="tiny-sync", dims=DIM, max_tokens=32)
COLLECTION = "reviews_v1"
NIGHT = datetime(2026, 10, 20)


def days_ago(n: float) -> datetime:
    return NIGHT - timedelta(days=n)


@pytest.fixture
def engine(migrated_engine: Engine) -> Engine:
    create_index_tables(migrated_engine)
    return migrated_engine


@pytest.fixture(scope="module")
def embedder(tmp_path_factory: pytest.TempPathFactory) -> OnnxEmbedder:
    root = tmp_path_factory.mktemp("sync")
    write_tiny_embedder(root / "tiny", SPEC, dim=DIM)
    return OnnxEmbedder(root / "tiny")


def write_mirror(
    engine: Engine, collection: str, rows: Iterable[Mapping[str, Any]], run_id: int = 1
) -> None:
    with engine.begin() as connection:
        connection.execute(
            tables.qdrant_point.insert(),
            [
                {
                    "collection": collection,
                    "review_id": row["review_id"],
                    "version": row["version"],
                    "fingerprint": row["fingerprint"],
                    "synced_run_id": run_id,
                }
                for row in rows
            ],
        )


def ids(frame: pd.DataFrame) -> list[int]:
    return [int(i) for i in frame["review_id"]]


def sync(
    engine: Engine,
    client: PointWriter,
    collection: str,
    embedder: OnnxEmbedder,
    *,
    run_id: int,
    batch: int = BATCH,
    deadline: float = float("inf"),
    clock: Callable[[], float] = time.monotonic,
) -> SyncStats:
    with engine.connect() as connection:
        plan = plan_sync(connection, collection=collection)
    return apply_sync(
        client,
        engine,
        plan,
        embedder,
        collection=collection,
        run_id=run_id,
        deadline=deadline,
        clock=clock,
        batch=batch,
    )


def test_sync_plan_covers_every_kind_of_change(engine: Engine) -> None:
    new = index_row(1, created=days_ago(1))
    newest = index_row(8, created=days_ago(0.5))
    edited_before = index_row(2, created=days_ago(30), aspects="performance")
    edited = index_row(2, created=days_ago(30), version=days_ago(2), aspects="performance")
    rescored_before = index_row(3, created=days_ago(10))
    rescored = index_row(3, created=days_ago(10), aspects="performance|content")
    champion_before = index_row(4, created=days_ago(12), aspects="content")
    champion = index_row(4, created=days_ago(12), aspects="content", model="s1a-2")
    patch_window = index_row(5, created=days_ago(400))  # an old review a new patch brings in
    unchanged = index_row(6, created=days_ago(5))
    out_of_cap = index_row(7, created=days_ago(90))
    write_target(engine, [new, newest, edited, rescored, champion, patch_window, unchanged])
    write_mirror(
        engine, COLLECTION, [edited_before, rescored_before, champion_before, unchanged, out_of_cap]
    )
    # Another collection's mirror (a rebuild in progress) is no part of this collection's plan.
    write_mirror(engine, "reviews_v2", [new, index_row(9, created=days_ago(3))])

    with engine.connect() as connection:
        plan = plan_sync(connection, collection=COLLECTION)

    assert ids(plan.upserts) == [8, 1, 2, 5]  # newest first, the patch window's old review last
    assert sorted(ids(plan.field_updates)) == [3, 4]
    assert plan.deletes == [7]
    assert plan.size == 7
    upserts = plan.upserts.set_index("review_id")
    assert upserts.loc[2, "version"] == days_ago(2)
    assert upserts.loc[8, "fingerprint"] == newest["fingerprint"]
    updates = plan.field_updates.set_index("review_id")
    assert updates.loc[3, "aspects"] == "performance|content"
    assert updates.loc[3, "fingerprint"] == rescored["fingerprint"]
    assert updates.loc[4, "aspect_model_version"] == "s1a-2"


class Delegating:
    """A client that passes every write to the real one; subclasses change `upsert`."""

    def __init__(self, client: QdrantClient) -> None:
        self.client = client

    def upsert(self, collection_name: str, points: Any, *, wait: bool) -> Any:
        return self.client.upsert(collection_name, points, wait=wait)

    def batch_update_points(
        self, collection_name: str, update_operations: Any, *, wait: bool
    ) -> Any:
        return self.client.batch_update_points(collection_name, update_operations, wait=wait)

    def delete(self, collection_name: str, points_selector: Any, *, wait: bool) -> Any:
        return self.client.delete(collection_name, points_selector, wait=wait)


@pytest.mark.qdrant
def test_apply_sync_end_to_end(
    engine: Engine,
    qdrant: QdrantClient,
    new_collection: Callable[[], str],
    embedder: OnnxEmbedder,
) -> None:
    collection = new_collection()
    ensure_collection(qdrant, collection, DIM)
    rows = {
        rid: index_row(rid, created=days_ago(10 - rid), aspects="performance" if rid % 2 else "")
        for rid in range(1, 6)
    }
    write_target(engine, rows.values())
    write_texts(engine, {rid: f"[b]good game[/b] number {rid}" for rid in rows})

    first = sync(engine, qdrant, collection, embedder, run_id=1)

    assert first == SyncStats(upserted=5, updated=0, deleted=0, remaining=0)
    (point,) = qdrant.retrieve(collection, [3], with_payload=True, with_vectors=True)
    payload = point.payload or {}
    assert payload["text"] == "good game number 3"  # cleaned
    assert payload["version"] == rows[3]["version"].isoformat()
    assert payload["aspects"] == ["performance"]
    assert isinstance(point.vector, dict)
    vector = point.vector[DENSE]
    fingerprint_at_upsert = rows[3]["fingerprint"]

    # The next night: review 1 edited, 3 re-scored by a new champion, 4 out of the cap, 6 new.
    rows[1] = index_row(1, created=days_ago(9), version=days_ago(1), aspects="performance")
    rows[3] = index_row(3, created=days_ago(7), aspects="performance|stability_bugs", model="s1a-2")
    del rows[4]
    rows[6] = index_row(6, created=days_ago(0.5))
    write_target(engine, rows.values())
    write_texts(engine, {1: "bad game crashes", 6: "great story"})

    second = sync(engine, qdrant, collection, embedder, run_id=2)

    assert second == SyncStats(upserted=2, updated=1, deleted=1, remaining=0)
    assert sorted(int(p.id) for p in qdrant.scroll(collection, limit=100)[0]) == [1, 2, 3, 5, 6]
    (edited,) = qdrant.retrieve(collection, [1], with_payload=True)
    assert (edited.payload or {})["text"] == "bad game crashes"
    assert (edited.payload or {})["version"] == days_ago(1).isoformat()
    (rescored,) = qdrant.retrieve(collection, [3], with_payload=True, with_vectors=True)
    payload = rescored.payload or {}
    assert payload["aspects"] == ["performance", "stability_bugs"]
    assert payload["aspect_model_version"] == "s1a-2"
    # A field update keeps the text, the vector (nothing embedded again) and the fingerprint
    # stored at the upsert; the mirror records the new one.
    assert payload["fingerprint"] == fingerprint_at_upsert
    assert payload["text"] == "good game number 3"
    assert isinstance(rescored.vector, dict)
    assert rescored.vector[DENSE] == vector
    assert mirror(engine, collection) == {
        rid: (row["version"], row["fingerprint"], 2 if rid in (1, 3, 6) else 1)
        for rid, row in rows.items()
    }
    with engine.connect() as connection:
        assert plan_sync(connection, collection=collection).size == 0


@pytest.mark.qdrant
def test_read_back_fills_an_empty_mirror(
    engine: Engine,
    qdrant: QdrantClient,
    new_collection: Callable[[], str],
    embedder: OnnxEmbedder,
) -> None:
    # The laptop's upload writes points but no mirror; the first night reads them back.
    collection = new_collection()
    ensure_collection(qdrant, collection, DIM)
    rows = [index_row(rid, created=days_ago(rid), aspects="content") for rid in range(1, 8)]
    write_target(engine, rows)
    write_texts(engine, {int(row["review_id"]): "good game" for row in rows})
    sync(engine, qdrant, collection, embedder, run_id=1)
    with engine.begin() as connection:
        connection.execute(tables.qdrant_point.delete())
    # Another collection's rows don't make this collection's mirror non-empty.
    write_mirror(engine, "reviews_v2", rows[:1])

    assert read_back(qdrant, engine, collection, page=3) == 7

    # Read back, no run wrote these points: synced_run_id stays empty.
    assert mirror(engine, collection) == {
        int(row["review_id"]): (row["version"], row["fingerprint"], None) for row in rows
    }
    assert read_back(qdrant, engine, collection) == 0  # only ever into an empty mirror
    with engine.connect() as connection:
        assert plan_sync(connection, collection=collection).size == 0


class Recording(Delegating):
    def __init__(self, client: QdrantClient) -> None:
        super().__init__(client)
        self.updates: list[list[Any]] = []

    def batch_update_points(
        self, collection_name: str, update_operations: Any, *, wait: bool
    ) -> Any:
        self.updates.append(list(update_operations))
        return super().batch_update_points(collection_name, update_operations, wait=wait)


@pytest.mark.qdrant
def test_field_updates_set_each_new_aspect_set_once(
    engine: Engine,
    qdrant: QdrantClient,
    new_collection: Callable[[], str],
    embedder: OnnxEmbedder,
) -> None:
    # A new champion re-scores reviews 1, 2 and 3; 1 and 3 get the same labels. One operation per
    # new aspect set, over its reviews: a point's payload is rewritten once per operation.
    collection = new_collection()
    ensure_collection(qdrant, collection, DIM)
    rows = {rid: index_row(rid, created=days_ago(rid), aspects="content") for rid in range(1, 5)}
    write_target(engine, rows.values())
    write_texts(engine, dict.fromkeys(rows, "good game"))
    sync(engine, qdrant, collection, embedder, run_id=1)
    for rid, aspects in (
        (1, "performance|content"),
        (2, "stability_bugs"),
        (3, "performance|content"),
    ):
        rows[rid] = index_row(rid, created=days_ago(rid), aspects=aspects, model="s1a-2")
    write_target(engine, rows.values())
    client = Recording(qdrant)

    stats = sync(engine, client, collection, embedder, run_id=2)

    assert stats == SyncStats(upserted=0, updated=3, deleted=0, remaining=0)
    (request,) = client.updates
    sets = sorted(
        (sorted(int(i) for i in op.set_payload.points), op.set_payload.payload) for op in request
    )
    assert sets == [
        ([1, 3], {"aspects": ["performance", "content"], "aspect_model_version": "s1a-2",
                  "rules_version": "r1"}),
        ([2], {"aspects": ["stability_bugs"], "aspect_model_version": "s1a-2",
               "rules_version": "r1"}),
    ]  # fmt: skip
    stored = {int(p.id): p.payload or {} for p in qdrant.retrieve(collection, [1, 2, 3, 4])}
    assert [stored[rid]["aspects"] for rid in (1, 2, 3, 4)] == [
        ["performance", "content"],
        ["stability_bugs"],
        ["performance", "content"],
        ["content"],
    ]
    assert {rid: m[1] for rid, m in mirror(engine, collection).items()} == {
        rid: row["fingerprint"] for rid, row in rows.items()
    }


class LosesThirdReply(Delegating):
    """Qdrant writes the third batch, but its reply never arrives (a timeout after the write)."""

    calls = 0

    def upsert(self, collection_name: str, points: Any, *, wait: bool) -> Any:
        self.calls += 1
        result = self.client.upsert(collection_name, points, wait=wait)
        if self.calls == 3:
            raise TimeoutError("the read timed out")
        return result


@pytest.mark.qdrant
def test_a_failed_batch_leaves_the_mirror_behind_not_ahead(
    engine: Engine,
    qdrant: QdrantClient,
    new_collection: Callable[[], str],
    embedder: OnnxEmbedder,
) -> None:
    collection = new_collection()
    ensure_collection(qdrant, collection, DIM)
    write_target(engine, [index_row(rid, created=days_ago(rid)) for rid in range(1, 8)])
    write_texts(engine, dict.fromkeys(range(1, 8), "good game"))

    with pytest.raises(TimeoutError):
        sync(engine, LosesThirdReply(qdrant), collection, embedder, run_id=1, batch=2)

    # Exactly the first two batches, the newest reviews; Qdrant holds the third too.
    assert set(mirror(engine, collection)) == {1, 2, 3, 4}
    assert qdrant.count(collection, exact=True).count == 6
    resumed = sync(engine, qdrant, collection, embedder, run_id=2, batch=2)
    assert resumed == SyncStats(upserted=3, updated=0, deleted=0, remaining=0)
    assert set(mirror(engine, collection)) == set(range(1, 8))
    assert qdrant.count(collection, exact=True).count == 7


class Ticker:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TakesAMinute(Delegating):
    """Each upsert uses a minute of the budget."""

    def __init__(self, client: QdrantClient, clock: Ticker) -> None:
        super().__init__(client)
        self.clock = clock

    def upsert(self, collection_name: str, points: Any, *, wait: bool) -> Any:
        self.clock.now += 60
        return self.client.upsert(collection_name, points, wait=wait)


@pytest.mark.qdrant
def test_budget_stops_and_the_next_run_resumes(
    engine: Engine,
    qdrant: QdrantClient,
    new_collection: Callable[[], str],
    embedder: OnnxEmbedder,
) -> None:
    collection = new_collection()
    ensure_collection(qdrant, collection, DIM)
    write_target(engine, [index_row(rid, created=days_ago(rid)) for rid in range(1, 11)])
    write_texts(engine, dict.fromkeys(range(1, 11), "good game"))
    clock = Ticker()
    slow = TakesAMinute(qdrant, clock)

    spent = sync(engine, slow, collection, embedder, run_id=1, batch=2, deadline=0, clock=clock)
    assert spent == SyncStats(upserted=0, updated=0, deleted=0, remaining=10)
    assert mirror(engine, collection) == {}

    # Batches start at 0, 60 and 120 s; a fourth would start at 180 s, past the deadline.
    first = sync(engine, slow, collection, embedder, run_id=2, batch=2, deadline=150, clock=clock)

    assert first == SyncStats(upserted=6, updated=0, deleted=0, remaining=4)
    assert set(mirror(engine, collection)) == {1, 2, 3, 4, 5, 6}  # the newest went first
    second = sync(engine, qdrant, collection, embedder, run_id=3, batch=2)
    assert second == SyncStats(upserted=4, updated=0, deleted=0, remaining=0)
    assert qdrant.count(collection, exact=True).count == 10
