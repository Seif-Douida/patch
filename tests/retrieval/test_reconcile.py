"""Reconciliation and repair (plan Task 15, design D5): each night, core.review_index's count of
reviews per game and day against Qdrant's; the next night compares the ids of the days that
differed and corrects them."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest
from qdrant_client import QdrantClient, models
from review_index import (
    create_index_tables,
    index_row,
    mirror,
    sync,
    write_target,
    write_texts,
)
from sqlalchemy import Engine, text
from tiny_encoder import write_tiny_embedder

from patchpulse.retrieval.embedders import CANDIDATES, OnnxEmbedder
from patchpulse.retrieval.points import to_point
from patchpulse.retrieval.qdrant import ensure_collection
from patchpulse.retrieval.reconcile import (
    Reconciliation,
    RepairStats,
    mismatched_days,
    reconcile,
    repair,
)
from patchpulse.retrieval.sync import SyncStats

pytestmark = [pytest.mark.integration, pytest.mark.qdrant]

DIM = 8
SPEC = replace(CANDIDATES["e5-small"], name="tiny-reconcile", dims=DIM, max_tokens=32)
GAME, OTHER = 10, 20
NIGHT = datetime(2026, 10, 20, 12)


def days_ago(n: int) -> datetime:
    return NIGHT - timedelta(days=n)


def day(n: int) -> int:
    return int(days_ago(n).strftime("%Y%m%d"))


@pytest.fixture
def engine(migrated_engine: Engine) -> Engine:
    create_index_tables(migrated_engine)
    return migrated_engine


@pytest.fixture(scope="module")
def embedder(tmp_path_factory: pytest.TempPathFactory) -> OnnxEmbedder:
    root = tmp_path_factory.mktemp("reconcile")
    write_tiny_embedder(root / "tiny", SPEC, dim=DIM)
    return OnnxEmbedder(root / "tiny")


def reconciliation_rows(engine: Engine, run_id: int) -> dict[tuple[int, int], tuple[int, int]]:
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT appid, [day], sql_count, qdrant_count FROM ops.qdrant_reconciliation "
                "WHERE run_id = :run"
            ),
            {"run": run_id},
        )
        return {(r.appid, r.day): (r.sql_count, r.qdrant_count) for r in rows}


def test_reconciliation_counts_every_game_and_day(
    engine: Engine,
    qdrant: QdrantClient,
    new_collection: Callable[[], str],
    embedder: OnnxEmbedder,
) -> None:
    collection = new_collection()
    ensure_collection(qdrant, collection, DIM)
    rows = [
        index_row(rid, created=days_ago(rid % 3), appid=GAME if rid <= 6 else OTHER)
        for rid in range(1, 10)
    ]
    write_target(engine, rows)
    write_texts(engine, dict.fromkeys(range(1, 10), "good game"))
    sync(engine, qdrant, collection, embedder, run_id=1)

    result = reconcile(qdrant, engine, collection, run_id=1)

    assert result == Reconciliation(target=9, qdrant=9, difference=0, days_off=0)
    assert result.mismatch == 0
    expected: dict[tuple[int, int], int] = {}
    for row in rows:
        key = (int(row["appid"]), int(row["day"]))
        expected[key] = expected.get(key, 0) + 1
    assert reconciliation_rows(engine, 1) == {key: (n, n) for key, n in expected.items()}


def test_repair_fixes_a_mismatched_day(
    engine: Engine,
    qdrant: QdrantClient,
    new_collection: Callable[[], str],
    embedder: OnnxEmbedder,
) -> None:
    collection = new_collection()
    ensure_collection(qdrant, collection, DIM)
    rows = [index_row(rid, created=days_ago(1 if rid <= 4 else 2)) for rid in range(1, 9)]
    write_target(engine, rows)
    write_texts(engine, dict.fromkeys(range(1, 9), "good game"))
    sync(engine, qdrant, collection, embedder, run_id=1)
    # Behind the mirror's back: review 2 is lost, and a stray point appears on a day with no
    # reviews in the target.
    qdrant.delete(collection, models.PointIdsList(points=[2]), wait=True)
    stray = index_row(99, created=days_ago(5))
    qdrant.upsert(collection, [to_point(stray, "stray", np.full(DIM, 0.5, np.float32))], wait=True)

    first = reconcile(qdrant, engine, collection, run_id=1)

    assert first == Reconciliation(target=8, qdrant=8, difference=2, days_off=2)
    assert first.mismatch == pytest.approx(0.25)
    assert reconciliation_rows(engine, 1)[(GAME, day(1))] == (4, 3)
    assert reconciliation_rows(engine, 1)[(GAME, day(5))] == (0, 1)

    # The next night repairs those days, then syncs as usual.
    with engine.connect() as connection:
        days = mismatched_days(connection, collection, before_run=2)
    assert sorted(days) == [(GAME, day(5)), (GAME, day(1))]
    repaired = repair(qdrant, engine, collection, days, deadline=float("inf"), clock=time.monotonic)

    assert repaired == RepairStats(days=2, missing=1, extra=1)
    assert 2 not in mirror(engine, collection)  # so the plan puts it back
    assert qdrant.retrieve(collection, [99]) == []
    assert sync(engine, qdrant, collection, embedder, run_id=2) == SyncStats(
        upserted=1, updated=0, deleted=0, remaining=0
    )
    second = reconcile(qdrant, engine, collection, run_id=2)
    assert second == Reconciliation(target=8, qdrant=8, difference=0, days_off=0)
    with engine.connect() as connection:
        assert mismatched_days(connection, collection, before_run=3) == []
