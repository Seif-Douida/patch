"""The index stage's encoder (plan Task 15; the first indexed run, 20 Oct)."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from qdrant_client import QdrantClient
from review_index import create_index_tables, index_row, sync, write_target, write_texts
from sqlalchemy import Engine
from tiny_encoder import write_tiny_embedder

from patchpulse.config import Settings
from patchpulse.models.text import CLEANING_VERSION
from patchpulse.pipeline.indexing import INDEX_THREADS, IndexOutcome, index_embedder, run_index
from patchpulse.retrieval.embedders import CANDIDATES, OnnxEmbedder
from patchpulse.retrieval.pin import PIN_PATH, pin_for, read_pin, write_pin
from patchpulse.retrieval.points import to_point
from patchpulse.retrieval.qdrant import ensure_collection
from patchpulse.retrieval.reconcile import RepairStats, reconcile
from patchpulse.retrieval.rules import INDEX_RULES_VERSION
from patchpulse.retrieval.sync import SyncError

TINY = replace(CANDIDATES["e5-small"], name="tiny-index", dims=8, max_tokens=32)
QDRANT_TEST_URL = os.environ.get("PP_TEST_QDRANT_URL", "http://localhost:6333")


def test_the_stage_embeds_on_one_thread(tmp_path: Path) -> None:
    # pp-nightly has 1 vCPU, but ONNX Runtime's default starts a thread per core it sees (12 on
    # the laptop): in the jobs image under a 1-CPU quota that embedded 3.2x slower than one thread,
    # the setting the speed floor was measured with (reports/retrieval/embedder-speed.json).
    write_tiny_embedder(tmp_path / "encoder", TINY, dim=TINY.dims)

    embedder = index_embedder(tmp_path / "encoder")

    assert INDEX_THREADS == 1
    assert embedder.session_options().intra_op_num_threads == INDEX_THREADS


@pytest.mark.parametrize(
    ("change", "named"),
    [({"rules_version": "r0"}, "rules version"), ({"cleaning_version": "c0"}, "cleaning version")],
)
def test_the_stage_refuses_a_pin_for_other_rules_or_cleaning(
    tmp_path: Path, change: dict[str, str], named: str
) -> None:
    # Design D2: new index rules or a new cleaning build a new collection; the nightly never
    # rewrites the pinned one under other rules (it checks before touching Qdrant).
    encoder = tmp_path / "encoder"
    write_tiny_embedder(encoder, TINY, dim=TINY.dims)
    pin = tmp_path / "retrieval.yaml"
    write_pin(pin_for(encoder, 1).model_copy(update=change), pin)
    settings = Settings.model_construct(qdrant_url="https://qdrant.invalid", encoder_dir=encoder)

    with pytest.raises(SyncError, match=named):
        run_index(
            cast(Engine, None),
            settings,
            run_id=1,
            health_appid=1,
            clock=time.monotonic,
            deadline=time.monotonic() + 60,
            pin_path=pin,
        )


def test_the_committed_pin_matches_the_code() -> None:
    pin = read_pin(PIN_PATH)
    assert (pin.rules_version, pin.cleaning_version) == (INDEX_RULES_VERSION, CLEANING_VERSION)


@pytest.mark.integration
@pytest.mark.qdrant
def test_repair_waits_until_the_sync_has_caught_up(
    migrated_engine: Engine, qdrant: QdrantClient, new_collection: Callable[[], str], tmp_path: Path
) -> None:
    # Reconciliation counts pending upserts as mismatches too; repairing those days before the
    # sync spent the sync's budget on days only the sync can fix (the Phase 3 review).
    engine = migrated_engine
    create_index_tables(engine)
    encoder = tmp_path / "encoder"
    write_tiny_embedder(encoder, TINY, dim=TINY.dims)
    collection = new_collection()
    ensure_collection(qdrant, collection, TINY.dims)
    pin = tmp_path / "retrieval.yaml"
    write_pin(pin_for(encoder, 1).model_copy(update={"collection": collection}), pin)
    settings = Settings.model_construct(qdrant_url=QDRANT_TEST_URL, encoder_dir=encoder)
    night = datetime(2026, 10, 20, 12)
    rows = [index_row(rid, created=night - timedelta(days=1 + rid % 3)) for rid in range(1, 301)]
    write_target(engine, rows[:1])
    write_texts(engine, {int(row["review_id"]): "good game" for row in rows})
    sync(engine, qdrant, collection, OnnxEmbedder(encoder), run_id=1)  # the mirror isn't empty
    write_target(engine, rows)  # 299 new reviews to upsert
    stray = index_row(999, created=night - timedelta(days=9))
    qdrant.upsert(collection, [to_point(stray, "stray", np.full(TINY.dims, 0.5, np.float32))])
    reconcile(qdrant, engine, collection, run_id=1)  # the stray's day and the pending days differ

    def index(run_id: int, clock: Callable[[], float], deadline: float) -> IndexOutcome:
        return run_index(
            engine,
            settings,
            run_id=run_id,
            health_appid=int(rows[0]["appid"]),
            clock=clock,
            deadline=deadline,
            pin_path=pin,
        )

    ticks = iter([0.0])  # the budget runs out after the first batch of 256
    behind = index(2, lambda: next(ticks, 100.0), deadline=50.0)

    assert (behind.sync.upserted, behind.sync.remaining) == (256, 43)
    assert behind.repaired == RepairStats(days=0, missing=0, extra=0)
    assert qdrant.retrieve(collection, [999]) != []

    caught_up = index(3, time.monotonic, deadline=float("inf"))

    assert (caught_up.sync.upserted, caught_up.sync.remaining) == (43, 0)
    assert (caught_up.repaired.missing, caught_up.repaired.extra) == (0, 1)
    assert qdrant.retrieve(collection, [999]) == []
    assert caught_up.reconciliation.difference == 0
