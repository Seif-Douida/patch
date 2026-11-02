"""The laptop's catch-up of the nightly sync (runbook A, 1 Nov 2026).

The nightly's own plan and apply, run from the laptop in 5-minute chunks: each chunk starts only
when no night is running and the free cluster is calm (its collection green, its memory back
under the line), and the plan is recomputed before each, so a stop anywhere leaves what the next
run, or night, picks up.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

import pytest
from pydantic import SecretStr
from qdrant_client import QdrantClient
from review_index import create_index_tables, index_row, mirror, sync, write_target, write_texts
from sqlalchemy import Engine, text
from tiny_encoder import write_tiny_embedder

from patchpulse.retrieval.catchup import (
    CHUNK_S,
    MAX_WAIT_S,
    POLL_S,
    RESUME_BELOW,
    STOP_ABOVE,
    CatchUpOutcome,
    CloudCluster,
    ClusterState,
    Stop,
    catch_up,
)
from patchpulse.retrieval.embedders import CANDIDATES, OnnxEmbedder
from patchpulse.retrieval.qdrant import ensure_collection
from patchpulse.retrieval.sync import SyncError

pytestmark = [pytest.mark.integration, pytest.mark.qdrant]

DIM = 8
SPEC = replace(CANDIDATES["e5-small"], name="tiny-catchup", dims=DIM, max_tokens=32)
NIGHT = datetime(2026, 11, 1)
CALM = ClusterState(memory=0.47, green=True)


@pytest.fixture
def engine(migrated_engine: Engine) -> Engine:
    create_index_tables(migrated_engine)
    return migrated_engine


@pytest.fixture(scope="module")
def embedder(tmp_path_factory: pytest.TempPathFactory) -> OnnxEmbedder:
    root = tmp_path_factory.mktemp("catchup")
    write_tiny_embedder(root / "tiny", SPEC, dim=DIM)
    return OnnxEmbedder(root / "tiny")


@pytest.fixture
def collection(
    engine: Engine,
    qdrant: QdrantClient,
    new_collection: Callable[[], str],
    embedder: OnnxEmbedder,
) -> str:
    """A collection the nights have kept: one review synced by run 1, ten behind."""
    name = new_collection()
    ensure_collection(qdrant, name, DIM)
    old = index_row(100, created=NIGHT - timedelta(days=60))
    write_target(engine, [old])
    write_texts(engine, {100: "an old review"})
    sync(engine, qdrant, name, embedder, run_id=1)
    behind = [index_row(rid, created=NIGHT - timedelta(days=rid)) for rid in range(1, 11)]
    write_target(engine, [old, *behind])
    write_texts(engine, dict.fromkeys(range(1, 11), "good game"))
    return name


class Ticker:
    """A clock the test moves; `sleep` moves it too."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class Slow:
    """Passes every write to the real client; each upsert takes two minutes of the clock."""

    def __init__(self, client: QdrantClient, clock: Ticker) -> None:
        self.client = client
        self.clock = clock
        self.started: list[float] = []

    def upsert(self, collection_name: str, points: Any, *, wait: bool) -> Any:
        self.started.append(self.clock.now)
        self.clock.now += 120
        return self.client.upsert(collection_name, points, wait=wait)

    def batch_update_points(
        self, collection_name: str, update_operations: Any, *, wait: bool
    ) -> Any:
        return self.client.batch_update_points(collection_name, update_operations, wait=wait)

    def delete(self, collection_name: str, points_selector: Any, *, wait: bool) -> Any:
        return self.client.delete(collection_name, points_selector, wait=wait)


class Answers:
    """The cluster's state at each check, in order; the last one repeats."""

    def __init__(self, *states: ClusterState) -> None:
        self.states = list(states)
        self.checks = 0

    def check(self) -> ClusterState:
        state = self.states[min(self.checks, len(self.states) - 1)]
        self.checks += 1
        return state


def test_the_limits_are_the_runbooks() -> None:
    # Five minutes of upserts between checks; resume at 65% of RAM (the cluster sat at 47% with
    # 420k points on 1 Nov, ~60% projected at rest when caught up), stop above 80% (Phase 3's
    # upload watchdog paused at ~950 MiB of 1,167); give up after 20 minutes of waiting.
    assert (CHUNK_S, POLL_S, MAX_WAIT_S) == (300, 30, 1_200)
    assert (RESUME_BELOW, STOP_ABOVE) == (0.65, 0.80)


def test_catch_up_works_in_chunks_until_caught_up(
    engine: Engine, qdrant: QdrantClient, collection: str, embedder: OnnxEmbedder
) -> None:
    clock = Ticker()
    slow = Slow(qdrant, clock)

    outcome = catch_up(
        slow,
        engine,
        embedder,
        Answers(CALM),
        collection=collection,
        deadline=3_600,
        clock=clock,
        sleep=clock.sleep,
        batch=2,
    )

    # A chunk's batches start at 0, 120 and 240 s; a fourth would start past its 300 s. The
    # second chunk is planned again and finishes the last four.
    assert outcome == CatchUpOutcome(
        chunks=2, upserted=10, updated=0, deleted=0, remaining=0, stopped=Stop.CAUGHT_UP
    )
    assert slow.started == [0, 120, 240, 360, 480]
    # The laptop's rows carry no run id (as read-back rows don't): no night wrote them.
    synced = {rid: run for rid, (_, _, run) in mirror(engine, collection).items()}
    assert synced == {100: 1, **dict.fromkeys(range(1, 11), None)}
    assert qdrant.count(collection, exact=True).count == 11


def test_each_chunk_waits_for_a_calm_cluster(
    engine: Engine, qdrant: QdrantClient, collection: str, embedder: OnnxEmbedder
) -> None:
    clock = Ticker()
    slow = Slow(qdrant, clock)
    # Optimizing, then green but still above the resume line, then calm.
    cluster = Answers(ClusterState(memory=0.50, green=False), ClusterState(0.70, True), CALM)

    outcome = catch_up(
        slow,
        engine,
        embedder,
        cluster,
        collection=collection,
        deadline=3_600,
        clock=clock,
        sleep=clock.sleep,
        batch=2,
    )

    assert outcome.stopped is Stop.CAUGHT_UP
    assert slow.started[0] == 2 * POLL_S  # nothing was sent while the cluster was busy
    assert cluster.checks == 4  # three before the first chunk, one before the second


def test_memory_above_the_line_stops_the_catch_up(
    engine: Engine, qdrant: QdrantClient, collection: str, embedder: OnnxEmbedder
) -> None:
    clock = Ticker()

    outcome = catch_up(
        qdrant,
        engine,
        embedder,
        Answers(ClusterState(memory=0.81, green=True)),
        collection=collection,
        deadline=3_600,
        clock=clock,
        sleep=clock.sleep,
    )

    assert outcome == CatchUpOutcome(
        chunks=0, upserted=0, updated=0, deleted=0, remaining=10, stopped=Stop.MEMORY
    )
    assert set(mirror(engine, collection)) == {100}
    assert clock.now == 0  # it stops at once rather than waiting for memory to fall


def test_a_cluster_busy_for_20_minutes_stops_the_catch_up(
    engine: Engine, qdrant: QdrantClient, collection: str, embedder: OnnxEmbedder
) -> None:
    clock = Ticker()

    outcome = catch_up(
        qdrant,
        engine,
        embedder,
        Answers(ClusterState(memory=0.50, green=False)),
        collection=collection,
        deadline=7_200,
        clock=clock,
        sleep=clock.sleep,
    )

    assert outcome == CatchUpOutcome(
        chunks=0, upserted=0, updated=0, deleted=0, remaining=10, stopped=Stop.BUSY
    )
    assert clock.now == MAX_WAIT_S


def test_the_time_limit_stops_it_and_a_second_run_resumes(
    engine: Engine, qdrant: QdrantClient, collection: str, embedder: OnnxEmbedder
) -> None:
    clock = Ticker()

    first = catch_up(
        Slow(qdrant, clock),
        engine,
        embedder,
        Answers(CALM),
        collection=collection,
        deadline=200,
        clock=clock,
        sleep=clock.sleep,
        batch=2,
    )

    # Batches start at 0 and 120 s; the next would start at 240 s, past the 200 s limit.
    assert first == CatchUpOutcome(
        chunks=1, upserted=4, updated=0, deleted=0, remaining=6, stopped=Stop.TIME_LIMIT
    )
    second = catch_up(
        qdrant,
        engine,
        embedder,
        Answers(CALM),
        collection=collection,
        deadline=time.monotonic() + 600,
        clock=time.monotonic,
        sleep=time.sleep,
    )
    assert (second.upserted, second.remaining, second.stopped) == (6, 0, Stop.CAUGHT_UP)
    assert qdrant.count(collection, exact=True).count == 11


def test_it_never_runs_beside_a_live_night(
    engine: Engine, qdrant: QdrantClient, collection: str, embedder: OnnxEmbedder
) -> None:
    # Run 41 (31 Oct) failed with the database down and still reads `running`: not a live night.
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO ops.pipeline_run (kind, status, started_at) "
                "VALUES ('nightly', 'running', DATEADD(day, -1, SYSUTCDATETIME()))"
            )
        )
    clock = Ticker()

    def run() -> CatchUpOutcome:
        return catch_up(
            qdrant,
            engine,
            embedder,
            Answers(CALM),
            collection=collection,
            deadline=3_600,
            clock=clock,
            sleep=clock.sleep,
            batch=4,
        )

    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO ops.pipeline_run (kind, status, started_at) "
                "VALUES ('nightly', 'running', DATEADD(minute, -40, SYSUTCDATETIME()))"
            )
        )
    blocked = run()

    assert blocked == CatchUpOutcome(
        chunks=0, upserted=0, updated=0, deleted=0, remaining=10, stopped=Stop.NIGHTLY
    )
    assert set(mirror(engine, collection)) == {100}

    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE ops.pipeline_run SET status = 'succeeded' "
                "WHERE started_at >= DATEADD(hour, -1, SYSUTCDATETIME())"
            )
        )
    assert run().stopped is Stop.CAUGHT_UP  # the stuck row alone doesn't block


def test_it_refuses_an_empty_mirror(
    engine: Engine,
    qdrant: QdrantClient,
    new_collection: Callable[[], str],
    embedder: OnnxEmbedder,
) -> None:
    # An empty mirror makes every review an upsert: the full re-upload runbook A rules out.
    name = new_collection()
    ensure_collection(qdrant, name, DIM)
    write_target(engine, [index_row(1, created=NIGHT)])
    write_texts(engine, {1: "good game"})
    clock = Ticker()

    with pytest.raises(SyncError, match="no mirror rows"):
        catch_up(
            qdrant,
            engine,
            embedder,
            Answers(CALM),
            collection=name,
            deadline=3_600,
            clock=clock,
            sleep=clock.sleep,
        )
    assert qdrant.count(name, exact=True).count == 0


def test_cloud_cluster_reads_the_collection_and_the_memory(
    qdrant: QdrantClient, new_collection: Callable[[], str]
) -> None:
    name = new_collection()
    ensure_collection(qdrant, name, DIM)
    key = os.environ.get("PP_TEST_QDRANT_API_KEY")
    cluster = CloudCluster(
        qdrant,
        os.environ.get("PP_TEST_QDRANT_URL", "http://localhost:6333"),
        SecretStr(key) if key else None,
        name,
    )

    state = cluster.check()

    assert state.green
    assert 0 < state.memory < 1
