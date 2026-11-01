"""The laptop's catch-up of the nightly sync (runbook A; Seif, 1 Nov 2026).

After a large backlog the nights index ~11,000 reviews each: pp-nightly embeds on one vCPU. This
runs the same plan and apply (retrieval/sync.py, unchanged) from the laptop, embedding on every
core and sending over gRPC, against Azure SQL and the free cluster. It isn't the full re-upload
runbook A rules out: only what the mirror lacks is sent, and an empty mirror is refused.

It works in chunks of CHUNK_S. Before each one the plan is computed again, so a stop anywhere
leaves the rest to the next run or night (and apply_sync's log line counts what's truly left),
and the chunk starts only if:
- no night is running: a `running` pipeline run started in the last LIVE_RUN_MINUTES (pp-nightly's
  45-minute timeout and a margin; older `running` rows are runs that died with the database, like
  run 41 on 31 Oct);
- the cluster is calm: its collection green (the optimizer done merging segments, when memory
  peaks) and Qdrant's resident memory at or under RESUME_BELOW of the node's RAM. Above
  STOP_ABOVE it stops at once; a cluster still busy after MAX_WAIT_S stops it too.
The rows it writes to the mirror carry no run id, as read-back rows don't: no night wrote them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import SecretStr
from qdrant_client import QdrantClient, models
from sqlalchemy import Engine, text
from sqlalchemy.engine import Connection

from patchpulse.db.engine import connect_with_resume_retry
from patchpulse.retrieval.cloud import read_telemetry
from patchpulse.retrieval.embedders import PassageEmbedder
from patchpulse.retrieval.sync import (
    BATCH,
    PointWriter,
    SyncError,
    apply_sync,
    has_mirror,
    plan_sync,
)

log = logging.getLogger("patchpulse.retrieval")

CHUNK_S = 300.0
POLL_S = 30.0
MAX_WAIT_S = 1_200.0
# Shares of the node's RAM: the cluster sat at 47% with 420k points on 1 Nov (~60% projected at
# rest when caught up); Phase 3's upload watchdog paused at ~950 MiB of 1,167 (81%).
RESUME_BELOW = 0.65
STOP_ABOVE = 0.80
LIVE_RUN_MINUTES = 50

_LIVE_RUN = """
SELECT TOP 1 1 FROM ops.pipeline_run
WHERE status = 'running' AND started_at >= DATEADD(minute, :minutes, SYSUTCDATETIME())
"""


class Stop(StrEnum):
    CAUGHT_UP = "caught up"
    TIME_LIMIT = "time limit"
    NIGHTLY = "a nightly run is running"
    MEMORY = "Qdrant's memory passed the stop line"
    BUSY = "the cluster stayed busy"

    @property
    def ok(self) -> bool:
        """A stop that isn't a problem: done, or out of time with the rest left for later."""
        return self in (Stop.CAUGHT_UP, Stop.TIME_LIMIT)


@dataclass(frozen=True)
class ClusterState:
    memory: float  # Qdrant's resident memory, a share of the node's RAM
    green: bool  # the collection's optimizers are done


class Cluster(Protocol):
    def check(self) -> ClusterState: ...


class CloudCluster:
    """The free cluster's state: the collection's status and Qdrant's own memory figure."""

    def __init__(
        self, client: QdrantClient, url: str, api_key: SecretStr | None, collection: str
    ) -> None:
        self.client = client
        self.url = url
        self.api_key = api_key
        self.collection = collection

    def check(self) -> ClusterState:
        telemetry = read_telemetry(self.url, self.api_key, self.collection)
        status = self.client.get_collection(self.collection).status
        return ClusterState(
            memory=telemetry.resident_bytes / telemetry.ram_bytes,
            green=status == models.CollectionStatus.GREEN,
        )


@dataclass(frozen=True)
class CatchUpOutcome:
    chunks: int
    upserted: int
    updated: int
    deleted: int
    remaining: int  # the plan's changes left when it stopped
    stopped: Stop


def nightly_running(connection: Connection) -> bool:
    """Whether a pipeline run started in the last LIVE_RUN_MINUTES is still running."""
    found = connection.execute(text(_LIVE_RUN), {"minutes": -LIVE_RUN_MINUTES}).first()
    return found is not None


def catch_up(
    client: PointWriter,
    engine: Engine,
    embedder: PassageEmbedder,
    cluster: Cluster,
    *,
    collection: str,
    deadline: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    batch: int = BATCH,
) -> CatchUpOutcome:
    """Sync the collection chunk by chunk until it's caught up or a stop condition holds."""
    with connect_with_resume_retry(engine) as connection:  # wakes a paused database
        if not has_mirror(connection, collection):
            raise SyncError(
                f"no mirror rows for {collection}: a night reads the collection back first; "
                "catching up now would send every point again"
            )
    chunks = upserted = updated = deleted = 0
    while True:
        with connect_with_resume_retry(engine) as connection:
            plan = plan_sync(connection, collection=collection)
            live = nightly_running(connection)
        stop: Stop | None
        if plan.size == 0:
            stop = Stop.CAUGHT_UP
        elif live:
            stop = Stop.NIGHTLY
        elif clock() >= deadline:
            stop = Stop.TIME_LIMIT
        else:
            stop = _wait_for_calm(cluster, deadline=deadline, clock=clock, sleep=sleep)
        if stop is not None:
            log.info(
                "%s: %s after %d chunk(s): %d upserted, %d updated, %d deleted; %d left",
                collection,
                stop.value,
                chunks,
                upserted,
                updated,
                deleted,
                plan.size,
            )
            return CatchUpOutcome(chunks, upserted, updated, deleted, plan.size, stop)
        stats = apply_sync(
            client,
            engine,
            plan,
            embedder,
            collection=collection,
            run_id=None,
            deadline=min(clock() + CHUNK_S, deadline),
            clock=clock,
            batch=batch,
        )
        chunks += 1
        upserted += stats.upserted
        updated += stats.updated
        deleted += stats.deleted


def _wait_for_calm(
    cluster: Cluster,
    *,
    deadline: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> Stop | None:
    """None once the cluster is calm; the reason to stop otherwise."""
    waited = 0.0
    while True:
        state = cluster.check()
        if state.memory > STOP_ABOVE:
            log.warning(
                "Qdrant at %.0f%% of RAM, above the %.0f%% stop line",
                state.memory * 100,
                STOP_ABOVE * 100,
            )
            return Stop.MEMORY
        if state.green and state.memory <= RESUME_BELOW:
            return None
        if waited >= MAX_WAIT_S:
            return Stop.BUSY
        if clock() + POLL_S >= deadline:
            return Stop.TIME_LIMIT
        if waited == 0:
            log.info(
                "waiting for the cluster: collection %s, Qdrant at %.0f%% of RAM",
                "green" if state.green else "optimizing",
                state.memory * 100,
            )
        sleep(POLL_S)
        waited += POLL_S
