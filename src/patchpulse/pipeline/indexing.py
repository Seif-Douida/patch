"""The nightly `index` stage's work (plan Task 15, design D5); the runner wraps it.

In order:
1. read the collection back into an empty mirror (the first night after the laptop's upload);
2. plan the sync and apply it until the runner's deadline (PP_INDEX_BUDGET_MINUTES, within the
   night);
3. once the sync has caught up, repair the game-days the previous night's reconciliation found
   off, with what's left of the time;
4. reconcile the counts per game and day;
5. a fixed hybrid search on one game, to check that Qdrant answers.

The collection is the one config/retrieval.yaml pins, never the alias: the mirror is kept per
collection, so a rebuild gets its own. The encoder in PP_ENCODER_DIR must be the pinned one (the
SHA-256 of its embedder.json), and the collection must exist: the laptop uploads it (design D4);
this stage never creates one.

A Qdrant that doesn't answer (down, suspended, unreachable) raises QdrantUnreachableError, a
ConnectionError, so the runner can tell it from other failures without importing this module.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sqlalchemy import Engine

from patchpulse.config import Settings
from patchpulse.models.artifact import sha256_file
from patchpulse.models.text import CLEANING_VERSION
from patchpulse.retrieval.embedders import MANIFEST_FILE, OnnxEmbedder
from patchpulse.retrieval.pin import PIN_PATH, read_pin
from patchpulse.retrieval.qdrant import make_client
from patchpulse.retrieval.reconcile import (
    Reconciliation,
    RepairStats,
    mismatched_days,
    reconcile,
    repair,
)
from patchpulse.retrieval.rules import INDEX_RULES_VERSION
from patchpulse.retrieval.search import search
from patchpulse.retrieval.sync import SyncError, SyncStats, apply_sync, plan_sync, read_back

log = logging.getLogger("patchpulse.pipeline.indexing")

HEALTH_QUERY = "performance"
HEALTH_LANGUAGE = "english"
HEALTH_K = 5
# pp-nightly has 1 vCPU (infra/modules/jobs.bicep). ONNX Runtime's default starts a thread per
# core it sees, which under a 1-CPU quota embedded 3.2x slower in the jobs image (20 Oct).
INDEX_THREADS = 1


class QdrantUnreachableError(ConnectionError):
    """Qdrant didn't answer: no connection, a timeout, or a server error."""


@dataclass(frozen=True)
class IndexOutcome:
    collection: str
    read_back: int  # mirror rows read back from the collection (the first night only)
    repaired: RepairStats
    sync: SyncStats
    reconciliation: Reconciliation
    health_hits: int
    seconds: dict[str, float]  # per step

    @property
    def mismatch(self) -> float:
        return self.reconciliation.mismatch

    def record(self) -> dict[str, Any]:
        """The stage's record for ops.pipeline_run.stages: counts only."""
        return {
            "collection": self.collection,
            "read_back": self.read_back,
            "repaired": asdict(self.repaired),
            **asdict(self.sync),
            **asdict(self.reconciliation),
            "mismatch": round(self.mismatch, 6),
            "health_hits": self.health_hits,
            "seconds": self.seconds,
        }


def index_embedder(encoder_dir: Path) -> OnnxEmbedder:
    """The pinned encoder as the stage runs it: on pp-nightly's one vCPU."""
    return OnnxEmbedder(encoder_dir, threads=INDEX_THREADS)


def run_index(
    engine: Engine,
    settings: Settings,
    *,
    run_id: int,
    health_appid: int,
    clock: Callable[[], float],
    deadline: float,
    pin_path: Path = PIN_PATH,
) -> IndexOutcome:
    """The stage, with the client and encoder its settings name, working until `deadline` (the
    runner's: PP_INDEX_BUDGET_MINUTES, within the night)."""
    if settings.qdrant_url is None or settings.encoder_dir is None:
        raise ValueError("the index stage needs PP_QDRANT_URL and PP_ENCODER_DIR")
    pin = read_pin(pin_path)
    # Design D2: new index rules or a new cleaning build a new collection, never rewrite this one.
    if pin.rules_version != INDEX_RULES_VERSION:
        raise SyncError(
            f"{pin.collection} has rules version {pin.rules_version}, the code "
            f"{INDEX_RULES_VERSION}"
        )
    if pin.cleaning_version != CLEANING_VERSION:
        raise SyncError(
            f"{pin.collection} has cleaning version {pin.cleaning_version}, the code "
            f"{CLEANING_VERSION}"
        )
    if sha256_file(settings.encoder_dir / MANIFEST_FILE) != pin.embedder_json_sha256:
        raise SyncError(f"the encoder in PP_ENCODER_DIR isn't the one {pin_path} pins")
    embedder = index_embedder(settings.encoder_dir)
    client = make_client(settings.qdrant_url, settings.qdrant_api_key)
    try:
        return _index(
            client,
            engine,
            embedder,
            collection=pin.collection,
            run_id=run_id,
            health_appid=health_appid,
            deadline=deadline,
            clock=clock,
        )
    except ResponseHandlingException as error:  # no connection, or a timeout
        raise QdrantUnreachableError(
            f"Qdrant didn't answer: {type(error.source).__name__}"
        ) from error
    except UnexpectedResponse as error:
        if error.status_code is None or error.status_code >= 500:
            raise QdrantUnreachableError(f"Qdrant answered HTTP {error.status_code}") from error
        raise
    finally:
        client.close()


def _index(
    client: QdrantClient,
    engine: Engine,
    embedder: OnnxEmbedder,
    *,
    collection: str,
    run_id: int,
    health_appid: int,
    deadline: float,
    clock: Callable[[], float],
) -> IndexOutcome:
    seconds: dict[str, float] = {}

    def step(name: str, started: float) -> None:
        seconds[name] = round(time.perf_counter() - started, 1)

    started = time.perf_counter()
    if not client.collection_exists(collection):
        raise SyncError(f"collection {collection} doesn't exist: the laptop uploads it (D4)")
    read = read_back(client, engine, collection)
    step("read_back", started)

    started = time.perf_counter()
    with engine.connect() as connection:
        plan = plan_sync(connection, collection=collection)
    stats = apply_sync(
        client,
        engine,
        plan,
        embedder,
        collection=collection,
        run_id=run_id,
        deadline=deadline,
        clock=clock,
    )
    step("sync", started)

    # Repair only once the sync has caught up: until then the mismatched days are mostly pending
    # upserts, which only the sync can fix. A repaired review the mirror forgets is upserted the
    # next night.
    started = time.perf_counter()
    repaired = RepairStats(days=0, missing=0, extra=0)
    if stats.remaining == 0:
        with engine.connect() as connection:
            days = mismatched_days(connection, collection, before_run=run_id)
        repaired = repair(client, engine, collection, days, deadline=deadline, clock=clock)
    step("repair", started)

    started = time.perf_counter()
    reconciliation = reconcile(client, engine, collection, run_id=run_id)
    step("reconcile", started)

    started = time.perf_counter()
    hits = search(
        client,
        collection,
        embedder,
        HEALTH_QUERY,
        appid=health_appid,
        language=HEALTH_LANGUAGE,
        k=HEALTH_K,
    )
    step("health", started)
    return IndexOutcome(
        collection=collection,
        read_back=read,
        repaired=repaired,
        sync=stats,
        reconciliation=reconciliation,
        health_hits=len(hits),
        seconds=seconds,
    )
