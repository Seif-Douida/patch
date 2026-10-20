"""Keeping a collection in step with core.review_index, night by night (plan Task 14, design D5).

dbt rebuilds core.review_index, the reviews Qdrant should hold, every night; ops.qdrant_point,
the mirror, records what this stage has written to the collection. `plan_sync` compares the two
in one set-based query:
- a review in the target but not in the mirror, or at another version (a new review, an edit, an
  older review a new patch's window brings in), is embedded and upserted with its text and fields;
- one at the same version with another field fingerprint (re-scored aspects, a new champion) has
  its fields set without being embedded again (ADR-011);
- one in the mirror but no longer in the target (out of its game's cap) is deleted.
The plan is the ids; each batch reads its rows from core.review_index (and, to embed, its texts
from core.review_text), so a champion change's plan is a list of ids, not every row's fields.

A field update sets the three fields that can change at the same version: aspects, their model
version and the rules version. The others (game, day, creation time, language, thumbs, playtime,
first run) come from the review version itself, so they change with an upsert, and new index rules
build a new collection (design D2). Reviews given the same new values share one operation, as
Qdrant rewrites a point's payload once per operation that touches it (design D5: batched by aspect
set; measured on the 378k-point local collection, a champion change in ~7 minutes against ~22 one
point at a time). The point's stored `fingerprint` therefore stays the one written with its last
upsert; the mirror records the target's.

`apply_sync` works through the plan in batches until the deadline: deletes, then upserts newest
first, then field updates. Qdrant confirms each batch (`wait=True`) before its mirror rows are
written, in one transaction, so a failure leaves the mirror behind Qdrant and never ahead: the next
night's plan redoes what the mirror lacks, with the same ids, so a repeat is safe.

`read_back` fills an empty mirror from the collection itself: the first night after the laptop's
upload, which writes points but no mirror (design D4). Read back after field updates (a lost
mirror), the stored fingerprints make the plan set fields again, which is harmless.

The nightly job runs this without pandas (the jobs image has none).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from qdrant_client import QdrantClient, models
from sqlalchemy import Engine, text
from sqlalchemy.engine import Connection

from patchpulse.models.text import clean_text
from patchpulse.retrieval.embedders import PassageEmbedder
from patchpulse.retrieval.points import fields, to_point
from patchpulse.retrieval.rules import INDEX_COLUMNS

log = logging.getLogger("patchpulse.retrieval")

BATCH = 256
FIELD_BATCH = 2_048  # field updates: ~140 operations, ~2.2 s a request on the laptop
# The fields that can change while a review's version stays the same.
CHANGEABLE = ("aspects", "aspect_model_version", "rules_version")
SCROLL_PAGE = 2_000

_PLAN = """
SELECT
    CASE
        WHEN t.review_id IS NULL THEN 'delete'
        WHEN m.review_id IS NULL OR m.[version] <> t.[version] THEN 'upsert'
        ELSE 'update'
    END AS change,
    COALESCE(t.review_id, m.review_id) AS review_id
FROM core.review_index AS t
FULL OUTER JOIN (
    SELECT review_id, [version], fingerprint FROM ops.qdrant_point WHERE collection = :collection
) AS m ON m.review_id = t.review_id
WHERE t.review_id IS NULL OR m.review_id IS NULL
   OR m.[version] <> t.[version] OR m.fingerprint <> t.fingerprint
ORDER BY t.created_ts DESC, COALESCE(t.review_id, m.review_id) DESC
"""
# A batch's ids and mirror rows go in as one JSON document each, unpacked by OPENJSON.
_IDS = "SELECT CAST([value] AS BIGINT) FROM OPENJSON(:ids)"
_ROWS = (
    f"SELECT {', '.join(f'[{c}]' for c in INDEX_COLUMNS)} "  # noqa: S608 - constants
    f"FROM core.review_index WHERE review_id IN ({_IDS})"
)
_TEXTS = f"SELECT review_id, [text] FROM core.review_text WHERE review_id IN ({_IDS})"  # noqa: S608
_FORGET = f"DELETE FROM ops.qdrant_point WHERE collection = :collection AND review_id IN ({_IDS})"  # noqa: S608
_RECORD = """
INSERT INTO ops.qdrant_point (collection, review_id, [version], fingerprint, synced_run_id)
SELECT :collection, review_id, [version], fingerprint, :run_id
FROM OPENJSON(:rows) WITH (review_id BIGINT, [version] DATETIME2(0), fingerprint CHAR(64))
"""
_ANY = "SELECT TOP 1 1 FROM ops.qdrant_point WHERE collection = :collection"


class SyncError(RuntimeError):
    """The target or the collection holds something the sync can't write."""


class PointWriter(Protocol):
    """The writes the sync makes; QdrantClient has them."""

    def upsert(self, collection_name: str, points: Any, *, wait: bool) -> Any: ...

    def batch_update_points(
        self, collection_name: str, update_operations: Any, *, wait: bool
    ) -> Any: ...

    def delete(self, collection_name: str, points_selector: Any, *, wait: bool) -> Any: ...


@dataclass(frozen=True)
class SyncPlan:
    upserts: list[int]  # review ids, newest first
    field_updates: list[int]  # at the mirror's version, with other fields
    deletes: list[int]  # in the mirror but not the target

    @property
    def size(self) -> int:
        return len(self.upserts) + len(self.field_updates) + len(self.deletes)


@dataclass(frozen=True)
class SyncStats:
    upserted: int
    updated: int
    deleted: int
    remaining: int  # the plan's changes left for the next night


def plan_sync(connection: Connection, *, collection: str) -> SyncPlan:
    """Compare core.review_index with the collection's mirror rows."""
    changes: dict[str, list[int]] = {"upsert": [], "update": [], "delete": []}
    for change, review_id in connection.execute(text(_PLAN), {"collection": collection}):
        changes[change].append(int(review_id))
    return SyncPlan(
        upserts=changes["upsert"],
        field_updates=changes["update"],
        deletes=sorted(changes["delete"]),
    )


def _rows(connection: Connection, ids: Sequence[int]) -> list[dict[str, Any]]:
    """The target's rows for these ids, in their order."""
    found = {
        int(row["review_id"]): dict(row)
        for row in connection.execute(text(_ROWS), {"ids": json.dumps(list(ids))}).mappings()
    }
    missing = [rid for rid in ids if rid not in found]
    if missing:
        raise SyncError(f"review(s) {missing[:5]} left core.review_index during the sync")
    return [found[rid] for rid in ids]


def forget(connection: Connection, collection: str, ids: Sequence[int]) -> None:
    """Drop the mirror's rows for these ids: the next plan treats them as not written."""
    connection.execute(text(_FORGET), {"collection": collection, "ids": json.dumps(list(ids))})


def _record(
    connection: Connection, collection: str, rows: Sequence[dict[str, Any]], run_id: int | None
) -> None:
    """The rows' mirror entries, replacing any the collection had."""
    ids = [int(row["review_id"]) for row in rows]
    mirrored = [
        {
            "review_id": rid,
            "version": (
                row["version"].isoformat()
                if isinstance(row["version"], datetime)
                else str(row["version"])
            ),
            "fingerprint": str(row["fingerprint"]),
        }
        for rid, row in zip(ids, rows, strict=True)
    ]
    forget(connection, collection, ids)
    connection.execute(
        text(_RECORD), {"collection": collection, "run_id": run_id, "rows": json.dumps(mirrored)}
    )


def _points(
    connection: Connection, rows: Sequence[dict[str, Any]], embedder: PassageEmbedder
) -> list[models.PointStruct]:
    """The rows as points: their texts read from core.review_text, cleaned, then embedded."""
    ids = [int(row["review_id"]) for row in rows]
    found = {
        int(rid): str(body)
        for rid, body in connection.execute(text(_TEXTS), {"ids": json.dumps(ids)})
    }
    missing = [rid for rid in ids if rid not in found]
    if missing:
        raise SyncError(f"no text in core.review_text for review(s) {missing[:5]}")
    texts = [clean_text(found[rid]) for rid in ids]
    vectors = embedder.embed_passages(texts)
    return [to_point(row, texts[i], vectors[i]) for i, row in enumerate(rows)]


def _field_updates(rows: Sequence[dict[str, Any]]) -> list[models.SetPayloadOperation]:
    """One operation per distinct new value of the changeable fields, over its reviews."""
    groups: dict[str, tuple[dict[str, Any], list[models.ExtendedPointId]]] = {}
    for row in rows:
        values = {name: value for name, value in fields(row).items() if name in CHANGEABLE}
        key = json.dumps(values, sort_keys=True)
        groups.setdefault(key, (values, []))[1].append(int(row["review_id"]))
    return [
        models.SetPayloadOperation(set_payload=models.SetPayload(payload=values, points=ids))
        for values, ids in groups.values()
    ]


def apply_sync(
    client: PointWriter,
    engine: Engine,
    plan: SyncPlan,
    embedder: PassageEmbedder,
    *,
    collection: str,
    run_id: int,
    deadline: float,
    clock: Callable[[], float],
    batch: int = BATCH,
    field_batch: int = FIELD_BATCH,
) -> SyncStats:
    """Write the plan to the collection and the mirror, a batch at a time, until `deadline` (on
    `clock`'s scale); what's left waits for the next night's plan."""
    deleted = upserted = updated = 0
    for start in range(0, len(plan.deletes), batch):
        if clock() >= deadline:
            break
        ids = plan.deletes[start : start + batch]
        client.delete(collection, models.PointIdsList(points=list(ids)), wait=True)
        with engine.begin() as connection:
            forget(connection, collection, ids)
        deleted += len(ids)
    for start in range(0, len(plan.upserts), batch):
        if clock() >= deadline:
            break
        with engine.connect() as connection:
            rows = _rows(connection, plan.upserts[start : start + batch])
            points = _points(connection, rows, embedder)
        client.upsert(collection, points, wait=True)
        with engine.begin() as connection:
            _record(connection, collection, rows, run_id)
        upserted += len(rows)
    for start in range(0, len(plan.field_updates), field_batch):
        if clock() >= deadline:
            break
        with engine.connect() as connection:
            rows = _rows(connection, plan.field_updates[start : start + field_batch])
        client.batch_update_points(collection, _field_updates(rows), wait=True)
        with engine.begin() as connection:
            _record(connection, collection, rows, run_id)
        updated += len(rows)
    stats = SyncStats(
        upserted=upserted,
        updated=updated,
        deleted=deleted,
        remaining=plan.size - upserted - updated - deleted,
    )
    log.info(
        "%s: %d upserted, %d updated, %d deleted; %d left",
        collection,
        stats.upserted,
        stats.updated,
        stats.deleted,
        stats.remaining,
    )
    return stats


def read_back(
    client: QdrantClient, engine: Engine, collection: str, *, page: int = SCROLL_PAGE
) -> int:
    """Fill an empty mirror with the collection's ids, versions and fingerprints, in one
    transaction (a failure leaves it empty for the next night). Returns the rows read back: 0 when
    the mirror already holds rows for the collection."""
    read = 0
    with engine.begin() as connection:
        if connection.execute(text(_ANY), {"collection": collection}).first() is not None:
            return 0
        offset: Any = None
        while True:
            records, offset = client.scroll(
                collection,
                limit=page,
                offset=offset,
                with_payload=["version", "fingerprint"],
                with_vectors=False,
            )
            rows = []
            for record in records:
                stored = record.payload or {}
                if "version" not in stored or "fingerprint" not in stored:
                    raise SyncError(
                        f"point {record.id} in {collection} has no version or fingerprint"
                    )
                rows.append({"review_id": record.id, **stored})
            if rows:
                _record(connection, collection, rows, None)
            read += len(rows)
            if offset is None:
                break
    log.info("%s: %d points read back into the mirror", collection, read)
    return read
