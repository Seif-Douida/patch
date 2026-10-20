"""Reconciliation and repair (plan Task 15, design D5).

Each night, after the sync, core.review_index's count of reviews per game and day is compared with
the collection's (exact facet counts on `day`, one request per game) and written to
ops.qdrant_reconciliation, every game-day. The mismatch is the sum of the absolute differences
over the target's total (the runner alerts above 0.5%).

The next night repairs the days that differed: it compares their ids in both, deletes the
collection's strays and drops the mirror rows of the reviews the collection lacks, so the sync
upserts them again. A day the budget doesn't reach stays off in that night's counts and is repaired
the night after.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models
from sqlalchemy import Engine, text
from sqlalchemy.engine import Connection

from patchpulse.retrieval.qdrant import FACET_LIMIT, facet_counts
from patchpulse.retrieval.sync import forget

log = logging.getLogger("patchpulse.retrieval")

_DAY_PAGE = 1_000

_COUNTS = "SELECT appid, [day], COUNT(*) FROM core.review_index GROUP BY appid, [day]"
_CLEAR = "DELETE FROM ops.qdrant_reconciliation WHERE run_id = :run_id AND collection = :collection"
_WRITE = """
INSERT INTO ops.qdrant_reconciliation (run_id, collection, appid, [day], sql_count, qdrant_count)
SELECT :run_id, :collection, appid, [day], sql_count, qdrant_count
FROM OPENJSON(:rows) WITH (appid INT, [day] INT, sql_count INT, qdrant_count INT)
"""
# The game-days the collection's latest reconciliation before this run found off.
_MISMATCHED = """
SELECT appid, [day] FROM ops.qdrant_reconciliation
WHERE collection = :collection AND sql_count <> qdrant_count
  AND run_id = (
      SELECT MAX(run_id) FROM ops.qdrant_reconciliation
      WHERE collection = :collection AND run_id < :before_run)
ORDER BY appid, [day]
"""
_DAY_IDS = "SELECT review_id FROM core.review_index WHERE appid = :appid AND [day] = :day"


@dataclass(frozen=True)
class Reconciliation:
    target: int  # reviews in core.review_index
    qdrant: int  # points in the collection
    difference: int  # the sum over game-days of |target - qdrant|
    days_off: int  # game-days whose counts differ

    @property
    def mismatch(self) -> float:
        if self.target == 0:
            return 0.0 if self.difference == 0 else 1.0
        return self.difference / self.target


@dataclass(frozen=True)
class RepairStats:
    days: int
    missing: int  # in the target but not the collection: the mirror forgets them
    extra: int  # in the collection but not the target: deleted


def _games(client: QdrantClient, collection: str) -> set[int]:
    response = client.facet(collection, key="appid", exact=True, limit=FACET_LIMIT)
    return {int(hit.value) for hit in response.hits}


def reconcile(
    client: QdrantClient, engine: Engine, collection: str, *, run_id: int
) -> Reconciliation:
    """Count both sides per game and day, and record every game-day for this run."""
    with engine.connect() as connection:
        target = {
            (int(appid), int(day)): int(n) for appid, day, n in connection.execute(text(_COUNTS))
        }
    held: dict[tuple[int, int], int] = {}
    for appid in sorted({appid for appid, _ in target} | _games(client, collection)):
        for day, n in facet_counts(client, collection, appid).items():
            held[(appid, day)] = n
    rows: list[dict[str, Any]] = [
        {
            "appid": appid,
            "day": day,
            "sql_count": target.get((appid, day), 0),
            "qdrant_count": held.get((appid, day), 0),
        }
        for appid, day in sorted(target.keys() | held.keys())
    ]
    with engine.begin() as connection:
        connection.execute(text(_CLEAR), {"run_id": run_id, "collection": collection})
        if rows:
            connection.execute(
                text(_WRITE),
                {"run_id": run_id, "collection": collection, "rows": json.dumps(rows)},
            )
    result = Reconciliation(
        target=sum(target.values()),
        qdrant=sum(held.values()),
        difference=sum(abs(r["sql_count"] - r["qdrant_count"]) for r in rows),
        days_off=sum(1 for r in rows if r["sql_count"] != r["qdrant_count"]),
    )
    log.info(
        "%s: %d points for %d reviews; %d off across %d game-days (%.3f%%)",
        collection,
        result.qdrant,
        result.target,
        result.difference,
        result.days_off,
        100 * result.mismatch,
    )
    return result


def mismatched_days(
    connection: Connection, collection: str, *, before_run: int
) -> list[tuple[int, int]]:
    """The (appid, day) pairs whose counts differed at the latest reconciliation before
    `before_run`."""
    rows = connection.execute(
        text(_MISMATCHED), {"collection": collection, "before_run": before_run}
    )
    return [(int(appid), int(day)) for appid, day in rows]


def _day_ids(client: QdrantClient, collection: str, appid: int, day: int) -> set[int]:
    where = models.Filter(
        must=[
            models.FieldCondition(key="appid", match=models.MatchValue(value=appid)),
            models.FieldCondition(key="day", match=models.MatchValue(value=day)),
        ]
    )
    ids: set[int] = set()
    offset: Any = None
    while True:
        records, offset = client.scroll(
            collection,
            scroll_filter=where,
            limit=_DAY_PAGE,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.update(int(record.id) for record in records)
        if offset is None:
            return ids


def repair(
    client: QdrantClient,
    engine: Engine,
    collection: str,
    days: Sequence[tuple[int, int]],
    *,
    deadline: float,
    clock: Callable[[], float],
) -> RepairStats:
    """Correct each game-day until `deadline`: delete the collection's strays, and forget the
    mirror rows of the reviews it lacks."""
    repaired = missing = extra = 0
    for appid, day in days:
        if clock() >= deadline:
            break
        with engine.connect() as connection:
            target = {
                int(rid)
                for (rid,) in connection.execute(text(_DAY_IDS), {"appid": appid, "day": day})
            }
        held = _day_ids(client, collection, appid, day)
        lacking, strays = sorted(target - held), sorted(held - target)
        if strays:
            client.delete(collection, models.PointIdsList(points=list(strays)), wait=True)
        if lacking or strays:
            with engine.begin() as connection:
                forget(connection, collection, lacking + strays)
        repaired += 1
        missing += len(lacking)
        extra += len(strays)
    if days:
        log.info(
            "%s: repaired %d of %d game-days (%d missing, %d extra)",
            collection,
            repaired,
            len(days),
            missing,
            extra,
        )
    return RepairStats(days=repaired, missing=missing, extra=extra)
