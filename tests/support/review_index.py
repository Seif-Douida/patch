"""core.review_index and core.review_text as dbt builds them (Phase 3, design D1/D2), for tests of
the nightly index stage that need its inputs without running dbt."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from typing import Any

from sqlalchemy import Engine, text

from patchpulse.db import tables
from patchpulse.retrieval.embedders import PassageEmbedder
from patchpulse.retrieval.rules import INDEX_COLUMNS
from patchpulse.retrieval.sync import BATCH, PointWriter, SyncStats, apply_sync, plan_sync

_DDL = (
    """
    CREATE TABLE core.review_index (
        review_id BIGINT NOT NULL, appid INT NOT NULL, [day] INT NOT NULL,
        created_ts BIGINT NOT NULL, [language] NVARCHAR(32) NOT NULL, voted_up BIT NOT NULL,
        playtime_bucket NVARCHAR(16) NULL, first_seen_run_id BIGINT NOT NULL,
        [version] DATETIME2(0) NOT NULL, aspects NVARCHAR(400) NOT NULL,
        aspect_model_version NVARCHAR(64) NULL, rules_version NVARCHAR(8) NOT NULL,
        fingerprint CHAR(64) NOT NULL
    )
    """,
    """
    CREATE TABLE core.review_text (
        review_id BIGINT NOT NULL, [text] NVARCHAR(MAX) NOT NULL, last_seen_run_id BIGINT NOT NULL
    )
    """,
)
_FINGERPRINTED = tuple(c for c in INDEX_COLUMNS if c not in ("review_id", "version", "fingerprint"))


def create_index_tables(engine: Engine) -> None:
    with engine.begin() as connection:
        for statement in _DDL:
            connection.execute(text(statement))


def fingerprint(row: Mapping[str, Any]) -> str:
    """A fingerprint of the fields (not dbt's formula: the tests only need one that changes with
    them)."""
    joined = "|".join("" if row[c] is None else str(row[c]) for c in _FINGERPRINTED)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def index_row(
    review_id: int,
    *,
    created: datetime,
    version: datetime | None = None,
    appid: int = 10,
    aspects: str = "",
    model: str | None = "s0-1",
    **overrides: Any,
) -> dict[str, Any]:
    """A core.review_index row, its fingerprint computed from its fields."""
    row: dict[str, Any] = {
        "review_id": review_id,
        "appid": appid,
        "day": int(created.strftime("%Y%m%d")),
        "created_ts": int((created - datetime(1970, 1, 1)).total_seconds()),
        "language": "english",
        "voted_up": True,
        "playtime_bucket": "2h_10h",
        "first_seen_run_id": 1,
        "version": version or created,
        "aspects": aspects,
        "aspect_model_version": model,
        "rules_version": "r1",
        **overrides,
    }
    row["fingerprint"] = fingerprint(row)
    return row


def write_target(engine: Engine, rows: Iterable[Mapping[str, Any]]) -> None:
    """Replace core.review_index's rows, as dbt's nightly rebuild does."""
    columns = ", ".join(f"[{c}]" for c in INDEX_COLUMNS)
    values = ", ".join(f":{c}" for c in INDEX_COLUMNS)
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM core.review_index"))
        records = [dict(row) for row in rows]
        if records:
            connection.execute(
                text(f"INSERT INTO core.review_index ({columns}) VALUES ({values})"),
                records,
            )


def write_texts(engine: Engine, texts: Mapping[int, str]) -> None:
    """Upsert core.review_text's rows."""
    with engine.begin() as connection:
        for review_id, body in texts.items():
            connection.execute(
                text("DELETE FROM core.review_text WHERE review_id = :id"), {"id": review_id}
            )
            connection.execute(
                text(
                    "INSERT INTO core.review_text (review_id, [text], last_seen_run_id) "
                    "VALUES (:id, :text, 1)"
                ),
                {"id": review_id, "text": body},
            )


def mirror(engine: Engine, collection: str) -> dict[int, tuple[datetime, str, int | None]]:
    """ops.qdrant_point for a collection: review id -> (version, fingerprint, synced run id)."""
    point = tables.qdrant_point
    with engine.connect() as connection:
        rows = connection.execute(point.select().where(point.c.collection == collection)).mappings()
        return {
            int(r["review_id"]): (r["version"], r["fingerprint"], r["synced_run_id"]) for r in rows
        }


def write_mirror(
    engine: Engine, collection: str, rows: Iterable[Mapping[str, Any]], run_id: int = 1
) -> None:
    """ops.qdrant_point rows claiming these core.review_index rows were written."""
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


def sync(
    engine: Engine,
    client: PointWriter,
    collection: str,
    embedder: PassageEmbedder,
    *,
    run_id: int,
    batch: int = BATCH,
    deadline: float = float("inf"),
    clock: Callable[[], float] = time.monotonic,
) -> SyncStats:
    """One night's plan and apply, without a budget unless given one."""
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
