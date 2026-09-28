"""Write Steam data into the raw and ops tables (Alembic owns them; dbt reads them).

Reviews and news are upserted through a temp table: rows already stored get their
`last_seen_run_id` bumped, new rows (for reviews: new *versions*, keyed on the edit time) are
inserted. That's two plain statements rather than MERGE, which has known pitfalls on SQL Server.
Datetimes are stored as naive UTC.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import Column, Connection, MetaData, Table, delete, insert, select, update

from patchpulse.db import tables
from patchpulse.ingest.coverage import Coverage
from patchpulse.ingest.games import Game
from patchpulse.ingest.models import NewsItem, PriceSnapshot, RawPage, ReviewRecord
from patchpulse.ingest.patches import CLASSIFIER_VERSION, Classification


def naive_utc(moment: datetime) -> datetime:
    return moment.astimezone(UTC).replace(tzinfo=None)


def aware_utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC)


# The f-string SQL below interpolates only table and column names from patchpulse.db.tables
# (code, never data); every value travels as a bound parameter. Hence the S608 exemptions.


def _temp_copy(connection: Connection, source: Table, name: str) -> Table:
    """An empty #temp table shaped like `source`, for the duration of this connection."""
    connection.exec_driver_sql(f"DROP TABLE IF EXISTS [{name}]")
    connection.exec_driver_sql(
        f"SELECT TOP 0 * INTO [{name}] FROM [{source.schema}].[{source.name}]"  # noqa: S608
    )
    return Table(name, MetaData(), *(Column(c.name, c.type) for c in source.columns))


def _upsert(
    connection: Connection,
    target: Table,
    rows: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
    refresh: Sequence[str],
) -> int:
    """Insert unseen rows, refresh `refresh` columns on seen ones; return the number inserted."""
    if not rows:
        return 0
    temp = _temp_copy(connection, target, f"#incoming_{target.name}")
    connection.execute(insert(temp), list(rows))
    match = " AND ".join(f"t.[{k}] = i.[{k}]" for k in keys)
    assignments = ", ".join(f"[{c}] = i.[{c}]" for c in refresh)
    qualified = f"[{target.schema}].[{target.name}]"
    connection.exec_driver_sql(
        f"UPDATE t SET {assignments} FROM {qualified} t JOIN [{temp.name}] i ON {match}"  # noqa: S608
    )
    columns = ", ".join(f"[{c.name}]" for c in target.columns)
    inserted = connection.exec_driver_sql(
        f"INSERT INTO {qualified} ({columns}) SELECT {columns} FROM [{temp.name}] i "  # noqa: S608
        f"WHERE NOT EXISTS (SELECT 1 FROM {qualified} t WHERE {match})"
    ).rowcount
    connection.exec_driver_sql(f"DROP TABLE [{temp.name}]")
    return int(inserted)


def upsert_reviews(connection: Connection, records: Sequence[ReviewRecord], *, run_id: int) -> int:
    """Store review versions; returns how many versions were new."""
    unique: dict[tuple[int, datetime], dict[str, Any]] = {}
    for record in records:
        row = record.model_dump()
        row["timestamp_created"] = naive_utc(record.timestamp_created)
        row["timestamp_updated"] = naive_utc(record.timestamp_updated)
        row["first_seen_run_id"] = row["last_seen_run_id"] = run_id
        unique[(record.recommendation_id, row["timestamp_updated"])] = row
    return _upsert(
        connection,
        tables.steam_review,
        list(unique.values()),
        keys=("recommendation_id", "timestamp_updated"),
        # Votes move without the review being edited, so a re-seen version refreshes them.
        refresh=(
            "votes_up",
            "votes_funny",
            "weighted_vote_score",
            "comment_count",
            "last_seen_run_id",
        ),
    )


def upsert_news(connection: Connection, items: Sequence[NewsItem], *, run_id: int) -> int:
    """Store news items; edits refresh the stored copy. Returns how many items were new."""
    unique = {
        item.gid: {
            "gid": item.gid,
            "appid": item.appid,
            "title": item.title,
            "url": item.url,
            "contents": item.contents,
            "published_at": naive_utc(item.published_at),
            "feedname": item.feedname,
            "feedlabel": item.feedlabel,
            "tags": json.dumps(item.tags),
            "first_seen_run_id": run_id,
            "last_seen_run_id": run_id,
        }
        for item in items
    }
    return _upsert(
        connection,
        tables.steam_news,
        list(unique.values()),
        keys=("gid",),
        refresh=("title", "url", "contents", "tags", "feedlabel", "last_seen_run_id"),
    )


def upsert_classifications(
    connection: Connection, classified: Mapping[str, Classification], *, run_id: int
) -> None:
    """Store one classification per news gid for the current classifier version."""
    rows = [
        {
            "gid": gid,
            "classifier_version": CLASSIFIER_VERSION,
            "patch_type": result.patch_type.value,
            "is_treatment_candidate": result.is_treatment_candidate,
            "rule": result.rule,
            "ambiguous": result.ambiguous,
            "run_id": run_id,
        }
        for gid, result in classified.items()
    ]
    _upsert(
        connection,
        tables.news_classification,
        rows,
        keys=("gid", "classifier_version"),
        refresh=("patch_type", "is_treatment_candidate", "rule", "ambiguous", "run_id"),
    )


def record_page(connection: Connection, page: RawPage, *, run_id: int) -> None:
    connection.execute(
        insert(tables.steam_page).values(
            run_id=run_id,
            source=page.source,
            appid=page.appid,
            request=json.dumps(dict(page.request), sort_keys=True),
            http_status=page.http_status,
            payload=json.dumps(page.payload, ensure_ascii=False),
            n_items=page.n_items,
        )
    )


def upsert_games(connection: Connection, games: Sequence[Game], *, run_id: int) -> None:
    for game in games:
        values = {
            "name": game.name,
            "genres": ", ".join(game.genres),
            "role": game.role.value,
            "backfill_start": game.backfill_start,
            "release_date": game.release_date,
            "developer": game.developer,
            "publisher": game.publisher,
            "updated_run_id": run_id,
        }
        table = tables.game
        changed = connection.execute(
            update(table).where(table.c.appid == game.appid).values(**values)
        ).rowcount
        if not changed:
            connection.execute(insert(table).values(appid=game.appid, **values))


def upsert_prices(
    connection: Connection,
    snapshots: Mapping[int, PriceSnapshot | None],
    *,
    snapshot_date: date,
    run_id: int,
) -> None:
    """One row per game per day; a re-run the same day overwrites it. None records a failure."""
    table = tables.steam_price
    for appid, snapshot in snapshots.items():
        values = {
            "ok": snapshot is not None,
            "currency": snapshot.currency if snapshot else None,
            "initial_price": snapshot.initial if snapshot else None,
            "final_price": snapshot.final if snapshot else None,
            "discount_percent": snapshot.discount_percent if snapshot else None,
            "run_id": run_id,
        }
        where = (table.c.appid == appid) & (table.c.snapshot_date == snapshot_date)
        if not connection.execute(update(table).where(where).values(**values)).rowcount:
            connection.execute(
                insert(table).values(appid=appid, snapshot_date=snapshot_date, **values)
            )


def read_coverage(connection: Connection, appid: int) -> Coverage | None:
    table = tables.ingest_state
    row = connection.execute(
        select(table.c.covered_from, table.c.covered_to).where(table.c.appid == appid)
    ).first()
    if row is None:
        return None
    return Coverage(aware_utc(row.covered_from), aware_utc(row.covered_to))


def write_coverage(connection: Connection, appid: int, coverage: Coverage, *, run_id: int) -> None:
    table = tables.ingest_state
    values = {
        "covered_from": naive_utc(coverage.covered_from),
        "covered_to": naive_utc(coverage.covered_to),
        "updated_run_id": run_id,
    }
    if not connection.execute(
        update(table).where(table.c.appid == appid).values(**values)
    ).rowcount:
        connection.execute(insert(table).values(appid=appid, **values))


def record_dq(
    connection: Connection,
    *,
    run_id: int,
    check: str,
    appid: int | None,
    severity: str,
    detail: str,
) -> None:
    connection.execute(
        insert(tables.data_quality_result).values(
            run_id=run_id, check_name=check, appid=appid, severity=severity, detail=detail[:2000]
        )
    )


def purge_raw_pages(connection: Connection, *, older_than: datetime) -> int:
    """Delete raw payloads fetched before `older_than` (spec §6.1 keeps them 30 days)."""
    table = tables.steam_page
    result = connection.execute(delete(table).where(table.c.fetched_at < naive_utc(older_than)))
    return int(result.rowcount)
