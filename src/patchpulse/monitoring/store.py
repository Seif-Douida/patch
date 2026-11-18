"""Phase 7's two tables (migration 0007): the teacher's labels and every measured number.

`raw.teacher_label` keys a label by review version, model and prompt; it never holds text.
`ops.eval_metric` holds one row per metric and scope: `week` (the drift and agreement checks),
`night`, `report:<id>` (the online evals), `drill` (the acceptance drill's numbers, never mixed
with the real weeks).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime

from sqlalchemy import Connection, and_, func, insert, select

from patchpulse.db.tables import eval_metric, teacher_label
from patchpulse.labeling.taxonomy import ASPECTS

_ID_BATCH = 1_000  # SQL Server allows about 2,100 parameters a statement


@dataclass(frozen=True)
class MetricRow:
    family: str  # drift, agreement, online, retrain
    metric: str
    scope: str  # week, night, report:<id>, drill
    window_start: date | None
    window_end: date | None
    value: float | None
    lo: float | None
    hi: float | None
    n: int | None


@dataclass(frozen=True)
class StoredTeacherLabel:
    recommendation_id: int
    timestamp_updated: datetime  # naive UTC, as raw.steam_review stores it
    model: str
    prompt_hash: str
    source: str  # silver or spotcheck
    aspects: frozenset[str]


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0)


def write_metrics(connection: Connection, rows: Sequence[MetricRow], *, run_id: int | None) -> int:
    if not rows:
        return 0
    created = _now()
    connection.execute(
        insert(eval_metric),
        [
            {
                "run_id": run_id,
                "family": row.family,
                "metric": row.metric,
                "scope": row.scope,
                "window_start": row.window_start,
                "window_end": row.window_end,
                "value": row.value,
                "lo": row.lo,
                "hi": row.hi,
                "n": row.n,
                "created_at": created,
            }
            for row in rows
        ],
    )
    return len(rows)


def read_metrics(
    connection: Connection, *, family: str, scope: str | None = None, since: date | None = None
) -> list[MetricRow]:
    """Oldest first; `since` keeps rows whose window ends on or after it."""
    conditions = [eval_metric.c.family == family]
    if scope is not None:
        conditions.append(eval_metric.c.scope == scope)
    if since is not None:
        conditions.append(eval_metric.c.window_end >= since)
    statement = (
        select(
            eval_metric.c.family,
            eval_metric.c.metric,
            eval_metric.c.scope,
            eval_metric.c.window_start,
            eval_metric.c.window_end,
            eval_metric.c.value,
            eval_metric.c.lo,
            eval_metric.c.hi,
            eval_metric.c.n,
        )
        .where(and_(*conditions))
        .order_by(eval_metric.c.id)
    )
    return [MetricRow(*row) for row in connection.execute(statement)]


def last_window_end(connection: Connection, *, family: str, scope: str = "week") -> date | None:
    statement = select(func.max(eval_metric.c.window_end)).where(
        eval_metric.c.family == family, eval_metric.c.scope == scope
    )
    found: date | None = connection.execute(statement).scalar_one()
    return found


def _key(label: StoredTeacherLabel) -> tuple[int, datetime, str, str]:
    return (label.recommendation_id, label.timestamp_updated, label.model, label.prompt_hash)


def write_teacher_labels(
    connection: Connection, labels: Sequence[StoredTeacherLabel], *, run_id: int | None
) -> int:
    """Insert the labels not stored yet (same version, model and prompt); returns how many."""
    columns = (
        teacher_label.c.recommendation_id,
        teacher_label.c.timestamp_updated,
        teacher_label.c.model,
        teacher_label.c.prompt_hash,
    )
    unique = list({_key(label): label for label in labels}.values())
    ids = sorted({label.recommendation_id for label in unique})
    stored: set[tuple[int, datetime, str, str]] = set()
    # SQL Server has no row-value IN: select by id, match the rest of the key here.
    for start in range(0, len(ids), _ID_BATCH):
        chunk = ids[start : start + _ID_BATCH]
        found = connection.execute(select(*columns).where(columns[0].in_(chunk)))
        stored |= {(int(r[0]), r[1], str(r[2]), str(r[3])) for r in found}
    new = [label for label in unique if _key(label) not in stored]
    if not new:
        return 0
    created = _now()
    connection.execute(
        insert(teacher_label),
        [
            {
                "recommendation_id": label.recommendation_id,
                "timestamp_updated": label.timestamp_updated,
                "model": label.model,
                "prompt_hash": label.prompt_hash,
                "source": label.source,
                "run_id": run_id,
                "created_at": created,
                **{f"l_{code}": code in label.aspects for code in ASPECTS},
            }
            for label in new
        ],
    )
    return len(new)


def read_teacher_labels(
    connection: Connection, *, source: str | None = None, since: datetime | None = None
) -> list[StoredTeacherLabel]:
    """`since` keeps labels created at or after it (naive UTC)."""
    conditions = []
    if source is not None:
        conditions.append(teacher_label.c.source == source)
    if since is not None:
        conditions.append(teacher_label.c.created_at >= since)
    statement = (
        select(teacher_label).where(and_(*conditions)) if conditions else select(teacher_label)
    )
    return [
        StoredTeacherLabel(
            recommendation_id=int(row.recommendation_id),
            timestamp_updated=row.timestamp_updated,
            model=str(row.model),
            prompt_hash=str(row.prompt_hash),
            source=str(row.source),
            aspects=frozenset(code for code in ASPECTS if getattr(row, f"l_{code}")),
        )
        for row in connection.execute(statement)
    ]
