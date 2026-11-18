"""Phase 7 Task 1: migration 0007's teacher labels and eval metrics, and their store."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text

from patchpulse.db.migrate import SCRIPT_LOCATION
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.monitoring.store import (
    MetricRow,
    StoredTeacherLabel,
    last_window_end,
    read_metrics,
    read_teacher_labels,
    write_metrics,
    write_teacher_labels,
)

pytestmark = pytest.mark.integration


def migrate_to(engine: Engine, revision: str, *, down: bool = False) -> None:
    with engine.begin() as connection:
        config = Config()
        config.set_main_option("script_location", SCRIPT_LOCATION)
        config.attributes["connection"] = connection
        (command.downgrade if down else command.upgrade)(config, revision)


def table_names(engine: Engine) -> set[str]:
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT s.name + '.' + t.name FROM sys.tables t JOIN sys.schemas s "
                "ON s.schema_id = t.schema_id"
            )
        )
        return {row[0] for row in rows}


def test_migration_creates_both_tables_and_downgrades(migrated_engine: Engine) -> None:
    added = {"raw.teacher_label", "ops.eval_metric"}
    assert added <= table_names(migrated_engine)

    migrate_to(migrated_engine, "0006", down=True)
    assert not added & table_names(migrated_engine)

    migrate_to(migrated_engine, "head")
    assert added <= table_names(migrated_engine)


def week(
    value: float | None, *, metric: str = "language_psi", end: date = date(2026, 11, 22)
) -> MetricRow:
    return MetricRow(
        family="drift",
        metric=metric,
        scope="week",
        window_start=date(2026, 11, 16),
        window_end=end,
        value=value,
        lo=None,
        hi=None,
        n=1200,
    )


def test_metrics_round_trip_with_nulls(migrated_engine: Engine) -> None:
    rows = [
        week(0.12),
        MetricRow(
            "online", "pre_verifier_pass_rate", "report:r1", None, None, None, None, None, None
        ),
    ]
    with migrated_engine.begin() as connection:
        assert write_metrics(connection, rows, run_id=7) == 2

    with migrated_engine.connect() as connection:
        drift = read_metrics(connection, family="drift")
        online = read_metrics(connection, family="online", scope="report:r1")

    assert drift == [rows[0]]
    assert online == [rows[1]]


def test_last_window_end_is_the_newest_week_of_a_family(migrated_engine: Engine) -> None:
    with migrated_engine.begin() as connection:
        assert last_window_end(connection, family="drift") is None
        write_metrics(
            connection,
            [week(0.1, end=date(2026, 11, 15)), week(0.2, end=date(2026, 11, 22))],
            run_id=1,
        )
        drill = MetricRow(
            "drift", "language_psi", "drill", None, date(2026, 12, 1), 0.9, None, None, 5
        )
        write_metrics(connection, [drill], run_id=2)

    with migrated_engine.connect() as connection:
        assert last_window_end(connection, family="drift") == date(2026, 11, 22)
        assert last_window_end(connection, family="drift", scope="drill") == date(2026, 12, 1)
        assert read_metrics(connection, family="drift", scope="week", since=date(2026, 11, 20)) == [
            week(0.2, end=date(2026, 11, 22))
        ]


def label(rid: int, aspects: frozenset[str], source: str = "spotcheck") -> StoredTeacherLabel:
    return StoredTeacherLabel(
        recommendation_id=rid,
        timestamp_updated=datetime(2026, 11, 17, 3, 0, 0),
        model="gemma-4-31b-it",
        prompt_hash="abc123",
        source=source,
        aspects=aspects,
    )


def test_teacher_labels_round_trip_and_skip_duplicates(migrated_engine: Engine) -> None:
    first = [label(1, frozenset({"performance"})), label(2, frozenset())]
    with migrated_engine.begin() as connection:
        assert write_teacher_labels(connection, first, run_id=5) == 2
    with migrated_engine.begin() as connection:
        again = [label(2, frozenset()), label(3, frozenset({"content", "off_topic"}), "silver")]
        assert write_teacher_labels(connection, again, run_id=6) == 1

    with migrated_engine.connect() as connection:
        every = read_teacher_labels(connection)
        spot = read_teacher_labels(connection, source="spotcheck")

    assert sorted(every, key=lambda x: x.recommendation_id) == [*first, again[1]]
    assert {x.recommendation_id for x in spot} == {1, 2}


def test_the_label_table_holds_no_text(migrated_engine: Engine) -> None:
    columns = {
        c["name"] for c in inspect(migrated_engine).get_columns("teacher_label", schema="raw")
    }
    expected = {
        "recommendation_id",
        "timestamp_updated",
        "model",
        "prompt_hash",
        "source",
        "run_id",
        "created_at",
        *(f"l_{code}" for code in ASPECTS),
    }
    assert columns == expected
