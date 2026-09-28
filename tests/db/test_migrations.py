"""Alembic owns the raw and ops schemas; dbt's schemas exist before dbt first runs (spec §6.4)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import SecretStr
from sqlalchemy import Engine, text
from sqlalchemy.exc import DBAPIError

from patchpulse.config import Settings
from patchpulse.db.engine import make_engine
from patchpulse.db.migrate import downgrade_to_base, upgrade_to_head
from patchpulse.db.provision import WRITER_USER, provision_writer

pytestmark = pytest.mark.integration

EXPECTED_TABLES = {
    "raw.game",
    "raw.steam_page",
    "raw.steam_review",
    "raw.steam_news",
    "raw.news_classification",
    "raw.steam_price",
    "ops.pipeline_run",
    "ops.ingest_state",
    "ops.data_quality_result",
}
EXPECTED_SCHEMAS = {"raw", "ops", "stg", "core", "mart"}


def tables(engine: Engine) -> set[str]:
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT s.name + '.' + t.name FROM sys.tables t "
                "JOIN sys.schemas s ON s.schema_id = t.schema_id WHERE s.name IN ('raw', 'ops')"
            )
        )
        return {row[0] for row in rows}


def schemas(engine: Engine) -> set[str]:
    with engine.connect() as connection:
        rows = connection.execute(text("SELECT name FROM sys.schemas"))
        return {row[0] for row in rows} & EXPECTED_SCHEMAS


def as_writer(admin: Settings, password: str) -> Engine:
    # model_copy skips validation, so the secret must already be a SecretStr.
    return make_engine(
        admin.model_copy(update={"db_user": WRITER_USER, "db_password": SecretStr(password)})
    )


@pytest.fixture
def migrated(new_database: Callable[[], Settings]) -> tuple[Settings, Engine]:
    admin = new_database()
    engine = make_engine(admin)
    upgrade_to_head(engine)
    return admin, engine


def test_upgrade_creates_every_schema_and_table(migrated: tuple[Settings, Engine]) -> None:
    _, engine = migrated
    assert schemas(engine) == EXPECTED_SCHEMAS
    assert tables(engine) == EXPECTED_TABLES


def test_only_surrogate_keys_are_identity_columns(migrated: tuple[Settings, Engine]) -> None:
    # SQLAlchemy makes a lone integer primary key an IDENTITY column unless told not to; Steam's
    # app IDs must be inserted as given.
    _, engine = migrated
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT s.name + '.' + t.name + '.' + c.name FROM sys.identity_columns c "
                "JOIN sys.tables t ON t.object_id = c.object_id "
                "JOIN sys.schemas s ON s.schema_id = t.schema_id"
            )
        )
        identities = {row[0] for row in rows}

    assert identities == {
        "ops.pipeline_run.run_id",
        "ops.data_quality_result.id",
        "raw.steam_page.page_id",
    }


def test_downgrade_then_upgrade_round_trips(migrated: tuple[Settings, Engine]) -> None:
    _, engine = migrated

    downgrade_to_base(engine)
    assert tables(engine) == set()

    upgrade_to_head(engine)
    assert tables(engine) == EXPECTED_TABLES


def test_provision_writer_is_idempotent_and_resyncs_the_password(
    migrated: tuple[Settings, Engine],
) -> None:
    admin, engine = migrated
    with engine.begin() as connection:
        provision_writer(connection, password="Writer-One-1")
    with engine.begin() as connection:
        provision_writer(connection, password="Writer-Two-2")

    with as_writer(admin, "Writer-Two-2").connect() as connection:
        assert connection.execute(text("SELECT USER_NAME()")).scalar_one() == WRITER_USER
    with pytest.raises(DBAPIError, match="Login failed"):
        as_writer(admin, "Writer-One-1").connect()


def test_writer_can_load_raw_and_build_dbt_schemas(migrated: tuple[Settings, Engine]) -> None:
    admin, engine = migrated
    with engine.begin() as connection:
        provision_writer(connection, password="Writer-One-1")

    with as_writer(admin, "Writer-One-1").begin() as connection:
        connection.execute(
            text(
                "INSERT INTO raw.game (appid, name, genres, role, backfill_start, updated_run_id) "
                "VALUES (1, 'Test', 'RPG', 'tracked', '2024-10-01', 1)"
            )
        )
        connection.execute(text("DELETE FROM raw.steam_page WHERE 1 = 0"))
        connection.execute(text("CREATE TABLE core.dbt_probe (id INT)"))
        connection.execute(text("CREATE VIEW mart.dbt_probe_view AS SELECT id FROM core.dbt_probe"))
        connection.execute(text("DROP VIEW mart.dbt_probe_view"))
        connection.execute(text("DROP TABLE core.dbt_probe"))


@pytest.mark.parametrize(
    "statement",
    [
        "ALTER TABLE raw.game ADD sneaky INT",
        "DROP TABLE ops.pipeline_run",
        "CREATE TABLE raw.sneaky (id INT)",
        "CREATE SCHEMA sneaky",
    ],
)
def test_writer_cannot_change_what_it_does_not_own(
    migrated: tuple[Settings, Engine], statement: str
) -> None:
    admin, engine = migrated
    with engine.begin() as connection:
        provision_writer(connection, password="Writer-One-1")

    with (
        pytest.raises(DBAPIError, match=r"permission|does not exist|Cannot find"),
        as_writer(admin, "Writer-One-1").begin() as connection,
    ):
        connection.execute(text(statement))
