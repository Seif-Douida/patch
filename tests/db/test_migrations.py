"""Alembic owns the raw and ops schemas; dbt's schemas exist before dbt first runs (spec §6.4)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import pytest
from alembic import command
from alembic.config import Config
from pydantic import SecretStr
from sqlalchemy import Engine, insert, text
from sqlalchemy.exc import DBAPIError

from patchpulse.config import Settings
from patchpulse.db import tables as schema
from patchpulse.db.engine import make_engine
from patchpulse.db.migrate import SCRIPT_LOCATION, downgrade_to_base, upgrade_to_head
from patchpulse.db.provision import WRITER_USER, provision_writer

pytestmark = pytest.mark.integration

EXPECTED_TABLES = {
    "raw.game",
    "raw.steam_page",
    "raw.steam_review",
    "raw.steam_news",
    "raw.news_classification",
    "raw.steam_price",
    "raw.review_aspect_score",
    "ops.pipeline_run",
    "ops.ingest_state",
    "ops.data_quality_result",
    "ops.qdrant_point",
    "ops.qdrant_reconciliation",
    "raw.steam_review_daily",
    "app.patch_effect",
}
EXPECTED_SCHEMAS = {"raw", "ops", "stg", "core", "mart", "app"}


def tables(engine: Engine) -> set[str]:
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT s.name + '.' + t.name FROM sys.tables t JOIN sys.schemas s "
                "ON s.schema_id = t.schema_id WHERE s.name IN ('raw', 'ops', 'app')"
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


def test_pp_writer_can_maintain_the_mirror(new_database: Callable[[], Settings]) -> None:
    # pp_writer was provisioned before migration 0003 existed: its grants on the ops schema must
    # cover the tables added since, which the nightly index stage writes (Phase 3, design D5).
    admin = new_database()
    engine = make_engine(admin)
    with engine.begin() as connection:
        config = Config()
        config.set_main_option("script_location", SCRIPT_LOCATION)
        config.attributes["connection"] = connection
        command.upgrade(config, "0002")
    with engine.begin() as connection:
        provision_writer(connection, password="Writer-One-1")
    upgrade_to_head(engine)

    with as_writer(admin, "Writer-One-1").begin() as connection:
        connection.execute(
            text(
                "INSERT INTO ops.qdrant_point "
                "(collection, review_id, [version], fingerprint, synced_run_id) "
                "VALUES ('reviews_v1', 1, '2026-10-20 08:00:00', REPLICATE('a', 64), 1)"
            )
        )
        connection.execute(
            text("UPDATE ops.qdrant_point SET fingerprint = REPLICATE('b', 64), synced_run_id = 2")
        )
        assert (
            connection.execute(
                text("SELECT synced_run_id FROM ops.qdrant_point WHERE review_id = 1")
            ).scalar_one()
            == 2
        )
        connection.execute(text("DELETE FROM ops.qdrant_point WHERE review_id = 1"))
        connection.execute(
            text(
                "INSERT INTO ops.qdrant_reconciliation "
                "(run_id, collection, appid, [day], sql_count, qdrant_count) "
                "VALUES (2, 'reviews_v1', 10, 20261020, 5, 4)"
            )
        )
        assert (
            connection.execute(
                text("SELECT qdrant_count FROM ops.qdrant_reconciliation")
            ).scalar_one()
            == 4
        )
        connection.execute(text("DELETE FROM ops.qdrant_reconciliation"))


def test_coverage_is_recomputed_from_the_reviews_stored(
    new_database: Callable[[], Settings],
) -> None:
    # 0004 (20 Oct 2026): Steam had ended three games' backfill windows early, so ops.ingest_state
    # claimed history back to backfill_start that raw.steam_review doesn't hold. Coverage restarts
    # at each game's oldest stored review; the fixed backfill fetches what's older.
    engine = make_engine(new_database())
    with engine.begin() as connection:
        config = Config()
        config.set_main_option("script_location", SCRIPT_LOCATION)
        config.attributes["connection"] = connection
        command.upgrade(config, "0003")
    floor, now = datetime(2024, 10, 1), datetime(2026, 10, 20)
    oldest = {10: datetime(2026, 9, 14, 7), 20: datetime(2024, 10, 1, 5)}  # 30 has none stored
    with engine.begin() as connection:
        for appid in (10, 20, 30):
            connection.execute(
                insert(schema.ingest_state).values(
                    appid=appid, covered_from=floor, covered_to=now, updated_run_id=1
                )
            )
        for rid, (appid, created) in enumerate(
            [(10, oldest[10]), (10, now), (20, oldest[20]), (20, now)], start=1
        ):
            row = stored_review(rid, appid, created)
            connection.execute(insert(schema.steam_review).values(row))

    upgrade_to_head(engine)

    with engine.connect() as connection:
        rows = connection.execute(text("SELECT appid, covered_from FROM ops.ingest_state")).all()
    covered = {int(row.appid): row.covered_from for row in rows}
    assert covered == {10: oldest[10], 20: floor, 30: floor}


def stored_review(rid: int, appid: int, created: datetime) -> dict[str, object]:
    return {
        "recommendation_id": rid,
        "timestamp_updated": created,
        "appid": appid,
        "author_hash": "a" * 64,
        "language": "english",
        "review_text": "fine",
        "timestamp_created": created,
        "voted_up": True,
        "votes_up": 0,
        "votes_funny": 0,
        "weighted_vote_score": 0.0,
        "comment_count": 0,
        "steam_purchase": True,
        "received_for_free": False,
        "written_during_early_access": False,
        "first_seen_run_id": 1,
        "last_seen_run_id": 1,
    }


def migrate_to(engine: Engine, revision: str, *, down: bool = False) -> None:
    with engine.begin() as connection:
        config = Config()
        config.set_main_option("script_location", SCRIPT_LOCATION)
        config.attributes["connection"] = connection
        (command.downgrade if down else command.upgrade)(config, revision)


def test_0005_creates_and_drops_its_tables(migrated: tuple[Settings, Engine]) -> None:
    # Phase 4: the comparison games' daily counts and the stored effect estimates.
    _, engine = migrated
    added = {"raw.steam_review_daily", "app.patch_effect"}
    assert added <= tables(engine)

    migrate_to(engine, "0004", down=True)

    assert not added & tables(engine)
    assert "app" not in schemas(engine)


def test_pp_writer_can_write_counts_and_effects(migrated: tuple[Settings, Engine]) -> None:
    # pp-migrate provisions the writer after every upgrade, so its grants reach the new schema.
    admin, engine = migrated
    with engine.begin() as connection:
        provision_writer(connection, password="Writer-One-1")

    with as_writer(admin, "Writer-One-1").begin() as connection:
        connection.execute(
            text(
                "INSERT INTO raw.steam_review_daily "
                "(appid, [day], positive, negative, fetched_run_id, fetched_at) "
                "VALUES (10, '2026-10-20', 7, 2, 1, '2026-10-21 03:00:00')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO app.patch_effect (bundle_id, patch_ids, appid, outcome, sample, "
                "horizon, method, run_id, effect, lo90, hi90, p_value, pre_fit_rmse, donors, "
                "flags, reason, fingerprint, version, computed_at) VALUES ('b1', '[\"p1\"]', 10, "
                "'positive_share', 'all', 14, 'its', 1, -2.5, -4.0, -1.0, NULL, NULL, '[]', '', "
                "NULL, REPLICATE('a', 64), 'e2', '2026-10-21 03:00:00')"
            )
        )
        stored: int = connection.execute(
            text("SELECT positive + negative FROM raw.steam_review_daily")
        ).scalar_one()
        effect: float = connection.execute(text("SELECT effect FROM app.patch_effect")).scalar_one()
        connection.execute(text("DELETE FROM app.patch_effect"))
        connection.execute(text("DELETE FROM raw.steam_review_daily"))

    assert stored == 9
    assert effect == pytest.approx(-2.5)
