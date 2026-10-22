"""mart.daily_comparison_counts: the comparison games' review counts per day (Phase 4, design D4).

On the SQL Server container, built by the real dbt project. dbt loads the real seed; the module's
build then replaces the seed table's rows for its own games and rebuilds the two models, so the
test reads known seed rows whatever the committed seed holds.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime

import pytest
from pydantic import SecretStr
from sqlalchemy import Engine, text

from patchpulse.config import Settings
from patchpulse.db.engine import make_engine
from patchpulse.db.migrate import upgrade_to_head
from patchpulse.db.provision import WRITER_USER, provision_writer
from patchpulse.ingest.day_counts import DayCounts
from patchpulse.ingest.games import Game, Role
from patchpulse.ingest.load import store_day_counts, upsert_games
from patchpulse.pipeline.dbt import run_dbt

pytestmark = pytest.mark.integration

DONOR, TRACKED = 9_000_001, 9_000_002  # not in the committed seed
DAY1, DAY2, DAY3 = date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 3)
WRITER_PASSWORD = "Writer-Comparison-1"
FETCHED = datetime(2026, 10, 22, 3, tzinfo=UTC)


def game(appid: int, role: Role) -> Game:
    return Game(
        appid=appid, name=f"Game {appid}", genres=["Action"], role=role, backfill_start=DAY1
    )


@pytest.fixture(scope="module")
def built(new_database: Callable[[], Settings], tmp_path_factory: pytest.TempPathFactory) -> Engine:
    admin_settings = new_database()
    admin = make_engine(admin_settings)
    upgrade_to_head(admin)
    with admin.begin() as connection:
        provision_writer(connection, password=WRITER_PASSWORD)
        upsert_games(connection, [game(DONOR, Role.DONOR), game(TRACKED, Role.BOTH)], run_id=1)
        for counts in (
            DayCounts(DONOR, DAY2, 9, 1),  # also in the seed below: this one wins
            DayCounts(DONOR, DAY3, 4, 0),
            DayCounts(TRACKED, DAY1, 3, 3),  # a tracked game's counts never reach the mart
        ):
            store_day_counts(connection, counts, run_id=1, fetched_at=FETCHED)
    writer = admin_settings.model_copy(
        update={"db_user": WRITER_USER, "db_password": SecretStr(WRITER_PASSWORD)}
    )
    first = run_dbt(["build"], settings=writer, target_path=tmp_path_factory.mktemp("dbt"))
    assert first.ok, first.failures
    with admin.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO core.donor_daily_counts (appid, [day], positive, negative) VALUES "
                f"({DONOR}, '2026-10-01', 5, 5), ({DONOR}, '2026-10-02', 1, 1), "
                f"({TRACKED}, '2026-10-02', 2, 2)"
            )
        )
    again = run_dbt(
        ["build", "--select", "stg_steam_review_daily", "daily_comparison_counts"],
        settings=writer,
        target_path=tmp_path_factory.mktemp("dbt"),
    )
    assert again.ok, again.failures
    return admin


def mart(engine: Engine) -> list[tuple[int, int, int, int, int]]:
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT game_key, date_key, n_positive, n_negative, n_reviews "
                f"FROM mart.daily_comparison_counts WHERE game_key IN ({DONOR}, {TRACKED}) "
                "ORDER BY game_key, date_key"
            )
        )
        return [tuple(row) for row in rows]


def test_raw_counts_win_over_the_seed(built: Engine) -> None:
    assert [row for row in mart(built) if row[0] == DONOR] == [
        (DONOR, 20261001, 5, 5, 10),  # the seed's
        (DONOR, 20261002, 9, 1, 10),  # in both: the stored row, fetched later
        (DONOR, 20261003, 4, 0, 4),  # the nightly stage's
    ]


def test_only_donors_in_the_mart(built: Engine) -> None:
    assert TRACKED not in {row[0] for row in mart(built)}
