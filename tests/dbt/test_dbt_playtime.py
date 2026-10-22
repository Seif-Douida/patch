"""mart.daily_playtime_sentiment: reviews per game, day and playtime bucket (Phase 4, design D8).

On the SQL Server container, built by the real dbt project.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr
from sqlalchemy import text

from patchpulse.config import Settings
from patchpulse.db.engine import make_engine
from patchpulse.db.migrate import upgrade_to_head
from patchpulse.db.provision import WRITER_USER, provision_writer
from patchpulse.ingest.games import Game, Role
from patchpulse.ingest.load import upsert_games, upsert_reviews
from patchpulse.ingest.models import ReviewRecord
from patchpulse.pipeline.dbt import run_dbt

pytestmark = pytest.mark.integration

GAME = 553850
TODAY = datetime.now(UTC).replace(hour=12, minute=0, second=0, microsecond=0)
DAY1, DAY2 = TODAY - timedelta(days=3), TODAY - timedelta(days=2)
WRITER_PASSWORD = "Writer-Playtime-1"


def review(rid: int, created: datetime, *, minutes: int | None, voted_up: bool) -> ReviewRecord:
    return ReviewRecord(
        recommendation_id=rid,
        appid=GAME,
        author_hash=f"{rid:064d}",
        author_num_games_owned=10,
        author_num_reviews=2,
        author_playtime_forever=minutes or 0,
        author_playtime_at_review=minutes,
        author_playtime_last_two_weeks=0,
        language="english",
        review_text="A review",
        timestamp_created=created,
        timestamp_updated=created,
        voted_up=voted_up,
        votes_up=0,
        votes_funny=0,
        weighted_vote_score=0.5,
        comment_count=0,
        steam_purchase=True,
        received_for_free=False,
        written_during_early_access=False,
    )


def test_buckets_sum_to_the_daily_reviews(
    new_database: Callable[[], Settings], tmp_path_factory: pytest.TempPathFactory
) -> None:
    admin_settings = new_database()
    admin = make_engine(admin_settings)
    upgrade_to_head(admin)
    game = Game(
        appid=GAME,
        name="Helldivers 2",
        genres=["Action"],
        role=Role.TRACKED,
        backfill_start=(TODAY - timedelta(days=30)).date(),
    )
    reviews = [
        review(1, DAY1, minutes=30, voted_up=False),  # under 2 h
        review(2, DAY1, minutes=30, voted_up=True),
        review(3, DAY1, minutes=700, voted_up=True),  # 10-50 h
        review(4, DAY1, minutes=13_000, voted_up=True),  # 200 h and more
        review(5, DAY2, minutes=None, voted_up=False),  # playtime unknown
        review(6, DAY2, minutes=200, voted_up=True),  # 2-10 h
    ]
    with admin.begin() as connection:
        provision_writer(connection, password=WRITER_PASSWORD)
        upsert_games(connection, [game], run_id=1)
        upsert_reviews(connection, reviews, run_id=1)
    writer = admin_settings.model_copy(
        update={"db_user": WRITER_USER, "db_password": SecretStr(WRITER_PASSWORD)}
    )

    result = run_dbt(["build"], settings=writer, target_path=tmp_path_factory.mktemp("dbt"))

    assert result.ok, result.failures
    with admin.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT date_key, playtime_bucket, n_reviews, n_positive "
                "FROM mart.daily_playtime_sentiment ORDER BY date_key, playtime_bucket"
            )
        ).all()
    day1, day2 = (int(d.strftime("%Y%m%d")) for d in (DAY1, DAY2))
    assert [tuple(r) for r in rows] == [
        (day1, "10h_50h", 1, 1),
        (day1, "200h_plus", 1, 1),
        (day1, "lt_2h", 2, 1),
        (day2, "2h_10h", 1, 1),
        (day2, "unknown", 1, 0),
    ]
    admin.dispose()
