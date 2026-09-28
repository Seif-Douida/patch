"""`dbt build` end to end on the SQL Server container, running as the least-privilege writer.

Two small games are loaded through the real loaders, then dbt builds stg / core / mart and runs
every data test. A second load then edits a review and dbt runs again, incrementally.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr
from sqlalchemy import Engine, text

from patchpulse.config import Settings
from patchpulse.db.engine import make_engine
from patchpulse.db.migrate import upgrade_to_head
from patchpulse.db.provision import WRITER_USER, provision_writer
from patchpulse.export.site import build_site_export
from patchpulse.ingest.games import Game, Role
from patchpulse.ingest.load import (
    upsert_classifications,
    upsert_games,
    upsert_news,
    upsert_prices,
    upsert_reviews,
)
from patchpulse.ingest.models import NewsItem, PriceSnapshot, ReviewRecord
from patchpulse.ingest.patches import classify
from patchpulse.pipeline.dbt import DbtResult, run_dbt

pytestmark = pytest.mark.integration

TODAY = datetime.now(UTC).replace(hour=12, minute=0, second=0, microsecond=0)
DAY1, DAY2, DAY3 = (TODAY - timedelta(days=d) for d in (6, 5, 4))
GAME_A, GAME_B = 553850, 949230
WRITER_PASSWORD = "Writer-Build-1"


def date_key(moment: datetime) -> int:
    return int(moment.strftime("%Y%m%d"))


def review(
    rid: int,
    appid: int,
    created: datetime,
    *,
    voted_up: bool = True,
    edited: datetime | None = None,
) -> ReviewRecord:
    return ReviewRecord(
        recommendation_id=rid,
        appid=appid,
        author_hash=f"{rid:064d}",
        author_num_games_owned=10,
        author_num_reviews=2,
        author_playtime_forever=900,
        author_playtime_at_review=60 if rid % 2 else 600,
        author_playtime_last_two_weeks=0,
        language="english" if rid != 4 else "french",
        review_text="A fine game with good patches",
        timestamp_created=created,
        timestamp_updated=edited or created,
        voted_up=voted_up,
        votes_up=1,
        votes_funny=0,
        weighted_vote_score=0.5,
        comment_count=0,
        steam_purchase=True,
        received_for_free=False,
        written_during_early_access=False,
    )


def news(gid: str, title: str, feedname: str, tags: list[str]) -> NewsItem:
    return NewsItem(
        gid=gid,
        appid=GAME_A,
        title=title,
        url="https://example.com",
        contents="",
        published_at=DAY2,
        feedname=feedname,
        feedlabel="",
        tags=tags,
    )


@dataclass
class Built:
    admin: Engine
    writer: Settings
    target: Path
    first: DbtResult

    def rows(self, sql: str) -> list[Any]:
        with self.admin.connect() as connection:
            return list(connection.execute(text(sql)))


@pytest.fixture(scope="module")
def built(new_database: Callable[[], Settings], tmp_path_factory: pytest.TempPathFactory) -> Built:
    admin_settings = new_database()
    admin = make_engine(admin_settings)
    upgrade_to_head(admin)
    games = [
        Game(appid=GAME_A, name="Helldivers 2", genres=["Action"], role=Role.TRACKED,
             backfill_start=(TODAY - timedelta(days=30)).date()),
        Game(appid=GAME_B, name="Cities: Skylines II", genres=["Simulation"], role=Role.TRACKED,
             backfill_start=(TODAY - timedelta(days=30)).date()),
    ]  # fmt: skip
    items = [
        news("100", "Devoid of Liberty: 7.1.1", "steam_community_announcements", ["patchnotes"]),
        news("101", "Helldivers 2 gets a hotfix, says press", "PC Gamer", []),
    ]
    with admin.begin() as connection:
        provision_writer(connection, password=WRITER_PASSWORD)
        upsert_games(connection, games, run_id=1)
        upsert_reviews(
            connection,
            [
                review(1, GAME_A, DAY1),
                review(2, GAME_A, DAY1),
                review(3, GAME_A, DAY1, voted_up=False),
                review(4, GAME_A, DAY3),
                review(5, GAME_B, DAY2),
            ],
            run_id=1,
        )
        upsert_news(connection, items, run_id=1)
        upsert_classifications(connection, {i.gid: classify(i) for i in items}, run_id=1)
        upsert_prices(
            connection,
            {GAME_A: PriceSnapshot(currency="USD", initial=3999, final=1999, discount_percent=50)},
            snapshot_date=TODAY.date(),
            run_id=1,
        )
    writer = admin_settings.model_copy(
        update={"db_user": WRITER_USER, "db_password": SecretStr(WRITER_PASSWORD)}
    )
    target = tmp_path_factory.mktemp("dbt-target")
    return Built(admin, writer, target, run_dbt(["build"], settings=writer, target_path=target))


def test_dbt_build_passes_every_model_and_test_as_the_writer(built: Built) -> None:
    assert built.first.ok, built.first.failures
    assert built.first.counts.get("error", 0) == 0
    assert built.first.counts.get("fail", 0) == 0


def test_daily_sentiment_counts_and_bounds(built: Built) -> None:
    (day,) = built.rows(
        "SELECT n_reviews, n_positive, pos_share, pos_lo90, pos_hi90, share_new_players "
        f"FROM mart.daily_game_sentiment WHERE game_key = {GAME_A} AND date_key = {date_key(DAY1)}"
    )
    assert (day.n_reviews, day.n_positive) == (3, 2)
    assert day.pos_share == pytest.approx(2 / 3)
    assert day.pos_lo90 < day.pos_share < day.pos_hi90
    assert day.share_new_players == pytest.approx(2 / 3)  # reviews 1 and 3 had 60 minutes


def test_day_without_reviews_has_null_share_not_an_error(built: Built) -> None:
    (day,) = built.rows(
        "SELECT n_reviews, pos_share, pos_lo90 FROM mart.daily_game_sentiment "
        f"WHERE game_key = {GAME_A} AND date_key = {date_key(DAY2)}"
    )
    assert tuple(day) == (0, None, None)


def test_sale_and_discount_columns_come_from_the_calendar_and_prices(built: Built) -> None:
    (today,) = built.rows(
        "SELECT discount_pct FROM mart.daily_game_sentiment "
        f"WHERE game_key = {GAME_A} AND date_key = {date_key(TODAY)}"
    )
    assert today.discount_pct == 50


def test_press_items_are_not_patches(built: Built) -> None:
    patches = built.rows("SELECT patch_id, patch_type, is_treatment_candidate FROM core.fact_patch")
    assert [tuple(p) for p in patches] == [("100", "patch", True)]


def test_patch_window_describes_treatment_candidates(built: Built) -> None:
    (window,) = built.rows("SELECT patch_id, n_pre, n_post FROM mart.patch_window")
    assert (window.patch_id, window.n_pre, window.n_post) == ("100", 3, 1)


def test_site_export_reads_the_marts(built: Built) -> None:
    with built.admin.connect() as connection:
        files = build_site_export(connection, data_version=1, generated_at=TODAY)

    index = json.loads(files["index.json"])
    assert [g["appid"] for g in index["games"]] == [GAME_B, GAME_A]  # sorted by name
    game = json.loads(files[f"games/{GAME_A}.json"])
    first = next(p for p in game["series"] if p["date"] == DAY1.date().isoformat())
    assert (first["n"], first["pos"]) == (3, 2)
    assert [(p["title"], p["type"], p["treatment"]) for p in game["patches"]] == [
        ("Devoid of Liberty: 7.1.1", "patch", True)
    ]


def test_reloading_an_edited_review_updates_not_duplicates(built: Built) -> None:
    # Runs last in this module: it changes the data the other tests read.
    with built.admin.begin() as connection:
        upsert_reviews(connection, [review(3, GAME_A, DAY1, voted_up=True, edited=DAY3)], run_id=2)
    again = run_dbt(["build"], settings=built.writer, target_path=built.target)

    assert again.ok, again.failures
    stored = built.rows("SELECT voted_up FROM core.fact_review WHERE review_id = 3")
    assert [r.voted_up for r in stored] == [True]
    (day,) = built.rows(
        "SELECT n_positive FROM mart.daily_game_sentiment "
        f"WHERE game_key = {GAME_A} AND date_key = {date_key(DAY1)}"
    )
    assert day.n_positive == 3
