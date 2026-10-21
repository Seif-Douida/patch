"""`core.review_index` (plan Task 2, design D1, D5) under `dbt build` on the SQL Server container.

It decides which reviews Qdrant holds: long enough (20 words, or 40 characters in Chinese,
Japanese and Thai), and a game's 40,000 most recent plus every review within 30 days of a
treatment patch. Each row carries the review's filter fields, its version and a fingerprint of the
fields, which the nightly sync compares with what Qdrant holds. The first module's tests run in
order: the first build, a re-score, then an edit, each followed by a build.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr
from sqlalchemy import Engine, insert, text

from patchpulse.config import Settings
from patchpulse.db import tables
from patchpulse.db.engine import make_engine
from patchpulse.db.migrate import upgrade_to_head
from patchpulse.db.provision import WRITER_USER, provision_writer
from patchpulse.ingest.games import Game, Role
from patchpulse.ingest.load import upsert_classifications, upsert_games, upsert_news, upsert_reviews
from patchpulse.ingest.models import NewsItem, ReviewRecord
from patchpulse.ingest.patches import Classification, PatchType
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.pipeline.dbt import DbtResult, run_dbt

pytestmark = pytest.mark.integration

TODAY = datetime.now(UTC).replace(hour=12, minute=0, second=0, microsecond=0)
GAME, OTHER_GAME = 553850, 1091500
WRITER_PASSWORD = "Writer-Index-1"
FIRST = ("s0-20261013T000000Z", 2)  # (model version, scoring run)
SECOND = ("s0-20261020T000000Z", 4)


def days_ago(n: float) -> datetime:
    return TODAY - timedelta(days=n)


def words(n: int) -> str:
    return " ".join(f"word{i}" for i in range(n))


def review(
    rid: int,
    created: datetime,
    body: str = words(25),
    *,
    language: str = "english",
    appid: int = GAME,
    edited: datetime | None = None,
) -> ReviewRecord:
    return ReviewRecord(
        recommendation_id=rid,
        appid=appid,
        author_hash=f"{rid:064d}",
        author_num_games_owned=10,
        author_num_reviews=2,
        author_playtime_forever=900,
        author_playtime_at_review=600,
        author_playtime_last_two_weeks=0,
        language=language,
        review_text=body,
        timestamp_created=created,
        timestamp_updated=edited or created,
        voted_up=False,
        votes_up=1,
        votes_funny=0,
        weighted_vote_score=0.5,
        comment_count=0,
        steam_purchase=True,
        received_for_free=False,
        written_during_early_access=False,
    )


def score(by: tuple[str, int], rid: int, updated: datetime, positives: set[str]) -> dict[str, Any]:
    return {
        "recommendation_id": rid,
        "timestamp_updated": updated.replace(tzinfo=None),
        "model_version": by[0],
        "model_name": "aspect-classifier",
        "scored_run_id": by[1],
        "scored_at": TODAY.replace(tzinfo=None),
        **{f"p_{code}": 0.9 if code in positives else 0.1 for code in ASPECTS},
        **{f"l_{code}": code in positives for code in ASPECTS},
    }


def patch(
    gid: str, appid: int, published: datetime, kind: PatchType
) -> tuple[NewsItem, Classification]:
    item = NewsItem(
        gid=gid,
        appid=appid,
        title=f"Update {gid}",
        url=f"https://store.steampowered.com/news/{gid}",
        contents="Patch notes",
        published_at=published,
        feedname="steam_community_announcements",
        feedlabel="Community Announcements",
    )
    return item, Classification(patch_type=kind, rule="version_number")


@dataclass
class Database:
    admin: Engine
    writer: Settings
    target: Path
    dbt_vars: dict[str, Any]

    def build(self) -> DbtResult:
        args = ["build"] + (["--vars", json.dumps(self.dbt_vars)] if self.dbt_vars else [])
        return run_dbt(args, settings=self.writer, target_path=self.target)

    def index(self) -> dict[int, Any]:
        with self.admin.connect() as connection:
            rows = connection.execute(text("SELECT * FROM core.review_index")).mappings().all()
        return {row["review_id"]: row for row in rows}

    def add_reviews(self, records: list[ReviewRecord], *, run_id: int) -> None:
        with self.admin.begin() as connection:
            upsert_reviews(connection, records, run_id=run_id)

    def add_scores(self, rows: list[dict[str, Any]]) -> None:
        with self.admin.begin() as connection:
            connection.execute(insert(tables.review_aspect_score), rows)


def make_database(
    new_database: Callable[[], Settings],
    target: Path,
    *,
    reviews: list[ReviewRecord],
    patches: Sequence[tuple[NewsItem, Classification]] = (),
    dbt_vars: dict[str, Any] | None = None,
    released: datetime | None = None,
    roles: Mapping[int, Role] | None = None,
) -> Database:
    admin_settings = new_database()
    admin = make_engine(admin_settings)
    upgrade_to_head(admin)
    games = [
        Game(
            appid=appid,
            name=f"Game {appid}",
            genres=["Action"],
            role=(roles or {}).get(appid, Role.TRACKED),
            backfill_start=days_ago(400).date(),
            release_date=released.date() if released and appid == GAME else None,
        )
        for appid in (GAME, OTHER_GAME)
    ]
    with admin.begin() as connection:
        provision_writer(connection, password=WRITER_PASSWORD)
        upsert_games(connection, games, run_id=1)
        upsert_reviews(connection, reviews, run_id=1)
        upsert_news(connection, [item for item, _ in patches], run_id=1)
        upsert_classifications(connection, {item.gid: c for item, c in patches}, run_id=1)
    writer = admin_settings.model_copy(
        update={"db_user": WRITER_USER, "db_password": SecretStr(WRITER_PASSWORD)}
    )
    return Database(admin, writer, target, dbt_vars or {})


# One database, built in steps: odd reviews and the aspect lifecycle.

ODD = {
    1: review(1, days_ago(1), "Too short"),
    2: review(2, days_ago(1), words(20)),  # exactly 20 words
    3: review(3, days_ago(1), words(19)),
    4: review(4, days_ago(1), "\U0001f600" * 30),  # emoji only: one "word"
    # BBCode counts as written: the tags stay attached to the words around them.
    5: review(5, days_ago(1), "[list]" + "".join(f"[*]item{i} " for i in range(20)) + "[/list]"),
    6: review(6, days_ago(1), "lag " * 2_000),  # 8,000 characters
    7: review(7, days_ago(1), "好" * 39, language="schinese"),
    8: review(8, days_ago(1), "好" * 40, language="schinese"),
    9: review(9, days_ago(1), "差" * 40, language="tchinese"),
    10: review(10, days_ago(1), "あ" * 40, language="japanese"),
    11: review(11, days_ago(1), "ก" * 45, language="thai"),
    12: review(12, days_ago(1), " ".join(["게임"] * 20), language="koreana"),
    13: review(13, days_ago(1), "게임" * 30, language="koreana"),  # Korean uses spaces
    14: review(14, days_ago(1), ""),
    15: review(15, days_ago(1), words(30), language="klingon"),  # a code Steam may add later
}
INDEXED = {2, 5, 6, 8, 9, 10, 11, 12, 15}
SCORED, NO_POSITIVE, UNSCORED = 101, 102, 103


@pytest.fixture(scope="module")
def lifecycle(
    new_database: Callable[[], Settings], tmp_path_factory: pytest.TempPathFactory
) -> Database:
    database = make_database(
        new_database,
        tmp_path_factory.mktemp("dbt-index"),
        reviews=[
            *ODD.values(),
            review(SCORED, days_ago(2)),
            review(NO_POSITIVE, days_ago(2)),
            review(UNSCORED, days_ago(2)),
        ],
    )
    database.add_scores(
        [
            score(FIRST, SCORED, days_ago(2), {"stability_bugs", "performance"}),
            score(FIRST, NO_POSITIVE, days_ago(2), set()),
        ]
    )
    result = database.build()
    assert result.ok, result.failures
    return database


def test_index_rules_handle_odd_reviews(lifecycle: Database) -> None:
    index = lifecycle.index()

    assert set(index) - {SCORED, NO_POSITIVE, UNSCORED} == INDEXED
    row = index[2]
    created = ODD[2].timestamp_created
    assert row["appid"] == GAME
    assert row["day"] == int(created.strftime("%Y%m%d"))
    assert row["created_ts"] == int(created.timestamp())
    assert (row["language"], row["voted_up"], row["playtime_bucket"]) == (
        "english",
        False,
        "10h_50h",
    )
    assert row["first_seen_run_id"] == 1
    assert row["version"] == created.replace(tzinfo=None)
    assert row["rules_version"] == "r1"
    assert len(row["fingerprint"]) == 64
    assert int(row["fingerprint"], 16) >= 0  # hex


def test_unscored_reviews_are_indexed_with_no_aspects(lifecycle: Database) -> None:
    index = lifecycle.index()

    assert (index[UNSCORED]["aspects"], index[UNSCORED]["aspect_model_version"]) == ("", None)
    # Scored with nothing positive: no aspects, but the model that said so.
    assert (index[NO_POSITIVE]["aspects"], index[NO_POSITIVE]["aspect_model_version"]) == (
        "",
        FIRST[0],
    )
    # In taxonomy order, whatever the order the labels were given in.
    assert index[SCORED]["aspects"] == "performance|stability_bugs"
    assert index[SCORED]["aspect_model_version"] == FIRST[0]
    assert len({index[r]["fingerprint"] for r in (SCORED, NO_POSITIVE, UNSCORED)}) == 3


def test_aspects_and_fingerprint_follow_the_latest_score(lifecycle: Database) -> None:
    before = lifecycle.index()
    lifecycle.add_scores([score(SECOND, SCORED, days_ago(2), {"content"})])

    result = lifecycle.build()
    after = lifecycle.index()

    assert result.ok, result.failures
    assert (after[SCORED]["aspects"], after[SCORED]["aspect_model_version"]) == (
        "content",
        SECOND[0],
    )
    assert after[SCORED]["fingerprint"] != before[SCORED]["fingerprint"]
    assert after[SCORED]["version"] == before[SCORED]["version"]
    for rid in set(after) - {SCORED}:
        assert after[rid]["fingerprint"] == before[rid]["fingerprint"]


def test_an_edit_changes_the_version_not_the_fingerprint(lifecycle: Database) -> None:
    before = lifecycle.index()
    edited = days_ago(0.5)
    lifecycle.add_reviews([review(NO_POSITIVE, days_ago(2), edited=edited)], run_id=5)
    lifecycle.add_scores([score(("s0-20261013T000000Z", 5), NO_POSITIVE, edited, set())])

    result = lifecycle.build()
    after = lifecycle.index()

    assert result.ok, result.failures
    assert after[NO_POSITIVE]["version"] == edited.replace(tzinfo=None)
    assert after[NO_POSITIVE]["fingerprint"] == before[NO_POSITIVE]["fingerprint"]


# Another database, with the cap at 3 reviews a game.


def test_cap_keeps_the_most_recent_every_patch_window_and_the_launch(
    new_database: Callable[[], Settings], tmp_path_factory: pytest.TempPathFactory
) -> None:
    capped = make_database(
        new_database,
        tmp_path_factory.mktemp("dbt-index-cap"),
        reviews=[
            review(201, days_ago(1)),
            review(202, days_ago(2)),
            review(203, days_ago(3)),  # the same moment as 204: the higher id is the more recent
            review(204, days_ago(3)),
            review(205, days_ago(4)),
            review(210, days_ago(80)),  # 20 days before the patch of day -60
            review(211, days_ago(95)),  # 35 days before it; another game's patch that day
            review(212, days_ago(30)),  # 30 days after it, the window's last day
            review(214, days_ago(150)),  # on a hotfix's day: not a treatment patch
            review(215, days_ago(0.5), "Too short"),  # newest, but not eligible: takes no slot
            review(216, days_ago(290)),  # 10 days after the game's launch
            review(217, days_ago(340)),  # 40 days before it
            review(301, days_ago(1), appid=OTHER_GAME),
            review(302, days_ago(200), appid=OTHER_GAME),
        ],
        patches=[
            patch("p1", GAME, days_ago(60), PatchType.PATCH),
            patch("p2", GAME, days_ago(150), PatchType.HOTFIX),
            patch("p3", OTHER_GAME, days_ago(95), PatchType.MAJOR_UPDATE),
        ],
        dbt_vars={"index_cap_per_game": 3},
        released=days_ago(300),
    )

    result = capped.build()

    # Equal daily counts also make a flat volume history, which the mart's volume check survives.
    assert result.ok, result.failures
    assert set(capped.index()) == {201, 202, 204, 210, 212, 216, 301, 302}


def test_only_tracked_games_are_indexed(
    new_database: Callable[[], Settings], tmp_path_factory: pytest.TempPathFactory
) -> None:
    # Donor games (Phase 4's synthetic control) feed the marts, not the free cluster: each would
    # take up to 40,000 points of its memory (the Phase 3 review).
    games = make_database(
        new_database,
        tmp_path_factory.mktemp("dbt-index-roles"),
        reviews=[review(401, days_ago(1)), review(402, days_ago(1), appid=OTHER_GAME)],
        roles={OTHER_GAME: Role.DONOR},
    )

    result = games.build()

    assert result.ok, result.failures
    assert set(games.index()) == {401}
