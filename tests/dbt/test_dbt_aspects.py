"""The aspect models (plan Task 17, design D8) under `dbt build` on the SQL Server container.

`core.bridge_review_aspect` holds the positive labels of the latest score of each review's latest
version; `mart.daily_aspect_share` counts them per game, day and aspect, beside the number of
scored negative and positive reviews, so partial scoring is visible. The site export reads the
mart (plan Task 18). The tests run in order: the first build, then an edit, then a re-score, each
followed by an incremental build.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
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
from patchpulse.export.site import AspectModel, build_site_export
from patchpulse.ingest.games import Game, Role
from patchpulse.ingest.load import upsert_games, upsert_reviews
from patchpulse.ingest.models import ReviewRecord
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.pipeline.dbt import DbtResult, run_dbt

pytestmark = pytest.mark.integration

TODAY = datetime.now(UTC).replace(hour=12, minute=0, second=0, microsecond=0)
DAY1, DAY2, DAY3 = (TODAY - timedelta(days=d) for d in (6, 5, 4))
GAME = 553850
WRITER_PASSWORD = "Writer-Aspect-1"


@dataclass(frozen=True)
class Scoring:
    model_version: str
    run_id: int
    scored_at: datetime


FIRST = Scoring("s0-20261013T000000Z", run_id=2, scored_at=DAY3)
SECOND = Scoring("s0-20261020T000000Z", run_id=4, scored_at=DAY3 + timedelta(days=1))


def date_key(moment: datetime) -> int:
    return int(moment.strftime("%Y%m%d"))


def review(
    rid: int, created: datetime, *, voted_up: bool, edited: datetime | None = None
) -> ReviewRecord:
    return ReviewRecord(
        recommendation_id=rid,
        appid=GAME,
        author_hash=f"{rid:064d}",
        author_num_games_owned=10,
        author_num_reviews=2,
        author_playtime_forever=900,
        author_playtime_at_review=600,
        author_playtime_last_two_weeks=0,
        language="english",
        review_text="Runs badly since the patch",
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


def score(
    by: Scoring, rid: int, updated: datetime, positives: Mapping[str, float]
) -> dict[str, Any]:
    """A raw score row: the given aspects positive with their probability, the rest at 0.1."""
    return {
        "recommendation_id": rid,
        "timestamp_updated": updated.replace(tzinfo=None),
        "model_version": by.model_version,
        "model_name": "aspect-classifier",
        "scored_run_id": by.run_id,
        "scored_at": by.scored_at.replace(tzinfo=None),
        **{f"p_{code}": positives.get(code, 0.1) for code in ASPECTS},
        **{f"l_{code}": code in positives for code in ASPECTS},
    }


def add_scores(engine: Engine, rows: list[dict[str, Any]]) -> None:
    with engine.begin() as connection:
        connection.execute(insert(tables.review_aspect_score), rows)


@dataclass
class Built:
    admin: Engine
    writer: Settings
    target: Path
    first: DbtResult

    def rows(self, sql: str) -> list[Any]:
        with self.admin.connect() as connection:
            return list(connection.execute(text(sql)))

    def build(self) -> DbtResult:
        return run_dbt(["build"], settings=self.writer, target_path=self.target)

    def bridge(self) -> list[tuple[int, str, str]]:
        rows = self.rows(
            "SELECT review_id, aspect_key, model_version FROM core.bridge_review_aspect"
        )
        return sorted((r.review_id, r.aspect_key, r.model_version) for r in rows)

    def share(self, day: datetime, aspect: str) -> Any:
        (row,) = self.rows(
            "SELECT n_neg_mentions, n_pos_mentions, n_neg_scored, n_pos_scored, "
            "share_of_negative, share_of_positive FROM mart.daily_aspect_share "
            f"WHERE game_key = {GAME} AND date_key = {date_key(day)} AND aspect_key = '{aspect}'"
        )
        return row


@pytest.fixture(scope="module")
def built(new_database: Callable[[], Settings], tmp_path_factory: pytest.TempPathFactory) -> Built:
    """Day 1: reviews 1-2 positive, 3-4 negative, all scored (4 mentions nothing). Day 2: review 5,
    positive and not scored yet."""
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
    with admin.begin() as connection:
        provision_writer(connection, password=WRITER_PASSWORD)
        upsert_games(connection, [game], run_id=1)
        upsert_reviews(
            connection,
            [
                review(1, DAY1, voted_up=True),
                review(2, DAY1, voted_up=True),
                review(3, DAY1, voted_up=False),
                review(4, DAY1, voted_up=False),
                review(5, DAY2, voted_up=True),
            ],
            run_id=1,
        )
    add_scores(
        admin,
        [
            score(FIRST, 1, DAY1, {"performance": 0.9}),
            score(FIRST, 2, DAY1, {"performance": 0.8, "content": 0.7}),
            score(FIRST, 3, DAY1, {"performance": 0.95, "stability_bugs": 0.6}),
            score(FIRST, 4, DAY1, {}),
        ],
    )
    writer = admin_settings.model_copy(
        update={"db_user": WRITER_USER, "db_password": SecretStr(WRITER_PASSWORD)}
    )
    target = tmp_path_factory.mktemp("dbt-target")
    return Built(admin, writer, target, run_dbt(["build"], settings=writer, target_path=target))


def test_aspect_shares_are_computed_from_scored_reviews(built: Built) -> None:
    assert built.first.ok, built.first.failures
    assert built.first.counts.get("fail", 0) == 0
    assert built.bridge() == [
        (1, "performance", FIRST.model_version),
        (2, "content", FIRST.model_version),
        (2, "performance", FIRST.model_version),
        (3, "performance", FIRST.model_version),
        (3, "stability_bugs", FIRST.model_version),
    ]
    performance = built.share(DAY1, "performance")
    assert tuple(performance) == (1, 2, 2, 2, 0.5, 1.0)
    assert tuple(built.share(DAY1, "content")) == (0, 1, 2, 2, 0.0, 0.5)
    assert tuple(built.share(DAY1, "monetization")) == (0, 0, 2, 2, 0.0, 0.0)
    (n,) = built.rows(
        f"SELECT COUNT(*) AS n FROM mart.daily_aspect_share WHERE date_key = {date_key(DAY1)}"
    )
    assert n.n == len(ASPECTS)


def test_site_export_reads_the_aspect_mart(built: Built) -> None:
    model = AspectModel("s0", FIRST.model_version, 0.61)
    with built.admin.connect() as connection:
        files = build_site_export(
            connection, data_version=2, generated_at=TODAY, aspect_model=model
        )

    aspects = json.loads(files[f"games/{GAME}.json"])["aspects"]
    (day1,) = aspects["days"]  # day 2's review isn't scored: no aspect day
    assert day1["date"] == DAY1.date().isoformat()
    assert day1["scored"] == [2, 2]
    neg, pos = (
        dict(zip(ASPECTS, day1["neg"], strict=True)),
        dict(zip(ASPECTS, day1["pos"], strict=True)),
    )
    assert (neg["performance"], neg["stability_bugs"], pos["performance"], pos["content"]) == (
        1,
        1,
        2,
        1,
    )
    (game,) = json.loads(files["index.json"])["games"]
    assert game["top_complaint_30d"] == {"code": "performance", "share": 0.5, "n": 2}


def test_partial_scoring_shows_in_the_scored_counts(built: Built) -> None:
    # Day 2's review isn't scored: the mart says so instead of reporting a share of nothing.
    assert tuple(built.share(DAY2, "performance")) == (0, 0, 0, 0, None, None)
    # Review 1 is edited; until its new version is scored, it neither mentions nor counts.
    with built.admin.begin() as connection:
        upsert_reviews(connection, [review(1, DAY1, voted_up=True, edited=DAY3)], run_id=3)

    again = built.build()

    assert again.ok, again.failures
    assert [row for row in built.bridge() if row[0] == 1] == []
    assert tuple(built.share(DAY1, "performance")) == (1, 1, 2, 1, 0.5, 1.0)


def test_rescored_review_replaces_its_old_labels(built: Built) -> None:
    # Runs last: a new champion re-scores reviews 2 and 3 and scores review 1's new version.
    add_scores(
        built.admin,
        [
            score(SECOND, 1, DAY3, {"ux_controls": 0.7}),
            score(SECOND, 2, DAY1, {}),
            score(SECOND, 3, DAY1, {"online_servers": 0.9}),
        ],
    )

    again = built.build()

    assert again.ok, again.failures
    assert built.bridge() == [
        (1, "ux_controls", SECOND.model_version),
        (3, "online_servers", SECOND.model_version),
    ]
    assert tuple(built.share(DAY1, "performance")) == (0, 0, 2, 2, 0.0, 0.0)
    assert tuple(built.share(DAY1, "online_servers")) == (1, 0, 2, 2, 0.5, 0.0)
    (prob,) = built.rows("SELECT prob FROM core.bridge_review_aspect WHERE review_id = 3")
    assert prob.prob == pytest.approx(0.9)


def test_returning_to_the_first_model_brings_back_its_labels(built: Built) -> None:
    # Runs after the re-score: the pin goes back to the first model, whose rows the scoring
    # stage re-stamps with the returning run (it doesn't predict again).
    with built.admin.begin() as connection:
        connection.execute(
            text(
                "UPDATE raw.review_aspect_score SET scored_run_id = 5, scored_at = :at "
                "WHERE model_version = :model AND recommendation_id IN (2, 3)"
            ),
            {"at": (DAY3 + timedelta(days=2)).replace(tzinfo=None), "model": FIRST.model_version},
        )

    again = built.build()

    assert again.ok, again.failures
    assert built.bridge() == [
        (1, "ux_controls", SECOND.model_version),  # the first model never scored this version
        (2, "content", FIRST.model_version),
        (2, "performance", FIRST.model_version),
        (3, "performance", FIRST.model_version),
        (3, "stability_bugs", FIRST.model_version),
    ]
