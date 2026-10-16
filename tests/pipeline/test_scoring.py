"""The scoring stage (plan Task 15, design D8): each review's latest version gets a score from the
current champion, newest first, within a time budget; old scores stay. On the SQL Server container,
with a small S0 trained on invented reviews."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import SecretStr
from sqlalchemy import Engine, event, insert, text
from toy_reviews import toy_reviews

from patchpulse.config import Settings
from patchpulse.db import tables
from patchpulse.db.engine import make_engine
from patchpulse.db.migrate import upgrade_to_head
from patchpulse.db.provision import WRITER_USER, provision_writer
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.artifact import Scorer, load_scorer
from patchpulse.models.s0 import S0Model, save_s0, train_s0
from patchpulse.pipeline.scoring import MODEL_NAME, score_reviews

pytestmark = pytest.mark.integration

NOW = datetime(2026, 10, 12, 12, 0, 0)


@pytest.fixture(scope="module")
def s0_model() -> S0Model:
    texts, labels = toy_reviews(300, seed=31)
    return train_s0(texts, labels)


def scorer(model: S0Model, directory: Path, version: str) -> Scorer:
    save_s0(model, directory, thresholds=np.full(len(ASPECTS), 0.5), version=version)
    return load_scorer(directory)


@pytest.fixture
def champion(s0_model: S0Model, tmp_path: Path) -> Scorer:
    return scorer(s0_model, tmp_path / "s0-a", "20261013T000000Z")


def review(rid: int, created: datetime, updated: datetime | None = None) -> dict[str, Any]:
    return {
        "recommendation_id": rid,
        "timestamp_updated": updated or created,
        "appid": 553850,
        "author_hash": "a" * 64,
        "language": "english",
        "review_text": f"the game stutters and fps drops ({rid})",
        "timestamp_created": created,
        "voted_up": False,
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


def add_reviews(engine: Engine, rows: list[dict[str, Any]]) -> None:
    with engine.begin() as connection:
        connection.execute(insert(tables.steam_review), rows)


def scored(engine: Engine, model_version: str | None = None) -> list[tuple[int, datetime]]:
    sql = "SELECT recommendation_id, timestamp_updated FROM raw.review_aspect_score"
    if model_version is not None:
        sql += f" WHERE model_version = '{model_version}'"
    with engine.connect() as connection:
        return sorted((r[0], r[1]) for r in connection.execute(text(sql)))


class Ticks:
    """A monotonic clock that reads the given values in turn, then stays on the last one."""

    def __init__(self, *values: float) -> None:
        self.values = list(values)

    def __call__(self) -> float:
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]


EXPIRES_AFTER_ONE_BATCH = (0.0, 0.0, 61.0)  # deadline set at 0; batch one at 0; then past 60 s


def test_new_reviews_are_scored_newest_first(migrated_engine: Engine, champion: Scorer) -> None:
    add_reviews(migrated_engine, [review(rid, NOW - timedelta(days=rid)) for rid in range(1, 6)])

    stats = score_reviews(
        migrated_engine,
        champion,
        run_id=7,
        budget_s=60,
        clock=Ticks(*EXPIRES_AFTER_ONE_BATCH),
        batch=2,
    )

    assert (stats.scored, stats.remaining) == (2, 3)
    assert stats.model_version == "s0-20261013T000000Z"
    assert [rid for rid, _ in scored(migrated_engine)] == [1, 2]  # the two newest
    with migrated_engine.connect() as connection:
        row = connection.execute(text("SELECT * FROM raw.review_aspect_score")).mappings().first()
    assert row is not None
    assert (row["model_name"], row["scored_run_id"]) == (MODEL_NAME, 7)
    assert all(0.0 <= row[f"p_{code}"] <= 1.0 for code in ASPECTS)
    assert all(row[f"l_{code}"] in (True, False) for code in ASPECTS)


def test_budget_stops_scoring_and_the_next_run_resumes(
    migrated_engine: Engine, champion: Scorer
) -> None:
    add_reviews(migrated_engine, [review(rid, NOW - timedelta(days=rid)) for rid in range(1, 6)])
    score_reviews(
        migrated_engine,
        champion,
        run_id=7,
        budget_s=60,
        clock=Ticks(*EXPIRES_AFTER_ONE_BATCH),
        batch=2,
    )

    stats = score_reviews(migrated_engine, champion, run_id=8, budget_s=60, clock=Ticks(0.0))

    assert (stats.scored, stats.remaining) == (3, 0)
    assert [rid for rid, _ in scored(migrated_engine)] == [1, 2, 3, 4, 5]


def test_one_to_do_list_a_night_whatever_the_pages(
    migrated_engine: Engine, champion: Scorer
) -> None:
    # Run 19 (16 Oct): every page looked for unscored reviews by checking the score table row by
    # row from the newest review down, so a night first stepped over every review already scored
    # (746k) and the backlog slowed to a crawl. The night's to-do list is now built once.
    add_reviews(migrated_engine, [review(rid, NOW - timedelta(days=rid)) for rid in range(1, 8)])
    statements: list[str] = []

    def record(*args: Any) -> None:
        statements.append(str(args[2]))  # (connection, cursor, statement, ...)

    event.listen(migrated_engine, "before_cursor_execute", record)
    try:
        stats = score_reviews(
            migrated_engine, champion, run_id=7, budget_s=60, clock=Ticks(0.0), batch=2
        )
    finally:
        event.remove(migrated_engine, "before_cursor_execute", record)

    assert (stats.scored, stats.remaining) == (7, 0)
    unscored_checks = [s for s in statements if "NOT EXISTS" in s and "review_aspect_score" in s]
    assert len(unscored_checks) == 1  # 4 pages, one to-do list


def test_edited_review_is_rescored(migrated_engine: Engine, champion: Scorer) -> None:
    add_reviews(migrated_engine, [review(rid, NOW - timedelta(days=rid)) for rid in range(1, 4)])
    score_reviews(migrated_engine, champion, run_id=7, budget_s=60, clock=Ticks(0.0))
    edited = NOW - timedelta(hours=1)
    add_reviews(migrated_engine, [review(2, NOW - timedelta(days=2), updated=edited)])

    stats = score_reviews(migrated_engine, champion, run_id=8, budget_s=60, clock=Ticks(0.0))

    assert (stats.scored, stats.remaining) == (1, 0)
    versions = [updated for rid, updated in scored(migrated_engine) if rid == 2]
    assert versions == [NOW - timedelta(days=2), edited]  # the old score stays in raw


def test_new_champion_rescores_everything_newest_first(
    migrated_engine: Engine, champion: Scorer, s0_model: S0Model, tmp_path: Path
) -> None:
    add_reviews(migrated_engine, [review(rid, NOW - timedelta(days=rid)) for rid in range(1, 5)])
    score_reviews(migrated_engine, champion, run_id=7, budget_s=60, clock=Ticks(0.0))
    successor = scorer(s0_model, tmp_path / "s0-b", "20261014T000000Z")

    stats = score_reviews(
        migrated_engine,
        successor,
        run_id=8,
        budget_s=60,
        clock=Ticks(*EXPIRES_AFTER_ONE_BATCH),
        batch=2,
    )

    assert (stats.scored, stats.remaining) == (2, 2)
    assert [rid for rid, _ in scored(migrated_engine, "s0-20261014T000000Z")] == [1, 2]
    assert len(scored(migrated_engine, "s0-20261013T000000Z")) == 4


def latest_models(engine: Engine) -> dict[int, str]:
    """Each review version's latest score's model (the highest run id), as dbt reads it."""
    sql = """
        SELECT recommendation_id, model_version FROM (
            SELECT recommendation_id, model_version, ROW_NUMBER() OVER (
                PARTITION BY recommendation_id, timestamp_updated
                ORDER BY scored_run_id DESC, scored_at DESC, model_version DESC) AS rank_
            FROM raw.review_aspect_score) AS ranked
        WHERE rank_ = 1
    """
    with engine.connect() as connection:
        return {r[0]: r[1] for r in connection.execute(text(sql))}


def test_returning_to_a_model_makes_its_scores_the_latest_again(
    migrated_engine: Engine, champion: Scorer, s0_model: S0Model, tmp_path: Path
) -> None:
    # Champion A, then B re-scores everything, then the pin goes back to A (a rollback). A's
    # scores are re-stamped with the run that returned to it, not predicted again.
    add_reviews(migrated_engine, [review(rid, NOW - timedelta(days=rid)) for rid in range(1, 4)])
    score_reviews(migrated_engine, champion, run_id=7, budget_s=60, clock=Ticks(0.0))
    successor = scorer(s0_model, tmp_path / "s0-b", "20261014T000000Z")
    score_reviews(migrated_engine, successor, run_id=8, budget_s=60, clock=Ticks(0.0))
    assert set(latest_models(migrated_engine).values()) == {"s0-20261014T000000Z"}

    back = score_reviews(migrated_engine, champion, run_id=9, budget_s=60, clock=Ticks(0.0))

    assert (back.scored, back.restamped, back.remaining) == (0, 3, 0)
    assert set(latest_models(migrated_engine).values()) == {"s0-20261013T000000Z"}
    assert len(scored(migrated_engine, "s0-20261014T000000Z")) == 3  # B's scores stay in raw
    again = score_reviews(migrated_engine, champion, run_id=10, budget_s=60, clock=Ticks(0.0))
    assert (again.scored, again.restamped, again.remaining) == (0, 0, 0)


def test_pp_writer_can_write_scores(
    new_database: Callable[[], Settings], champion: Scorer, s0_model: S0Model, tmp_path: Path
) -> None:
    admin = new_database()
    engine = make_engine(admin)
    upgrade_to_head(engine)
    with engine.begin() as connection:
        provision_writer(connection, password="Writer-One-1")
    add_reviews(engine, [review(rid, NOW - timedelta(days=rid)) for rid in range(1, 4)])
    writer = make_engine(
        admin.model_copy(update={"db_user": WRITER_USER, "db_password": SecretStr("Writer-One-1")})
    )

    stats = score_reviews(writer, champion, run_id=7, budget_s=60, clock=Ticks(0.0))

    assert stats.scored == 3
    # Returning to a model re-stamps its rows: an UPDATE the writer may make too.
    successor = scorer(s0_model, tmp_path / "s0-b", "20261014T000000Z")
    score_reviews(writer, successor, run_id=8, budget_s=60, clock=Ticks(0.0))
    back = score_reviews(writer, champion, run_id=9, budget_s=60, clock=Ticks(0.0))
    assert back.restamped == 3
    writer.dispose()
    engine.dispose()
