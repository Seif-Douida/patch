"""The local snapshot Phase 2 works from (design D1): `pull` copies reviews to Parquet.

The integration tests run `pull` against the SQL Server container; in real use the engine comes
from `make_entra_engine` and points at Azure SQL.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import Engine

from patchpulse.config import Settings
from patchpulse.db.engine import make_engine
from patchpulse.ingest.games import Game, Role
from patchpulse.ingest.load import upsert_classifications, upsert_games, upsert_news, upsert_reviews
from patchpulse.ingest.models import NewsItem, ReviewRecord
from patchpulse.ingest.patches import Classification, PatchType
from patchpulse.models.cli import main
from patchpulse.models.snapshot import explain_pull_error, load_snapshot, pull

CREATED = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
NOW = datetime(2026, 10, 7, 9, 0, tzinfo=UTC)


def review(
    rid: int, *, appid: int = 553850, text: str = "", updated: datetime = CREATED
) -> ReviewRecord:
    return ReviewRecord(
        recommendation_id=rid,
        appid=appid,
        author_hash="a" * 64,
        author_num_games_owned=10,
        author_num_reviews=2,
        author_playtime_forever=600,
        author_playtime_at_review=300,
        author_playtime_last_two_weeks=5,
        language="english",
        review_text=text or f"review {rid}",
        timestamp_created=CREATED,
        timestamp_updated=updated,
        voted_up=rid % 2 == 0,
        votes_up=3,
        votes_funny=0,
        weighted_vote_score=0.52,
        comment_count=0,
        steam_purchase=True,
        received_for_free=False,
        written_during_early_access=False,
        primarily_steam_deck=None,
    )


def news(gid: str, appid: int, title: str) -> NewsItem:
    return NewsItem(
        gid=gid,
        appid=appid,
        title=title,
        url=f"https://example.com/{gid}",
        contents="",
        published_at=CREATED + timedelta(days=1),
        feedname="steam_community_announcements",
        feedlabel="Community Announcements",
        tags=[],
    )


def seed(engine: Engine) -> None:
    games = [
        Game(appid=553850, name="Helldivers 2", genres=["Action"], role=Role.TRACKED,
             backfill_start=CREATED.date()),
        Game(appid=292030, name="The Witcher 3", genres=["RPG"], role=Role.TRACKED,
             backfill_start=CREATED.date()),
    ]  # fmt: skip
    with engine.begin() as connection:
        upsert_games(connection, games, run_id=1)
        upsert_reviews(connection, [review(1), review(2), review(3, appid=292030)], run_id=1)
        upsert_news(
            connection, [news("10", 553850, "Patch 1.2"), news("11", 553850, "Sale")], run_id=1
        )
        upsert_classifications(
            connection,
            {
                "10": Classification(PatchType.PATCH, "patch_signal"),
                "11": Classification(PatchType.MARKETING, "sale"),
            },
            run_id=1,
        )


@pytest.mark.integration
def test_pull_writes_reviews_games_and_treatment_patches(
    migrated_engine: Engine, tmp_path: Path
) -> None:
    seed(migrated_engine)

    result = pull(migrated_engine, tmp_path, now=NOW)

    snapshot = load_snapshot(tmp_path)
    assert result.reviews_new == 3
    assert result.reviews_total == 3
    assert sorted(snapshot.reviews["review_id"]) == [1, 2, 3]
    assert set(snapshot.reviews.columns) >= {
        "review_id", "appid", "date", "language", "voted_up", "playtime_at_review",
        "timestamp_updated", "last_seen_run_id", "text",
    }  # fmt: skip
    assert snapshot.reviews.set_index("review_id").loc[3, "text"] == "review 3"
    assert sorted(snapshot.games["appid"]) == [292030, 553850]
    assert list(snapshot.patches["title"]) == ["Patch 1.2"]  # the sale isn't a treatment


@pytest.mark.integration
def test_second_pull_fetches_only_new_rows(migrated_engine: Engine, tmp_path: Path) -> None:
    seed(migrated_engine)
    pull(migrated_engine, tmp_path, now=NOW)
    with migrated_engine.begin() as connection:
        upsert_reviews(connection, [review(4)], run_id=2)

    result = pull(migrated_engine, tmp_path, now=NOW)

    assert result.reviews_new == 1
    assert result.reviews_total == 4
    assert result.max_run_id == 2


@pytest.mark.integration
def test_edited_review_replaces_its_old_version(migrated_engine: Engine, tmp_path: Path) -> None:
    seed(migrated_engine)
    pull(migrated_engine, tmp_path, now=NOW)
    edited = review(1, text="edited: now it crashes", updated=CREATED + timedelta(days=2))
    with migrated_engine.begin() as connection:
        upsert_reviews(connection, [edited], run_id=2)

    pull(migrated_engine, tmp_path, now=NOW)

    reviews = load_snapshot(tmp_path).reviews
    assert list(reviews["review_id"]).count(1) == 1
    assert reviews.set_index("review_id").loc[1, "text"] == "edited: now it crashes"


@pytest.mark.integration
def test_failed_pull_leaves_the_previous_snapshot_intact(
    migrated_engine: Engine, new_database: Callable[[], Settings], tmp_path: Path
) -> None:
    seed(migrated_engine)
    pull(migrated_engine, tmp_path, now=NOW)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    empty = make_engine(new_database())  # no raw tables: the queries fail

    with pytest.raises(Exception, match="steam_review"):
        pull(empty, tmp_path, now=NOW)

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in list(os.environ):
        if name.startswith("PP_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)  # no .env here


def test_pull_refuses_without_yes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_env(monkeypatch, tmp_path)
    monkeypatch.setenv("PP_PULL_HOST", "sql.example.net")

    assert main(["pull"]) == 2
    err = capsys.readouterr().err
    assert "--yes" in err
    assert "vCore" in err


def test_lab_settings_name_missing_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_env(monkeypatch, tmp_path)

    assert main(["pull", "--yes"]) == 2
    assert "PP_PULL_HOST" in capsys.readouterr().err


def test_firewall_and_login_errors_name_the_fix() -> None:
    firewall = RuntimeError(
        "Cannot open server 'sql-x' requested by the login. Client with IP address "
        "'81.2.3.4' is not allowed to access the server."
    )
    login = RuntimeError("DefaultAzureCredential failed to retrieve a token from the credentials")

    firewall_help = explain_pull_error(firewall)
    assert firewall_help is not None
    assert "DEV_IP_ADDRESS" in firewall_help
    assert "81.2.3.4" not in firewall_help  # a home IP is personal data
    login_help = explain_pull_error(login)
    assert login_help is not None
    assert "az login" in login_help
    assert explain_pull_error(RuntimeError("something else")) is None


def test_azure_sign_in_logs_are_quiet() -> None:
    # azure-identity logs each credential it tries and dumps HTTP requests at INFO.
    import logging

    from patchpulse.models.cli import configure_logging

    configure_logging()

    assert logging.getLogger("azure").getEffectiveLevel() >= logging.WARNING
    assert logging.getLogger("patchpulse").getEffectiveLevel() == logging.INFO
