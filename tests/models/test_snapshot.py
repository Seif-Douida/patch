"""The local snapshot Phase 2 works from (design D1): `pull` copies reviews to Parquet.

The integration tests run `pull` against the SQL Server container; in real use the engine comes
from `make_entra_engine` and points at Azure SQL.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import Engine, text

from patchpulse.config import Settings
from patchpulse.db.engine import make_engine
from patchpulse.ingest.games import Game, Role
from patchpulse.ingest.load import upsert_classifications, upsert_games, upsert_news, upsert_reviews
from patchpulse.ingest.models import NewsItem, ReviewRecord
from patchpulse.ingest.patches import Classification, PatchType
from patchpulse.models.cli import main
from patchpulse.models.snapshot import (
    GAMES_FILE,
    INDEX_COLUMNS,
    PATCHES_FILE,
    REVIEWS_FILE,
    explain_pull_error,
    load_snapshot,
    pull,
)

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


# core.review_index as dbt builds it (Phase 3, plan Task 2); the pull only copies it.
_INDEX_DDL = """
CREATE TABLE core.review_index (
    review_id BIGINT NOT NULL, appid INT NOT NULL, [day] INT NOT NULL, created_ts BIGINT NOT NULL,
    [language] NVARCHAR(32), voted_up BIT, playtime_bucket NVARCHAR(16),
    first_seen_run_id BIGINT, [version] DATETIME2, aspects NVARCHAR(400),
    aspect_model_version NVARCHAR(64), rules_version NVARCHAR(8), fingerprint CHAR(64)
)
"""


def index_row(rid: int, *, aspects: str = "") -> dict[str, object]:
    return {
        "review_id": rid,
        "appid": 553850,
        "day": 20260920,
        "created_ts": int(CREATED.timestamp()),
        "language": "english",
        "voted_up": rid % 2 == 0,
        "playtime_bucket": "2h_10h",
        "first_seen_run_id": 1,
        "version": CREATED.replace(tzinfo=None),
        "aspects": aspects,
        "aspect_model_version": "s0-20261013T000000Z" if aspects else None,
        "rules_version": "r1",
        "fingerprint": f"{rid:064x}",
    }


def set_index(engine: Engine, rows: list[dict[str, object]]) -> None:
    """Replace core.review_index's rows, creating the table the first time."""
    with engine.begin() as connection:
        if connection.execute(text("SELECT OBJECT_ID('core.review_index')")).scalar() is None:
            connection.execute(text(_INDEX_DDL))
        connection.execute(text("DELETE FROM core.review_index"))
        for row in rows:
            columns = ", ".join(f"[{name}]" for name in row)
            values = ", ".join(f":{name}" for name in row)
            connection.execute(
                text(f"INSERT INTO core.review_index ({columns}) VALUES ({values})"), row
            )


def seed(engine: Engine, *, index: bool = True) -> None:
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
    if index:
        set_index(engine, [index_row(1, aspects="performance|stability_bugs"), index_row(3)])


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


@pytest.mark.integration
def test_pull_writes_the_index_set(migrated_engine: Engine, tmp_path: Path) -> None:
    seed(migrated_engine)

    result = pull(migrated_engine, tmp_path, now=NOW)

    assert result.index_total == 2
    index = load_snapshot(tmp_path).index
    assert list(index.columns) == list(INDEX_COLUMNS)  # every column, and no text
    assert sorted(index["review_id"]) == [1, 3]
    first = index.set_index("review_id").loc[1]
    assert (first["aspects"], first["aspect_model_version"]) == (
        "performance|stability_bugs",
        "s0-20261013T000000Z",
    )
    assert first["fingerprint"] == f"{1:064x}"
    # The cap moves membership, so each pull replaces the index set whole.
    set_index(migrated_engine, [index_row(2), index_row(3)])
    pull(migrated_engine, tmp_path, now=NOW)
    assert sorted(load_snapshot(tmp_path).index["review_id"]) == [2, 3]


@pytest.mark.integration
def test_pull_without_the_index_table_names_it(migrated_engine: Engine, tmp_path: Path) -> None:
    seed(migrated_engine, index=False)

    with pytest.raises(RuntimeError, match=r"core\.review_index") as error:
        pull(migrated_engine, tmp_path, now=NOW)

    assert explain_pull_error(error.value) == str(error.value)
    assert "nightly" in str(error.value)
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_phase_2_commands_still_load_a_snapshot_without_an_index(tmp_path: Path) -> None:
    pd.DataFrame({"review_id": [1], "text": ["fine"]}).to_parquet(tmp_path / REVIEWS_FILE)
    pd.DataFrame({"appid": [553850]}).to_parquet(tmp_path / GAMES_FILE)
    pd.DataFrame({"appid": [553850]}).to_parquet(tmp_path / PATCHES_FILE)

    snapshot = load_snapshot(tmp_path)

    assert snapshot.index.empty
    assert list(snapshot.index.columns) == list(INDEX_COLUMNS)
    assert list(snapshot.reviews["review_id"]) == [1]


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
