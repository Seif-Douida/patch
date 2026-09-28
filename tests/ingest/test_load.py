"""Loading Steam data into the raw and ops tables (integration, on the SQL Server container)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import Engine, inspect, text

from patchpulse.db import tables
from patchpulse.ingest.coverage import Coverage
from patchpulse.ingest.games import Game, Role
from patchpulse.ingest.load import (
    purge_raw_pages,
    read_coverage,
    record_dq,
    record_page,
    upsert_classifications,
    upsert_games,
    upsert_news,
    upsert_prices,
    upsert_reviews,
    write_coverage,
)
from patchpulse.ingest.models import NewsItem, PriceSnapshot, RawPage, ReviewRecord
from patchpulse.ingest.patches import Classification, PatchType

pytestmark = pytest.mark.integration

CREATED = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)


def review(rid: int, *, updated: datetime = CREATED, votes_funny: int = 0) -> ReviewRecord:
    return ReviewRecord(
        recommendation_id=rid,
        appid=553850,
        author_hash="a" * 64,
        author_num_games_owned=10,
        author_num_reviews=2,
        author_playtime_forever=600,
        author_playtime_at_review=300,
        author_playtime_last_two_weeks=5,
        language="english",
        review_text=f"review {rid}",
        timestamp_created=CREATED,
        timestamp_updated=updated,
        voted_up=True,
        votes_up=3,
        votes_funny=votes_funny,
        weighted_vote_score=0.52,
        comment_count=0,
        steam_purchase=True,
        received_for_free=False,
        written_during_early_access=False,
        primarily_steam_deck=None,
    )


def rows(engine: Engine, sql: str) -> list[Any]:
    with engine.connect() as connection:
        return list(connection.execute(text(sql)))


def test_table_definitions_match_the_migrated_schema(migrated_engine: Engine) -> None:
    inspector = inspect(migrated_engine)
    for table in tables.metadata.sorted_tables:
        migrated = {c["name"] for c in inspector.get_columns(table.name, schema=table.schema)}
        assert migrated == set(table.columns.keys()), table.fullname


def test_upsert_reviews_keeps_one_row_per_version_and_updates_last_seen(
    migrated_engine: Engine,
) -> None:
    with migrated_engine.begin() as connection:
        assert upsert_reviews(connection, [review(1), review(2)], run_id=1) == 2
    with migrated_engine.begin() as connection:
        assert upsert_reviews(connection, [review(1)], run_id=2) == 0

    stored = rows(
        migrated_engine,
        "SELECT recommendation_id, first_seen_run_id, last_seen_run_id FROM raw.steam_review "
        "ORDER BY recommendation_id",
    )
    assert [tuple(r) for r in stored] == [(1, 1, 2), (2, 1, 1)]


def test_reseen_review_refreshes_its_vote_counts(migrated_engine: Engine) -> None:
    # Votes change without the review being edited (timestamp_updated stays the same).
    with migrated_engine.begin() as connection:
        upsert_reviews(connection, [review(1)], run_id=1)
        later = review(1).model_copy(update={"votes_up": 40, "weighted_vote_score": 0.9})
        upsert_reviews(connection, [later], run_id=2)

    (stored,) = rows(migrated_engine, "SELECT votes_up, weighted_vote_score FROM raw.steam_review")
    assert tuple(stored) == (40, 0.9)


def test_edited_review_adds_a_new_version(migrated_engine: Engine) -> None:
    with migrated_engine.begin() as connection:
        upsert_reviews(connection, [review(1)], run_id=1)
        upsert_reviews(connection, [review(1, updated=CREATED + timedelta(days=2))], run_id=2)

    assert rows(migrated_engine, "SELECT COUNT(*) FROM raw.steam_review")[0][0] == 2


def test_huge_vote_counts_fit(migrated_engine: Engine) -> None:
    with migrated_engine.begin() as connection:
        upsert_reviews(connection, [review(1, votes_funny=4294967295)], run_id=1)

    assert rows(migrated_engine, "SELECT votes_funny FROM raw.steam_review")[0][0] == 4294967295


def test_record_page_stores_the_payload_and_request(migrated_engine: Engine) -> None:
    page = RawPage("reviews", 553850, {"cursor": "*"}, 200, {"success": 1, "reviews": []}, 0)
    with migrated_engine.begin() as connection:
        record_page(connection, page, run_id=5)

    (stored,) = rows(
        migrated_engine,
        "SELECT run_id, source, appid, http_status, request, payload, n_items FROM raw.steam_page",
    )
    assert (stored.run_id, stored.source, stored.appid, stored.http_status) == (
        5,
        "reviews",
        553850,
        200,
    )
    assert json.loads(stored.request) == {"cursor": "*"}
    assert json.loads(stored.payload) == {"success": 1, "reviews": []}


def test_coverage_round_trips(migrated_engine: Engine) -> None:
    coverage = Coverage(
        covered_from=datetime(2024, 10, 1, tzinfo=UTC),
        covered_to=datetime(2026, 9, 28, 3, 0, tzinfo=UTC),
    )
    with migrated_engine.begin() as connection:
        assert read_coverage(connection, 553850) is None
        write_coverage(connection, 553850, coverage, run_id=1)
        write_coverage(connection, 553850, coverage, run_id=2)  # upsert, not a duplicate
        assert read_coverage(connection, 553850) == coverage


def test_upsert_games_updates_changed_fields(migrated_engine: Engine) -> None:
    game = Game(
        appid=553850,
        name="Helldivers 2",
        genres=["Action"],
        role=Role.TRACKED,
        backfill_start=date(2024, 10, 1),
    )
    with migrated_engine.begin() as connection:
        upsert_games(connection, [game], run_id=1)
        upsert_games(connection, [game.model_copy(update={"role": Role.BOTH})], run_id=2)

    stored = rows(migrated_engine, "SELECT appid, role, genres, updated_run_id FROM raw.game")
    assert [tuple(r) for r in stored] == [(553850, "both", "Action", 2)]


def test_upsert_news_inserts_and_updates_last_seen(migrated_engine: Engine) -> None:
    item = NewsItem(
        gid="5812",
        appid=553850,
        title="Patch 7.1.1",
        url="https://example.com/p",
        contents="Fixes",
        published_at=CREATED,
        feedname="steam_community_announcements",
        feedlabel="Community Announcements",
        tags=["patchnotes"],
    )
    with migrated_engine.begin() as connection:
        assert upsert_news(connection, [item], run_id=1) == 1
        assert upsert_news(connection, [item], run_id=2) == 0

    stored = rows(
        migrated_engine, "SELECT gid, tags, first_seen_run_id, last_seen_run_id FROM raw.steam_news"
    )
    assert [tuple(r) for r in stored] == [("5812", '["patchnotes"]', 1, 2)]


def test_upsert_prices_is_idempotent_per_day_and_records_failures(migrated_engine: Engine) -> None:
    snapshots = {
        1091500: PriceSnapshot(currency="USD", initial=5999, final=2999, discount_percent=50),
        553850: None,
    }
    with migrated_engine.begin() as connection:
        upsert_prices(connection, snapshots, snapshot_date=date(2026, 9, 28), run_id=1)
        upsert_prices(connection, snapshots, snapshot_date=date(2026, 9, 28), run_id=2)

    stored = rows(
        migrated_engine,
        "SELECT appid, ok, currency, final_price, discount_percent, run_id FROM raw.steam_price "
        "ORDER BY appid",
    )
    assert [tuple(r) for r in stored] == [
        (553850, False, None, None, None, 2),
        (1091500, True, "USD", 2999, 50, 2),
    ]


def test_purge_removes_only_pages_older_than_the_cutoff(migrated_engine: Engine) -> None:
    page = RawPage("news", 1, {}, 200, {}, 0)
    with migrated_engine.begin() as connection:
        record_page(connection, page, run_id=1)
        record_page(connection, page, run_id=2)
        connection.execute(
            text("UPDATE raw.steam_page SET fetched_at = '2026-08-01' WHERE run_id = 1")
        )
        removed = purge_raw_pages(connection, older_than=datetime(2026, 8, 29, tzinfo=UTC))

    assert removed == 1
    assert [r[0] for r in rows(migrated_engine, "SELECT run_id FROM raw.steam_page")] == [2]


def test_upsert_classifications_keeps_one_row_per_item_and_version(
    migrated_engine: Engine,
) -> None:
    patch = Classification(PatchType.PATCH, "patch_signal")
    unclear = Classification(PatchType.OTHER, "no_rule", ambiguous=True)
    with migrated_engine.begin() as connection:
        upsert_classifications(connection, {"10": patch, "11": unclear}, run_id=1)
        upsert_classifications(connection, {"10": patch}, run_id=2)

    stored = rows(
        migrated_engine,
        "SELECT gid, classifier_version, patch_type, is_treatment_candidate, ambiguous, run_id "
        "FROM raw.news_classification ORDER BY gid",
    )
    assert [tuple(r) for r in stored] == [
        ("10", "rules-v1", "patch", True, False, 2),
        ("11", "rules-v1", "other", False, True, 1),
    ]


def test_record_dq(migrated_engine: Engine) -> None:
    with migrated_engine.begin() as connection:
        record_dq(
            connection, run_id=3, check="cursor_repeat", appid=553850, severity="warn", detail="x"
        )

    stored = rows(
        migrated_engine, "SELECT run_id, check_name, appid, severity FROM ops.data_quality_result"
    )
    assert [tuple(r) for r in stored] == [(3, "cursor_repeat", 553850, "warn")]
