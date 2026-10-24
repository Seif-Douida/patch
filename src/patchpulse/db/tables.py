"""SQLAlchemy Core definitions of the tables Alembic owns, for the loaders.

They mirror migrations 0001 to 0005; `test_table_definitions_match_the_migrated_schema` keeps
them in step. Datetime columns hold naive UTC (DATETIME2 has no time zone).
"""

from __future__ import annotations

from sqlalchemy import (
    CHAR,
    REAL,
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    Table,
    Unicode,
    UnicodeText,
)
from sqlalchemy.dialects.mssql import TINYINT

metadata = MetaData()

game = Table(
    "game",
    metadata,
    Column("appid", Integer, primary_key=True, autoincrement=False),
    Column("name", Unicode(200), nullable=False),
    Column("genres", Unicode(400), nullable=False),
    Column("role", Unicode(16), nullable=False),
    Column("backfill_start", Date, nullable=False),
    Column("release_date", Date),
    Column("developer", Unicode(200)),
    Column("publisher", Unicode(200)),
    Column("updated_run_id", BigInteger, nullable=False),
    schema="raw",
)

steam_page = Table(
    "steam_page",
    metadata,
    Column("page_id", BigInteger, primary_key=True),
    Column("run_id", BigInteger, nullable=False),
    Column("source", Unicode(16), nullable=False),
    Column("appid", Integer),
    Column("request", UnicodeText, nullable=False),
    Column("fetched_at", DateTime),
    Column("http_status", Integer, nullable=False),
    Column("payload", UnicodeText, nullable=False),
    Column("n_items", Integer, nullable=False),
    schema="raw",
)

steam_review = Table(
    "steam_review",
    metadata,
    Column("recommendation_id", BigInteger, primary_key=True, autoincrement=False),
    Column("timestamp_updated", DateTime, primary_key=True),
    Column("appid", Integer, nullable=False),
    Column("author_hash", CHAR(64), nullable=False),
    Column("author_num_games_owned", Integer),
    Column("author_num_reviews", Integer),
    Column("author_playtime_forever", Integer),
    Column("author_playtime_at_review", Integer),
    Column("author_playtime_last_two_weeks", Integer),
    Column("language", Unicode(32), nullable=False),
    Column("review_text", UnicodeText, nullable=False),
    Column("timestamp_created", DateTime, nullable=False),
    Column("voted_up", Boolean, nullable=False),
    Column("votes_up", BigInteger, nullable=False),
    Column("votes_funny", BigInteger, nullable=False),
    Column("weighted_vote_score", Float, nullable=False),
    Column("comment_count", BigInteger, nullable=False),
    Column("steam_purchase", Boolean, nullable=False),
    Column("received_for_free", Boolean, nullable=False),
    Column("written_during_early_access", Boolean, nullable=False),
    Column("primarily_steam_deck", Boolean),
    Column("first_seen_run_id", BigInteger, nullable=False),
    Column("last_seen_run_id", BigInteger, nullable=False),
    schema="raw",
)

steam_news = Table(
    "steam_news",
    metadata,
    Column("gid", Unicode(32), primary_key=True),
    Column("appid", Integer, nullable=False),
    Column("title", Unicode(500), nullable=False),
    Column("url", Unicode(1000), nullable=False),
    Column("contents", UnicodeText, nullable=False),
    Column("published_at", DateTime, nullable=False),
    Column("feedname", Unicode(100), nullable=False),
    Column("feedlabel", Unicode(200), nullable=False),
    Column("tags", Unicode(1000), nullable=False),
    Column("first_seen_run_id", BigInteger, nullable=False),
    Column("last_seen_run_id", BigInteger, nullable=False),
    schema="raw",
)

news_classification = Table(
    "news_classification",
    metadata,
    Column("gid", Unicode(32), primary_key=True),
    Column("classifier_version", Unicode(32), primary_key=True),
    Column("patch_type", Unicode(16), nullable=False),
    Column("is_treatment_candidate", Boolean, nullable=False),
    Column("rule", Unicode(64), nullable=False),
    Column("ambiguous", Boolean, nullable=False),
    Column("run_id", BigInteger, nullable=False),
    schema="raw",
)

steam_price = Table(
    "steam_price",
    metadata,
    Column("appid", Integer, primary_key=True, autoincrement=False),
    Column("snapshot_date", Date, primary_key=True),
    Column("ok", Boolean, nullable=False),
    Column("currency", Unicode(3)),
    Column("initial_price", Integer),
    Column("final_price", Integer),
    Column("discount_percent", Integer),
    Column("run_id", BigInteger, nullable=False),
    schema="raw",
)

pipeline_run = Table(
    "pipeline_run",
    metadata,
    Column("run_id", BigInteger, primary_key=True),
    Column("kind", Unicode(16), nullable=False),
    Column("started_at", DateTime),
    Column("finished_at", DateTime),
    Column("status", Unicode(16), nullable=False),
    Column("image_version", Unicode(64)),
    Column("stages", UnicodeText),
    Column("error", Unicode(4000)),
    schema="ops",
)

ingest_state = Table(
    "ingest_state",
    metadata,
    Column("appid", Integer, primary_key=True, autoincrement=False),
    Column("covered_from", DateTime, nullable=False),
    Column("covered_to", DateTime, nullable=False),
    Column("updated_run_id", BigInteger, nullable=False),
    schema="ops",
)

data_quality_result = Table(
    "data_quality_result",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("run_id", BigInteger, nullable=False),
    Column("check_name", Unicode(64), nullable=False),
    Column("appid", Integer),
    Column("severity", Unicode(8), nullable=False),
    Column("detail", Unicode(2000), nullable=False),
    Column("created_at", DateTime),
    schema="ops",
)


# Taxonomy v1's aspects, as migration 0002 created their columns.
_ASPECTS = (
    "performance",
    "stability_bugs",
    "gameplay_balance",
    "content",
    "monetization",
    "online_servers",
    "story_world",
    "ux_controls",
    "platform_policy",
    "off_topic",
)

review_aspect_score = Table(
    "review_aspect_score",
    metadata,
    Column("recommendation_id", BigInteger, primary_key=True, autoincrement=False),
    Column("timestamp_updated", DateTime, primary_key=True),
    Column("model_version", Unicode(64), primary_key=True),
    Column("model_name", Unicode(64), nullable=False),
    Column("scored_run_id", BigInteger, nullable=False),
    Column("scored_at", DateTime),
    *(Column(f"p_{code}", REAL, nullable=False) for code in _ASPECTS),
    *(Column(f"l_{code}", Boolean, nullable=False) for code in _ASPECTS),
    schema="raw",
)


# The nightly index stage's record of Qdrant (migration 0003, Phase 3 design D5).
qdrant_point = Table(
    "qdrant_point",
    metadata,
    Column("collection", Unicode(64), primary_key=True),
    Column("review_id", BigInteger, primary_key=True, autoincrement=False),
    Column("version", DateTime, nullable=False),
    Column("fingerprint", CHAR(64), nullable=False),
    Column("synced_run_id", BigInteger),
    schema="ops",
)

qdrant_reconciliation = Table(
    "qdrant_reconciliation",
    metadata,
    Column("run_id", BigInteger, primary_key=True, autoincrement=False),
    Column("collection", Unicode(64), primary_key=True),
    Column("appid", Integer, primary_key=True, autoincrement=False),
    Column("day", Integer, primary_key=True, autoincrement=False),
    Column("sql_count", Integer, nullable=False),
    Column("qdrant_count", Integer, nullable=False),
    schema="ops",
)


# Phase 4 (migration 0005): the comparison games' daily counts and the stored effect estimates.
steam_review_daily = Table(
    "steam_review_daily",
    metadata,
    Column("appid", Integer, primary_key=True, autoincrement=False),
    Column("day", Date, primary_key=True),
    Column("positive", Integer, nullable=False),
    Column("negative", Integer, nullable=False),
    Column("fetched_run_id", BigInteger, nullable=False),
    Column("fetched_at", DateTime, nullable=False),
    schema="raw",
)

patch_effect = Table(
    "patch_effect",
    metadata,
    Column("bundle_id", Unicode(32), primary_key=True),
    Column("patch_ids", Unicode(400), nullable=False),
    Column("appid", Integer, nullable=False),
    Column("outcome", Unicode(64), primary_key=True),
    Column("sample", Unicode(16), primary_key=True),
    Column("horizon", TINYINT, primary_key=True, autoincrement=False),
    Column("method", Unicode(8), primary_key=True),
    Column("run_id", BigInteger, primary_key=True, autoincrement=False),
    Column("effect", REAL),
    Column("lo90", REAL),
    Column("hi90", REAL),
    Column("p_value", REAL),
    Column("pre_fit_rmse", REAL),
    Column("donors", UnicodeText, nullable=False),
    Column("flags", Unicode(200), nullable=False),
    Column("reason", Unicode(200)),
    Column("fingerprint", CHAR(64), nullable=False),
    Column("version", Unicode(8), nullable=False),
    Column("computed_at", DateTime, nullable=False),
    schema="app",
)
