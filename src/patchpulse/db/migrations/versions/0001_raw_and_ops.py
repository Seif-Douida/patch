"""Schemas raw, ops (owned here) and stg, core, mart (pre-created for dbt); raw and ops tables.

Revision ID: 0001
Revises:
Create Date: 2026-09-28

Steam timestamps are stored as UTC DATETIME2(0); pipeline times as DATETIME2(3). Author IDs never
reach the database: only `author_hash` (HMAC-SHA256 with a secret salt, spec C4).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mssql import DATETIME2

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

ALEMBIC_SCHEMAS = ("raw", "ops")
# SQLAlchemy's DATETIME2 constructor is untyped; build the two precisions once.
STEAM_TIME = DATETIME2(0)  # type: ignore[no-untyped-call]
PIPELINE_TIME = DATETIME2(3)  # type: ignore[no-untyped-call]
DBT_SCHEMAS = ("stg", "core", "mart")


def _utc_now() -> sa.TextClause:
    return sa.text("SYSUTCDATETIME()")


def upgrade() -> None:
    for schema in (*ALEMBIC_SCHEMAS, *DBT_SCHEMAS):
        op.execute(f"CREATE SCHEMA [{schema}]")

    op.create_table(
        "pipeline_run",
        sa.Column("run_id", sa.BigInteger, sa.Identity(start=1), primary_key=True),
        sa.Column("kind", sa.Unicode(16), nullable=False),
        sa.Column("started_at", PIPELINE_TIME, nullable=False, server_default=_utc_now()),
        sa.Column("finished_at", PIPELINE_TIME),
        sa.Column("status", sa.Unicode(16), nullable=False),
        sa.Column("image_version", sa.Unicode(64)),
        sa.Column("stages", sa.UnicodeText),
        sa.Column("error", sa.Unicode(4000)),
        schema="ops",
    )
    op.create_table(
        "ingest_state",
        sa.Column("appid", sa.Integer, primary_key=True, autoincrement=False),
        sa.Column("covered_from", STEAM_TIME, nullable=False),
        sa.Column("covered_to", STEAM_TIME, nullable=False),
        sa.Column("updated_run_id", sa.BigInteger, nullable=False),
        schema="ops",
    )
    op.create_table(
        "data_quality_result",
        sa.Column("id", sa.BigInteger, sa.Identity(start=1), primary_key=True),
        sa.Column("run_id", sa.BigInteger, nullable=False),
        sa.Column("check_name", sa.Unicode(64), nullable=False),
        sa.Column("appid", sa.Integer),
        sa.Column("severity", sa.Unicode(8), nullable=False),
        sa.Column("detail", sa.Unicode(2000), nullable=False),
        sa.Column("created_at", PIPELINE_TIME, nullable=False, server_default=_utc_now()),
        schema="ops",
    )

    op.create_table(
        "game",
        # A lone integer primary key would otherwise become an IDENTITY column on SQL Server.
        sa.Column("appid", sa.Integer, primary_key=True, autoincrement=False),
        sa.Column("name", sa.Unicode(200), nullable=False),
        sa.Column("genres", sa.Unicode(400), nullable=False),
        sa.Column("role", sa.Unicode(16), nullable=False),
        sa.Column("backfill_start", sa.Date, nullable=False),
        sa.Column("release_date", sa.Date),
        sa.Column("developer", sa.Unicode(200)),
        sa.Column("publisher", sa.Unicode(200)),
        sa.Column("updated_run_id", sa.BigInteger, nullable=False),
        schema="raw",
    )
    op.create_table(
        "steam_page",
        sa.Column("page_id", sa.BigInteger, sa.Identity(start=1), primary_key=True),
        sa.Column("run_id", sa.BigInteger, nullable=False),
        sa.Column("source", sa.Unicode(16), nullable=False),
        sa.Column("appid", sa.Integer),
        sa.Column("request", sa.UnicodeText, nullable=False),
        sa.Column("fetched_at", PIPELINE_TIME, nullable=False, server_default=_utc_now()),
        sa.Column("http_status", sa.Integer, nullable=False),
        sa.Column("payload", sa.UnicodeText, nullable=False),
        sa.Column("n_items", sa.Integer, nullable=False),
        schema="raw",
    )
    op.create_index("ix_steam_page_fetched_at", "steam_page", ["fetched_at"], schema="raw")
    op.create_table(
        "steam_review",
        sa.Column("recommendation_id", sa.BigInteger, primary_key=True),
        sa.Column("timestamp_updated", STEAM_TIME, primary_key=True),
        sa.Column("appid", sa.Integer, nullable=False),
        sa.Column("author_hash", sa.CHAR(64), nullable=False),
        sa.Column("author_num_games_owned", sa.Integer),
        sa.Column("author_num_reviews", sa.Integer),
        sa.Column("author_playtime_forever", sa.Integer),
        sa.Column("author_playtime_at_review", sa.Integer),
        sa.Column("author_playtime_last_two_weeks", sa.Integer),
        sa.Column("language", sa.Unicode(32), nullable=False),
        sa.Column("review_text", sa.UnicodeText, nullable=False),
        sa.Column("timestamp_created", STEAM_TIME, nullable=False),
        sa.Column("voted_up", sa.Boolean, nullable=False),
        # Steam sometimes sends 4294967295 (an unsigned -1) for vote counts; INT can't hold it.
        sa.Column("votes_up", sa.BigInteger, nullable=False),
        sa.Column("votes_funny", sa.BigInteger, nullable=False),
        sa.Column("weighted_vote_score", sa.Float, nullable=False),
        sa.Column("comment_count", sa.BigInteger, nullable=False),
        sa.Column("steam_purchase", sa.Boolean, nullable=False),
        sa.Column("received_for_free", sa.Boolean, nullable=False),
        sa.Column("written_during_early_access", sa.Boolean, nullable=False),
        sa.Column("primarily_steam_deck", sa.Boolean),
        sa.Column("first_seen_run_id", sa.BigInteger, nullable=False),
        sa.Column("last_seen_run_id", sa.BigInteger, nullable=False),
        schema="raw",
    )
    op.create_index(
        "ix_steam_review_appid_created",
        "steam_review",
        ["appid", "timestamp_created"],
        schema="raw",
    )
    op.create_index("ix_steam_review_last_seen", "steam_review", ["last_seen_run_id"], schema="raw")
    op.create_table(
        "steam_news",
        sa.Column("gid", sa.Unicode(32), primary_key=True),
        sa.Column("appid", sa.Integer, nullable=False),
        sa.Column("title", sa.Unicode(500), nullable=False),
        sa.Column("url", sa.Unicode(1000), nullable=False),
        sa.Column("contents", sa.UnicodeText, nullable=False),
        sa.Column("published_at", STEAM_TIME, nullable=False),
        sa.Column("feedname", sa.Unicode(100), nullable=False),
        sa.Column("feedlabel", sa.Unicode(200), nullable=False),
        sa.Column("tags", sa.Unicode(1000), nullable=False),
        sa.Column("first_seen_run_id", sa.BigInteger, nullable=False),
        sa.Column("last_seen_run_id", sa.BigInteger, nullable=False),
        schema="raw",
    )
    op.create_index("ix_steam_news_appid", "steam_news", ["appid", "published_at"], schema="raw")
    op.create_table(
        "news_classification",
        sa.Column("gid", sa.Unicode(32), primary_key=True),
        sa.Column("classifier_version", sa.Unicode(32), primary_key=True),
        sa.Column("patch_type", sa.Unicode(16), nullable=False),
        sa.Column("is_treatment_candidate", sa.Boolean, nullable=False),
        sa.Column("rule", sa.Unicode(64), nullable=False),
        sa.Column("ambiguous", sa.Boolean, nullable=False),
        sa.Column("run_id", sa.BigInteger, nullable=False),
        schema="raw",
    )
    op.create_table(
        "steam_price",
        sa.Column("appid", sa.Integer, primary_key=True),
        sa.Column("snapshot_date", sa.Date, primary_key=True),
        sa.Column("ok", sa.Boolean, nullable=False),
        sa.Column("currency", sa.Unicode(3)),
        sa.Column("initial_price", sa.Integer),
        sa.Column("final_price", sa.Integer),
        sa.Column("discount_percent", sa.Integer),
        sa.Column("run_id", sa.BigInteger, nullable=False),
        schema="raw",
    )


def downgrade() -> None:
    for table in (
        "steam_price",
        "news_classification",
        "steam_news",
        "steam_review",
        "steam_page",
        "game",
    ):
        op.drop_table(table, schema="raw")
    for table in ("data_quality_result", "ingest_state", "pipeline_run"):
        op.drop_table(table, schema="ops")
    # dbt's schemas must be empty to drop; downgrading a database dbt has built is not supported.
    for schema in (*DBT_SCHEMAS, *ALEMBIC_SCHEMAS):
        op.execute(f"DROP SCHEMA [{schema}]")
