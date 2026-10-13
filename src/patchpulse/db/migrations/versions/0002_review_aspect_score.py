"""raw.review_aspect_score: every aspect score a model has given a review version (plan Task 15).

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-13

One row per (review version, model version), with a probability (`p_<aspect>`) and a label
(`l_<aspect>`) for each aspect of taxonomy v1. Scores are never overwritten: an edited review or a
new champion adds rows, and dbt reads each review's latest version scored by the current model.

`ix_steam_review_created` lets the scoring stage walk reviews newest first in pages (keyset
paging on timestamp_created, recommendation_id) instead of sorting every review for each page.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mssql import DATETIME2

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

STEAM_TIME = DATETIME2(0)  # type: ignore[no-untyped-call]
PIPELINE_TIME = DATETIME2(3)  # type: ignore[no-untyped-call]
# Taxonomy v1, in its fixed order (patchpulse.labeling.taxonomy). A new taxonomy is a new migration.
ASPECTS = (
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


def upgrade() -> None:
    op.create_table(
        "review_aspect_score",
        sa.Column("recommendation_id", sa.BigInteger, primary_key=True, autoincrement=False),
        sa.Column("timestamp_updated", STEAM_TIME, primary_key=True),
        sa.Column("model_version", sa.Unicode(64), primary_key=True),
        sa.Column("model_name", sa.Unicode(64), nullable=False),
        sa.Column("scored_run_id", sa.BigInteger, nullable=False),
        sa.Column(
            "scored_at", PIPELINE_TIME, nullable=False, server_default=sa.text("SYSUTCDATETIME()")
        ),
        *(sa.Column(f"p_{code}", sa.REAL, nullable=False) for code in ASPECTS),
        *(sa.Column(f"l_{code}", sa.Boolean, nullable=False) for code in ASPECTS),
        schema="raw",
    )
    op.create_index(
        "ix_steam_review_created",
        "steam_review",
        ["timestamp_created", "recommendation_id"],
        schema="raw",
    )


def downgrade() -> None:
    op.drop_index("ix_steam_review_created", table_name="steam_review", schema="raw")
    op.drop_table("review_aspect_score", schema="raw")
