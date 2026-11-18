"""raw.teacher_label and ops.eval_metric (Phase 7 plan Task 1, design D3 and D7).

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-17

`raw.teacher_label` holds the teacher's aspect labels by review version: the silver set's (source
`silver`) and the nightly spot-check's (source `spotcheck`). It has no text column; the text stays
in raw.steam_review. `ops.eval_metric` holds every number Phase 7 measures: the weekly drift
statistics, the teacher-student agreement, the online agent evals and the retrain requests and
decisions, one row per metric and scope (`week`, `night`, `report:<id>`, `drill`).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mssql import DATETIME2

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None

STEAM_TIME = DATETIME2(0)  # type: ignore[no-untyped-call]
# Taxonomy v1, as migration 0002 lists it.
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
        "teacher_label",
        sa.Column("recommendation_id", sa.BigInteger, primary_key=True, autoincrement=False),
        sa.Column("timestamp_updated", STEAM_TIME, primary_key=True),
        sa.Column("model", sa.Unicode(64), primary_key=True),
        sa.Column("prompt_hash", sa.Unicode(64), primary_key=True),
        sa.Column("source", sa.Unicode(16), nullable=False),
        sa.Column("run_id", sa.BigInteger, nullable=True),
        sa.Column("created_at", STEAM_TIME, nullable=False),
        *(sa.Column(f"l_{code}", sa.Boolean, nullable=False) for code in ASPECTS),
        schema="raw",
    )
    op.create_table(
        "eval_metric",
        sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column("run_id", sa.BigInteger, nullable=True),
        sa.Column("family", sa.Unicode(32), nullable=False),
        sa.Column("metric", sa.Unicode(128), nullable=False),
        sa.Column("scope", sa.Unicode(64), nullable=False),
        sa.Column("window_start", sa.Date, nullable=True),
        sa.Column("window_end", sa.Date, nullable=True),
        sa.Column("value", sa.Float, nullable=True),
        sa.Column("lo", sa.Float, nullable=True),
        sa.Column("hi", sa.Float, nullable=True),
        sa.Column("n", sa.Integer, nullable=True),
        sa.Column("created_at", STEAM_TIME, nullable=False),
        schema="ops",
    )
    op.create_index(
        "ix_eval_metric_family_scope_end",
        "eval_metric",
        ["family", "scope", "window_end"],
        schema="ops",
    )


def downgrade() -> None:
    op.drop_index("ix_eval_metric_family_scope_end", table_name="eval_metric", schema="ops")
    op.drop_table("eval_metric", schema="ops")
    op.drop_table("teacher_label", schema="raw")
