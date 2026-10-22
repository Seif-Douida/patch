"""raw.steam_review_daily and app.patch_effect: the comparison games' daily counts and the stored
effect estimates (Phase 4 plan Task 9, design D4 and D10).

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-22

`steam_review_daily`: one row per comparison game and UTC day, the positive and negative reviews
Steam counted for it (`appreviews` with `num_per_page=0`: no review text). The history up to the
seed's last day is the dbt seed `donor_daily_counts`; the nightly `counts` stage stores the days
after it here.

`app.patch_effect`: one row per bundle, outcome, sample, horizon and method for each run that
(re)computed it. Rows are append-only: a changed fingerprint or EFFECTS_VERSION adds a row under the
new run, and readers take each key's latest. `patch_ids` and `donors` are JSON; `flags` is a
comma-separated list ('' for none).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mssql import DATETIME2, TINYINT

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

STEAM_TIME = DATETIME2(0)  # type: ignore[no-untyped-call]


def upgrade() -> None:
    op.create_table(
        "steam_review_daily",
        sa.Column("appid", sa.Integer, primary_key=True, autoincrement=False),
        sa.Column("day", sa.Date, primary_key=True),
        sa.Column("positive", sa.Integer, nullable=False),
        sa.Column("negative", sa.Integer, nullable=False),
        sa.Column("fetched_run_id", sa.BigInteger, nullable=False),
        sa.Column("fetched_at", STEAM_TIME, nullable=False),
        schema="raw",
    )
    op.execute("CREATE SCHEMA [app]")
    op.create_table(
        "patch_effect",
        sa.Column("bundle_id", sa.Unicode(32), primary_key=True),
        sa.Column("patch_ids", sa.Unicode(400), nullable=False),
        sa.Column("appid", sa.Integer, nullable=False),
        sa.Column("outcome", sa.Unicode(64), primary_key=True),
        sa.Column("sample", sa.Unicode(16), primary_key=True),
        sa.Column("horizon", TINYINT, primary_key=True, autoincrement=False),
        sa.Column("method", sa.Unicode(8), primary_key=True),
        sa.Column("run_id", sa.BigInteger, primary_key=True, autoincrement=False),
        sa.Column("effect", sa.REAL),
        sa.Column("lo90", sa.REAL),
        sa.Column("hi90", sa.REAL),
        sa.Column("p_value", sa.REAL),
        sa.Column("pre_fit_rmse", sa.REAL),
        sa.Column("donors", sa.UnicodeText, nullable=False),
        sa.Column("flags", sa.Unicode(200), nullable=False),
        sa.Column("reason", sa.Unicode(200)),
        sa.Column("fingerprint", sa.CHAR(64), nullable=False),
        sa.Column("version", sa.Unicode(8), nullable=False),
        sa.Column("computed_at", STEAM_TIME, nullable=False),
        schema="app",
    )
    op.create_index(
        "ix_patch_effect_appid_bundle", "patch_effect", ["appid", "bundle_id"], schema="app"
    )


def downgrade() -> None:
    op.drop_index("ix_patch_effect_appid_bundle", table_name="patch_effect", schema="app")
    op.drop_table("patch_effect", schema="app")
    op.execute("DROP SCHEMA [app]")
    op.drop_table("steam_review_daily", schema="raw")
