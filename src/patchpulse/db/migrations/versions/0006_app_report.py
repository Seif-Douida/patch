"""app.report: the agent's stored reports (Phase 5 plan Task 11, design D6 and D10).

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-25

One row per patch bundle, horizon and run that wrote its report. Rows are append-only: a report is
written again only when its evidence fingerprint changes (its window's core tool results, its
estimate rows, the rules' settings or the prompt version) or when the stored one is basic, and
readers take each (bundle, horizon)'s latest row. `report_json` is the stored copy (quote spans,
review ids and withheld text included: SQL is where review text lives); `public_json` is what the
site and the API show. `model_ids` is JSON.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mssql import DATETIME2, TINYINT

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

STEAM_TIME = DATETIME2(0)  # type: ignore[no-untyped-call]


def upgrade() -> None:
    op.create_table(
        "report",
        sa.Column("bundle_id", sa.Unicode(32), primary_key=True),
        sa.Column("horizon", TINYINT, primary_key=True, autoincrement=False),
        sa.Column("run_id", sa.BigInteger, primary_key=True, autoincrement=False),
        sa.Column("report_id", sa.Unicode(64), nullable=False),
        sa.Column("scope", sa.Unicode(16), nullable=False),
        sa.Column("data_version", sa.BigInteger, nullable=False),
        sa.Column("fingerprint", sa.CHAR(64), nullable=False),
        sa.Column("prompt_version", sa.Unicode(16), nullable=False),
        sa.Column("model_ids", sa.Unicode(400), nullable=False),
        sa.Column("basic", sa.Boolean, nullable=False),
        sa.Column("report_json", sa.UnicodeText, nullable=False),
        sa.Column("public_json", sa.UnicodeText, nullable=False),
        sa.Column("created_at", STEAM_TIME, nullable=False),
        schema="app",
    )


def downgrade() -> None:
    op.drop_table("report", schema="app")
