"""ops.qdrant_point and ops.qdrant_reconciliation: the nightly index stage's record of Qdrant
(Phase 3 plan Task 14, design D5).

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-20

`qdrant_point` is the mirror: one row per point the stage has written to a collection, with the
review version and field fingerprint it wrote, added only after Qdrant confirmed the write. Each
night's plan compares core.review_index with it. `synced_run_id` is null for points read back from
the collection: the laptop's upload wrote those, not a run.

`qdrant_reconciliation`: per run, collection, game and day (yyyymmdd, as the points' `day`), the
count core.review_index expects and the count Qdrant holds.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mssql import DATETIME2

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

STEAM_TIME = DATETIME2(0)  # type: ignore[no-untyped-call]


def upgrade() -> None:
    op.create_table(
        "qdrant_point",
        sa.Column("collection", sa.Unicode(64), primary_key=True),
        sa.Column("review_id", sa.BigInteger, primary_key=True, autoincrement=False),
        sa.Column("version", STEAM_TIME, nullable=False),
        sa.Column("fingerprint", sa.CHAR(64), nullable=False),
        sa.Column("synced_run_id", sa.BigInteger),
        schema="ops",
    )
    op.create_table(
        "qdrant_reconciliation",
        sa.Column("run_id", sa.BigInteger, primary_key=True, autoincrement=False),
        sa.Column("collection", sa.Unicode(64), primary_key=True),
        sa.Column("appid", sa.Integer, primary_key=True, autoincrement=False),
        sa.Column("day", sa.Integer, primary_key=True, autoincrement=False),
        sa.Column("sql_count", sa.Integer, nullable=False),
        sa.Column("qdrant_count", sa.Integer, nullable=False),
        schema="ops",
    )


def downgrade() -> None:
    op.drop_table("qdrant_reconciliation", schema="ops")
    op.drop_table("qdrant_point", schema="ops")
