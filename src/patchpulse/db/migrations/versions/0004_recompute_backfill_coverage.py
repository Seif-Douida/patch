"""Recompute ops.ingest_state.covered_from from the reviews actually stored (a data fix).

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-21

On 20 Oct 2026 Steam was found to have ended three games' backfill windows with empty pages long
before the counts it reported for them (Helldivers 2: 282,709 reviews between 1 Oct 2024 and
15 Sep 2026; Baldur's Gate 3: 139,380; Clair Obscur: 272,094), and the backfill had recorded each
window as covered. Coverage now restarts at each game's oldest stored review wherever it claims
more than a day of history that raw.steam_review doesn't hold; the backfill, which now checks
Steam's count before calling a window done, fetches what's older. A game fully backfilled keeps
its coverage (its oldest review is within a day of it); one with no stored review is untouched.

Downgrade leaves the data as it is: the old values were the error.
"""

from __future__ import annotations

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE s
        SET covered_from = r.oldest
        FROM ops.ingest_state AS s
        INNER JOIN (
            SELECT appid, MIN(timestamp_created) AS oldest
            FROM raw.steam_review
            GROUP BY appid
        ) AS r
            ON r.appid = s.appid
        WHERE r.oldest > DATEADD(day, 1, s.covered_from)
        """
    )


def downgrade() -> None:
    pass
