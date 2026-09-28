"""Which review windows a nightly run fetches for each game (plan Decision 3).

Each game has a covered range [covered_from, covered_to], stored in ops.ingest_state. A run:

* re-reads [covered_to - 3 days, now] to pick up new and edited reviews (the incremental window);
* spends the leftover time budget extending the range backwards to `backfill_start`, newest
  first (the backfill window), and records how far it got so the next night resumes there.

Coverage is only written after a successful run, so a failed night leaves no gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

REREAD = timedelta(days=3)

Window = tuple[datetime, datetime]


@dataclass(frozen=True)
class Coverage:
    covered_from: datetime
    covered_to: datetime


@dataclass(frozen=True)
class Windows:
    incremental: Window
    backfill: Window | None  # None when history is already filled back to backfill_start


def start_of_day(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def plan_windows(
    coverage: Coverage | None, *, backfill_start: date, now: datetime, reread: timedelta = REREAD
) -> Windows:
    floor = start_of_day(backfill_start)
    if coverage is None:
        incremental = (now - reread, now)
        oldest = now - reread
    else:
        incremental = (coverage.covered_to - reread, now)
        oldest = coverage.covered_from
    return Windows(incremental, (floor, oldest) if oldest > floor else None)


def advance(
    previous: Coverage | None, windows: Windows, *, backfill_reached: datetime | None
) -> Coverage:
    """Coverage after a successful run.

    `backfill_reached` is the oldest point the backfill covered: the window's start when it
    finished, the oldest review fetched when the deadline stopped it, or None if it didn't run.
    """
    covered_from = windows.incremental[0]
    if previous is not None:
        covered_from = min(previous.covered_from, covered_from)
    if windows.backfill is not None and backfill_reached is not None:
        covered_from = min(covered_from, max(backfill_reached, windows.backfill[0]))
    return Coverage(covered_from, windows.incremental[1])
