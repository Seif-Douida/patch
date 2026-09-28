"""Which review windows each nightly run fetches (spec §6.1; plan Decision 3).

Every game has a covered range [covered_from, covered_to]. Each night re-reads the last 3 days up
to now (to pick up edits) and spends the leftover time budget extending the range backwards
towards `backfill_start`, newest first, resuming where the previous night stopped.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from patchpulse.ingest.coverage import REREAD, Coverage, Windows, advance, plan_windows

NOW = datetime(2026, 9, 28, 3, 0, tzinfo=UTC)
START = date(2024, 10, 1)
START_AT = datetime(2024, 10, 1, tzinfo=UTC)


def test_new_game_reads_the_last_three_days_and_backfills_the_rest() -> None:
    windows = plan_windows(None, backfill_start=START, now=NOW)

    assert windows.incremental == (NOW - REREAD, NOW)
    assert windows.backfill == (START_AT, NOW - REREAD)


def test_incremental_window_rereads_three_days() -> None:
    last_night = NOW - timedelta(days=1)
    coverage = Coverage(covered_from=START_AT, covered_to=last_night)

    windows = plan_windows(coverage, backfill_start=START, now=NOW)

    assert windows.incremental == (last_night - timedelta(days=3), NOW)
    assert windows.backfill is None  # fully backfilled already


def test_backfill_window_stops_at_backfill_start() -> None:
    coverage = Coverage(covered_from=datetime(2025, 6, 1, tzinfo=UTC), covered_to=NOW)

    windows = plan_windows(coverage, backfill_start=START, now=NOW)

    assert windows.backfill == (START_AT, datetime(2025, 6, 1, tzinfo=UTC))


def test_backfill_resumes_from_the_oldest_fetched_review() -> None:
    windows = plan_windows(None, backfill_start=START, now=NOW)
    reached = datetime(2026, 3, 14, 9, 30, tzinfo=UTC)  # the deadline hit here

    after = advance(None, windows, backfill_reached=reached)

    assert after == Coverage(covered_from=reached, covered_to=NOW)
    assert plan_windows(after, backfill_start=START, now=NOW).backfill == (START_AT, reached)


def test_completed_backfill_covers_back_to_backfill_start() -> None:
    windows = plan_windows(None, backfill_start=START, now=NOW)

    after = advance(None, windows, backfill_reached=START_AT)

    assert after.covered_from == START_AT


def test_skipped_backfill_keeps_the_incremental_coverage() -> None:
    windows = plan_windows(None, backfill_start=START, now=NOW)

    after = advance(None, windows, backfill_reached=None)

    assert after == Coverage(covered_from=NOW - REREAD, covered_to=NOW)


def test_backfill_never_reaches_before_backfill_start() -> None:
    windows = Windows(incremental=(NOW - REREAD, NOW), backfill=(START_AT, NOW - REREAD))

    after = advance(None, windows, backfill_reached=START_AT - timedelta(days=30))

    assert after.covered_from == START_AT
