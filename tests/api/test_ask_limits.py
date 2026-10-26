"""Ask's limits (Phase 5 plan Task 16, design D9) on the Qdrant container: the counters live in a
small collection, so they survive the replica scaling to zero; visitors are only ever stored as
hashes made with a daily salt."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest
from fake_clock import FakeClock
from qdrant_client import QdrantClient, models

from patchpulse.agent.tracing import MonthUnits
from patchpulse.api.limits import (
    RESTING,
    Allowed,
    AskLimits,
    LimitStore,
    MemoryLimitStore,
    QdrantLimitStore,
    Refused,
    visitor_hash,
)

pytestmark = pytest.mark.qdrant

SECRET = b"test-secret"
VISITOR = "203.0.113.7"  # documentation addresses (RFC 5737)
NOON = datetime(2026, 10, 25, 12, tzinfo=UTC)


def limits(client: QdrantClient, name: str, clock: FakeClock) -> AskLimits:
    return AskLimits(
        QdrantLimitStore(client, name), secret=SECRET, per_visitor_hour=3, daily_cap=4, clock=clock
    )


def test_qdrant_limit_store_counts_survive_a_new_process(
    qdrant: QdrantClient, new_collection: Callable[[], str]
) -> None:
    name, clock = new_collection(), FakeClock(NOON)
    first = limits(qdrant, name, clock)
    assert all(isinstance(first.check(VISITOR), Allowed) for _ in range(3))

    # A new replica after scale-to-zero: nothing in memory, the same collection.
    again = limits(qdrant, name, clock)

    refused = again.check(VISITOR)
    assert isinstance(refused, Refused)
    assert refused.reason == "You've asked 3 questions this hour; try again in 60 minutes."
    assert refused.retry_after == 3_600
    assert isinstance(again.check("198.51.100.9"), Allowed)  # the day's 4th
    resting = again.check("192.0.2.1")
    assert resting == Refused(RESTING, 12 * 3_600)  # until midnight UTC


def test_visitors_are_never_stored_raw(
    qdrant: QdrantClient, new_collection: Callable[[], str]
) -> None:
    name, clock = new_collection(), FakeClock(NOON)
    ask = limits(qdrant, name, clock)
    ask.check(VISITOR)
    ask.check(VISITOR)

    points, _ = qdrant.scroll(name, limit=100, with_payload=True)
    stored = json.dumps([[str(p.id), p.payload] for p in points])
    assert VISITOR not in stored
    assert "203.0.113" not in stored
    today = visitor_hash(SECRET, "2026-10-25", VISITOR)
    assert today in stored

    # The next day: a new salt, so the same visitor can't be followed across days, and the
    # previous day's counters are gone.
    clock.now = NOON + timedelta(days=1)
    ask.check(VISITOR)
    points, _ = qdrant.scroll(name, limit=100, with_payload=True)
    stored = json.dumps([[str(p.id), p.payload] for p in points])
    tomorrow = visitor_hash(SECRET, "2026-10-26", VISITOR)
    assert tomorrow != today
    assert tomorrow in stored
    assert today not in stored
    assert "2026-10-25" not in stored


@pytest.mark.parametrize("kind", ["qdrant", "memory"])
def test_the_months_trace_units_survive_a_new_day(
    qdrant: QdrantClient, new_collection: Callable[[], str], kind: str
) -> None:
    # Task 18: the month's Langfuse units share the store with Ask's daily counters; a new day
    # forgets the visitors and the day's total, never the month's units.
    clock = FakeClock(NOON)
    store: LimitStore = (
        QdrantLimitStore(qdrant, new_collection()) if kind == "qdrant" else MemoryLimitStore()
    )
    ask = AskLimits(store, secret=SECRET, per_visitor_hour=3, daily_cap=4, clock=clock)
    units = MonthUnits(store, clock=clock)
    ask.check(VISITOR)
    units.add(40)

    clock.now = NOON + timedelta(days=1)
    ask.check(VISITOR)
    units.add(2)

    assert units.used() == 42
    assert store.get(f"visitor|2026-10-25|{visitor_hash(SECRET, '2026-10-25', VISITOR)}") is None
    assert store.get("day|2026-10-25") is None


def test_the_daily_cleanup_runs_in_the_clusters_strict_mode(
    qdrant: QdrantClient, new_collection: Callable[[], str]
) -> None:
    # The cluster refuses filters on unindexed fields (strict mode, read on 26 Oct): the store
    # indexes `day`, so a new day's cleanup runs there and forgets the old day.
    name, clock = new_collection(), FakeClock(NOON)
    strict = models.StrictModeConfig(
        enabled=True, unindexed_filtering_retrieve=False, unindexed_filtering_update=False
    )
    qdrant.create_collection(name, vectors_config={}, strict_mode_config=strict)
    ask = limits(qdrant, name, clock)
    ask.check(VISITOR)

    clock.now = NOON + timedelta(days=1)

    assert isinstance(ask.check(VISITOR), Allowed)
    assert QdrantLimitStore(qdrant, name).get("day|2026-10-25") is None


class FailingCleanup(MemoryLimitStore):
    def __init__(self) -> None:
        super().__init__()
        self.cleanups = 0

    def forget_other_days(self, day: str) -> None:
        self.cleanups += 1
        raise RuntimeError("refused")


def test_a_failed_cleanup_never_stops_ask(caplog: pytest.LogCaptureFixture) -> None:
    store, clock = FailingCleanup(), FakeClock(NOON)
    ask = AskLimits(store, secret=SECRET, per_visitor_hour=3, daily_cap=4, clock=clock)

    answers = [ask.check(VISITOR), ask.check(VISITOR)]

    assert all(isinstance(a, Allowed) for a in answers)
    assert store.cleanups == 1  # tried again the next day, not on every question
    assert "cleanup" in caplog.text
