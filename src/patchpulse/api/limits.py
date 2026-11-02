"""Ask's limits (Phase 5, design D9: Seif's option B, 25 Oct): a few questions per visitor an
hour and a daily cap, so Ask stays inside the free compute after the nightly job and the pings.

The counters live in a small Qdrant collection (`ask_limits`: free, already reachable, and it
survives the replica scaling to zero), one point per key with its counts as the payload. A visitor
is an HMAC of their address with a key derived from a server secret and the day: never stored
raw, and not linkable from one day to the next. The previous days' points are deleted on the first
check of a new day, best-effort (a failure is logged and tried again the next day); points without
a day (the month's Langfuse units, agent/tracing.py) stay. The cluster runs in strict mode, which
refuses filters on fields without an index, so the store indexes `day`. Only
questions that will run the agent are counted (the API checks the limits after its free answers:
refusals it can make without a model, and cached answers).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from qdrant_client import QdrantClient, models

log = logging.getLogger("patchpulse.api")

COLLECTION = "ask_limits"
HOUR_S = 3_600
RESTING = "Ask is resting until tomorrow; the nightly reports are still here."
_IDS = uuid.UUID("5d7a3f52-7a35-4c43-9a1e-3c0f1a6b2e91")  # the namespace of the points' ids


def _utc_now() -> datetime:
    return datetime.now(UTC)


def visitor_hash(secret: bytes, day: str, address: str) -> str:
    """The visitor's id for `day`: an HMAC of their address, keyed by the secret and the day."""
    daily = hmac.new(secret, f"ask-visitors|{day}".encode(), hashlib.sha256).digest()
    return hmac.new(daily, address.encode(), hashlib.sha256).hexdigest()


class LimitStore(Protocol):
    def get(self, key: str) -> dict[str, Any] | None: ...

    def put(self, key: str, payload: dict[str, Any]) -> None: ...

    def forget_other_days(self, day: str) -> None: ...


class MemoryLimitStore:
    """For tests and local runs without Qdrant: counts live as long as the process."""

    def __init__(self) -> None:
        self._points: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        return self._points.get(key)

    def put(self, key: str, payload: dict[str, Any]) -> None:
        self._points[key] = dict(payload)

    def forget_other_days(self, day: str) -> None:
        self._points = {k: p for k, p in self._points.items() if p.get("day") in (None, day)}


class QdrantLimitStore:
    """Points without vectors in `collection`, created on first use."""

    def __init__(self, client: QdrantClient, collection: str = COLLECTION) -> None:
        self.client = client
        self.collection = collection
        self._ready = False

    def _ensure(self) -> None:
        if self._ready:
            return
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection, vectors_config={})
        if "day" not in (self.client.get_collection(self.collection).payload_schema or {}):
            try:  # strict mode filters only on indexed fields; a key may not be allowed to index
                self.client.create_payload_index(
                    self.collection, "day", field_schema=models.PayloadSchemaType.KEYWORD, wait=True
                )
            except Exception as error:
                log.warning("ask_limits: `day` not indexed (%s)", type(error).__name__)
        self._ready = True

    @staticmethod
    def _id(key: str) -> str:
        return str(uuid.uuid5(_IDS, key))

    def get(self, key: str) -> dict[str, Any] | None:
        self._ensure()
        found = self.client.retrieve(self.collection, ids=[self._id(key)], with_payload=True)
        return dict(found[0].payload or {}) if found else None

    def put(self, key: str, payload: dict[str, Any]) -> None:
        self._ensure()
        point = models.PointStruct(id=self._id(key), vector={}, payload=payload)
        self.client.upsert(self.collection, points=[point])

    def forget_other_days(self, day: str) -> None:
        self._ensure()
        other_days = models.Filter(
            must_not=[
                models.FieldCondition(key="day", match=models.MatchValue(value=day)),
                models.IsEmptyCondition(is_empty=models.PayloadField(key="day")),
            ]
        )
        self.client.delete(
            self.collection, points_selector=models.FilterSelector(filter=other_days)
        )


@dataclass(frozen=True)
class Allowed:
    left: int  # the visitor's questions left this hour, this one counted


@dataclass(frozen=True)
class Refused:
    reason: str
    retry_after: int  # seconds


class AskLimits:
    def __init__(
        self,
        store: LimitStore,
        *,
        secret: bytes,
        per_visitor_hour: int = 3,
        daily_cap: int,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.store = store
        self.secret = secret
        self.per_visitor_hour = per_visitor_hour
        self.daily_cap = daily_cap
        self.clock = clock
        self._lock = threading.Lock()
        self._current_day: str | None = None

    def check(self, address: str) -> Allowed | Refused:
        """Count a question from `address`, or say why it can't be asked now."""
        with self._lock:
            now = self.clock().astimezone(UTC)
            day = now.date().isoformat()
            if day != self._current_day:
                try:
                    self.store.forget_other_days(day)
                except Exception as error:  # old counts linger a day; Ask goes on
                    log.warning("Ask's daily cleanup failed (%s); tried again tomorrow",
                                type(error).__name__)  # fmt: skip
                self._current_day = day
            total = self.store.get(f"day|{day}") or {"day": day, "count": 0}
            if total["count"] >= self.daily_cap:
                midnight = datetime.combine(
                    now.date() + timedelta(days=1), datetime.min.time(), UTC
                )
                return Refused(RESTING, math.ceil((midnight - now).total_seconds()))
            visitor = visitor_hash(self.secret, day, address)
            key = f"visitor|{day}|{visitor}"
            stamp = now.timestamp()
            asked = [t for t in (self.store.get(key) or {}).get("asked", []) if t > stamp - HOUR_S]
            if len(asked) >= self.per_visitor_hour:
                wait = math.ceil(min(asked) + HOUR_S - stamp)
                minutes = max(1, math.ceil(wait / 60))
                return Refused(
                    f"You've asked {len(asked)} questions this hour; try again in {minutes} "
                    "minutes.",
                    wait,
                )
            self.store.put(key, {"day": day, "visitor": visitor, "asked": [*asked, stamp]})
            self.store.put(f"day|{day}", {"day": day, "count": total["count"] + 1})
            return Allowed(left=self.per_visitor_hour - len(asked) - 1)

    def refund(self, address: str) -> None:
        """Give back the last question counted for `address` today: its run failed."""
        with self._lock:
            day = self.clock().astimezone(UTC).date().isoformat()
            key = f"visitor|{day}|{visitor_hash(self.secret, day, address)}"
            point = self.store.get(key)
            if point and point.get("asked"):
                self.store.put(key, {**point, "asked": point["asked"][:-1]})
            total = self.store.get(f"day|{day}")
            if total and total.get("count", 0) > 0:
                self.store.put(f"day|{day}", {**total, "count": total["count"] - 1})
