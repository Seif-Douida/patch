"""Page through a game's Steam reviews inside a date window (spec §6.1).

Steam's `appreviews` accepts `start_date`, `end_date` and `date_range_type=include` (verified on
2026-09-28), so both the nightly incremental and the progressive backfill ask for exactly the
window they need, newest first. Paging stops on an empty page, a repeated cursor (Steam's known
looping bug), a review older than the window, or the caller's deadline.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import ValidationError

from patchpulse.ingest.http import SteamHttp, SteamRequestError
from patchpulse.ingest.models import RawPage, ReviewRecord, schema_drift
from patchpulse.ingest.privacy import strip_review_page

REVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"
PAGE_SIZE = 100
FIRST_CURSOR = "*"


class StopReason(StrEnum):
    EMPTY = "empty"
    CURSOR_REPEAT = "cursor_repeat"
    WINDOW_PASSED = "window_passed"
    DEADLINE = "deadline"


@dataclass(frozen=True)
class ReviewPage:
    raw: RawPage
    records: list[ReviewRecord]
    next_cursor: str
    next_stop: StopReason | None  # why paging ends after this page; None if it continues
    drift: frozenset[str]
    invalid: int

    @property
    def payload(self) -> Any:
        return self.raw.payload


def fetch_review_pages(
    http: SteamHttp,
    appid: int,
    *,
    start: datetime,
    end: datetime,
    salt: str,
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Iterator[ReviewPage]:
    """Yield pages of reviews created in [start, end], newest first."""
    seen = {FIRST_CURSOR}
    cursor = FIRST_CURSOR
    while True:
        if deadline is not None and clock() >= deadline:
            return
        request: dict[str, str | int] = {
            "json": 1,
            "filter": "recent",
            "language": "all",
            "num_per_page": PAGE_SIZE,
            "purchase_type": "all",
            "filter_offtopic_activity": 0,
            "start_date": int(start.timestamp()),
            "end_date": int(end.timestamp()),
            "date_range_type": "include",
            "cursor": cursor,
        }
        payload, status = http.get_json(REVIEWS_URL.format(appid=appid), request)
        if payload.get("success") != 1:
            raise SteamRequestError(f"appreviews {appid}: success={payload.get('success')!r}")

        reviews = payload.get("reviews") or []
        records: list[ReviewRecord] = []
        drift: set[str] = set()
        invalid = 0
        window_passed = False
        for review in reviews:
            drift |= schema_drift(review)
            try:
                record = ReviewRecord.from_steam(review, appid=appid, salt=salt)
            except (ValidationError, KeyError, TypeError, ValueError):
                invalid += 1
                continue
            if record.timestamp_created < start:
                window_passed = True
                continue
            records.append(record)

        next_cursor = str(payload.get("cursor") or "")
        stop: StopReason | None
        if not reviews:
            stop = StopReason.EMPTY
        elif window_passed:
            stop = StopReason.WINDOW_PASSED
        elif not next_cursor or next_cursor in seen:
            stop = StopReason.CURSOR_REPEAT
        elif deadline is not None and clock() >= deadline:
            stop = StopReason.DEADLINE
        else:
            stop = None

        raw = RawPage("reviews", appid, request, status, strip_review_page(payload), len(reviews))
        yield ReviewPage(raw, records, next_cursor, stop, frozenset(drift), invalid)
        if stop is not None:
            return
        seen.add(next_cursor)
        cursor = next_cursor
