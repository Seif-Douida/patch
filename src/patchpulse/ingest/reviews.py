"""Page through a game's Steam reviews inside a date window (spec §6.1).

Steam's Web API method `IUserReviewsService/GetAppReviews` (no key needed) takes its parameters as
one `input_json` object, with `date_range_start` and `date_range_end` (unix seconds, both
included), so both the nightly incremental and the progressive backfill ask for exactly the window
they need, newest first. Valve switched off the store's `/appreviews` endpoint on 22 Oct 2026,
which this replaced (it answers HTTP 302 to the store's home page since). The answer sits in
`response`, with no `success` field: an HTTP error is the failure. Paging stops on an empty page,
a repeated cursor (Steam's known looping bug), a review older than the window, or the caller's
deadline.
"""

from __future__ import annotations

import json
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

REVIEWS_URL = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/"
# EUserReviewsAppReviewsFilter and EUserReviewsPurchaseType
# (partner.steamgames.com/doc/webapi/IUserReviewsService).
FILTER_RECENT = 1
PURCHASE_ALL = 1
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
    total: int | None = None  # the window's review count, which Steam reports on the first page

    @property
    def payload(self) -> Any:
        return self.raw.payload


def review_request(
    appid: int,
    start: datetime,
    end: datetime,
    *,
    cursor: str = FIRST_CURSOR,
    per_page: int = PAGE_SIZE,
) -> dict[str, str | int]:
    """GetAppReviews' parameters for reviews created in [start, end], newest first: every
    language and purchase type, off-topic reviews included (the filters Phase 1 chose)."""
    query = {
        "appid": appid,
        "filter": FILTER_RECENT,
        "languages": ["all"],
        "num_per_page": per_page,
        "purchase_type": PURCHASE_ALL,
        "filter_offtopic_activity": False,
        "date_range_start": int(start.timestamp()),
        "date_range_end": int(end.timestamp()),
        "cursor": cursor,
    }
    return {"input_json": json.dumps(query, separators=(",", ":"))}


def review_response(answer: dict[str, Any], appid: int) -> dict[str, Any]:
    """The page inside GetAppReviews' answer; the method has no `success` field."""
    payload = answer.get("response")
    if not isinstance(payload, dict):
        raise SteamRequestError(f"GetAppReviews {appid}: no response object")
    return payload


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
        request = review_request(appid, start, end, cursor=cursor)
        answer, status = http.get_json(REVIEWS_URL, request)
        payload = review_response(answer, appid)

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

        summary = payload.get("query_summary") or {}
        total = summary.get("total_reviews") if cursor == FIRST_CURSOR else None
        raw = RawPage("reviews", appid, request, status, strip_review_page(payload), len(reviews))
        yield ReviewPage(
            raw,
            records,
            next_cursor,
            stop,
            frozenset(drift),
            invalid,
            int(total) if total is not None else None,
        )
        if stop is not None:
            return
        seen.add(next_cursor)
        cursor = next_cursor
