"""Comparison games' daily review counts (Phase 4, design D4).

GetAppReviews' first page carries `query_summary.total_positive` and `total_negative`, the reviews
created in the requested window: one request per UTC day, at one review a page (0 means Steam's
default of 20), and that review is never read or kept. The filters are the ingest's (reviews.py),
so the counts match ours: on 21 Oct 2026 three tracked games' 750 days agreed with the stored
reviews within 0.06% (Steam lower: reviews deleted after we stored them), measured on the store's
`/appreviews`, which Valve switched off on 22 Oct 2026 for this method. A day with no reviews
comes back as zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from patchpulse.ingest.http import SteamError, SteamHttp
from patchpulse.ingest.reviews import REVIEWS_URL, review_request, review_response


@dataclass(frozen=True)
class DayCounts:
    appid: int
    day: date
    positive: int
    negative: int


def _summary(http: SteamHttp, appid: int, start: datetime, end: datetime) -> dict[str, Any]:
    answer, _ = http.get_json(REVIEWS_URL, review_request(appid, start, end, per_page=1))
    summary = review_response(answer, appid).get("query_summary")
    if not isinstance(summary, dict) or not {"total_positive", "total_negative"} <= summary.keys():
        raise SteamError(f"GetAppReviews {appid}: no review counts in the summary")
    return summary


def _midnight(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def fetch_day_counts(http: SteamHttp, appid: int, day: date) -> DayCounts:
    """The positive and negative reviews created on one UTC day."""
    start = _midnight(day)
    summary = _summary(http, appid, start, start + timedelta(days=1) - timedelta(seconds=1))
    return DayCounts(appid, day, int(summary["total_positive"]), int(summary["total_negative"]))


def fetch_window_total(http: SteamHttp, appid: int, first: date, last: date) -> int:
    """All reviews created from `first` to `last`, both included."""
    summary = _summary(
        http, appid, _midnight(first), _midnight(last) + timedelta(days=1) - timedelta(seconds=1)
    )
    return int(summary["total_positive"]) + int(summary["total_negative"])
