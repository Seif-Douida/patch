"""Comparison games' daily review counts (Phase 4, design D4).

`appreviews` with `num_per_page=0` returns only `query_summary`, whose `total_positive` and
`total_negative` count the reviews created in the requested window: one request per UTC day, no
review text. The filters are the ingest's (reviews.py), so the counts match ours: on 21 Oct 2026
three tracked games' 750 days agreed with the stored reviews within 0.06% (Steam lower: reviews
deleted after we stored them). A day with no reviews comes back as zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from patchpulse.ingest.http import SteamError, SteamHttp

REVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"


@dataclass(frozen=True)
class DayCounts:
    appid: int
    day: date
    positive: int
    negative: int


def _summary(http: SteamHttp, appid: int, start: datetime, end: datetime) -> dict[str, Any]:
    params: dict[str, str | int] = {
        "json": 1,
        "filter": "recent",
        "language": "all",
        "num_per_page": 0,
        "purchase_type": "all",
        "filter_offtopic_activity": 0,
        "start_date": int(start.timestamp()),
        "end_date": int(end.timestamp()),
        "date_range_type": "include",
        "cursor": "*",
    }
    payload, _ = http.get_json(REVIEWS_URL.format(appid=appid), params)
    if payload.get("success") != 1:
        raise SteamError(f"appreviews {appid}: success {payload.get('success')!r}")
    summary = payload.get("query_summary")
    if not isinstance(summary, dict) or not {"total_positive", "total_negative"} <= summary.keys():
        raise SteamError(f"appreviews {appid}: no review counts in the summary")
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
