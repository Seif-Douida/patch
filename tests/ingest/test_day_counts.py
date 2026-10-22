"""Steam's per-day review counts and app details for comparison games (Phase 4, design D3-D4)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from urllib.parse import parse_qs, urlsplit

import pytest
from fake_steam import FakeSteam

from patchpulse.ingest.appdetails import AppDetails, fetch_app_details, parse_release_date
from patchpulse.ingest.day_counts import DayCounts, fetch_day_counts, fetch_window_total
from patchpulse.ingest.http import SteamError

APP = 1363080
DAY = date(2026, 10, 1)


def at(day: int, hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, second, tzinfo=UTC)


def test_day_counts_from_the_summary() -> None:
    steam = FakeSteam()
    steam.add_review(APP, 1, at(1, 0, 0, 5))
    steam.add_review(APP, 2, at(1, 10))
    steam.add_review(APP, 3, at(1, 23, 59, 59), voted_up=False)
    steam.add_review(APP, 4, at(2, 0))  # the next day
    steam.add_review(APP, 5, datetime(2026, 9, 30, 23, 59, 59, tzinfo=UTC))  # the day before

    counts = fetch_day_counts(steam.http(), APP, DAY)

    assert counts == DayCounts(appid=APP, day=DAY, positive=2, negative=1)
    (request,) = steam.requests
    params = {k: v[0] for k, v in parse_qs(urlsplit(str(request.url)).query).items()}
    assert params["num_per_page"] == "0"
    assert (params["language"], params["purchase_type"], params["filter_offtopic_activity"]) == (
        "all",
        "all",
        "0",
    )
    assert params["date_range_type"] == "include"
    start = int(datetime(2026, 10, 1, tzinfo=UTC).timestamp())
    assert (int(params["start_date"]), int(params["end_date"])) == (start, start + 86_399)


def test_zero_review_day() -> None:
    # Steam answers success 1 with zero counts for a day with no reviews (checked 21 Oct 2026).
    assert fetch_day_counts(FakeSteam().http(), APP, DAY) == DayCounts(APP, DAY, 0, 0)


def test_missing_summary_or_failure_raises() -> None:
    steam = FakeSteam()
    steam.day_counts_broken[APP] = "no summary"
    with pytest.raises(SteamError, match="summary"):
        fetch_day_counts(steam.http(), APP, DAY)
    steam.day_counts_broken[APP] = "unsuccessful"
    with pytest.raises(SteamError, match="success"):
        fetch_day_counts(steam.http(), APP, DAY)


def test_window_total() -> None:
    steam = FakeSteam()
    for rid, day in enumerate((29, 30), start=1):
        steam.add_review(APP, rid, datetime(2026, 9, day, 12, tzinfo=UTC))
    steam.add_review(APP, 9, at(1, 12))
    steam.add_review(APP, 10, at(2, 12))  # after the window

    assert fetch_window_total(steam.http(), APP, date(2026, 9, 29), DAY) == 3


def test_app_details_parsed() -> None:
    steam = FakeSteam()
    steam.details[APP] = {
        "name": "Manor Lords",
        "genres": [
            {"id": "28", "description": "Simulation"},
            {"id": "2", "description": "Strategy"},
            {"id": "70", "description": "Early Access"},
        ],
        "release_date": {"coming_soon": False, "date": "Apr 26, 2024"},
    }

    assert fetch_app_details(steam.http(), APP) == AppDetails(
        appid=APP,
        name="Manor Lords",
        genres=("Simulation", "Strategy", "Early Access"),
        release_date=date(2024, 4, 26),
    )


def test_release_date_formats() -> None:
    assert parse_release_date({"coming_soon": False, "date": "Apr 26, 2024"}) == date(2024, 4, 26)
    assert parse_release_date({"coming_soon": False, "date": "26 Apr, 2024"}) == date(2024, 4, 26)
    assert parse_release_date({"coming_soon": False, "date": "April 2024"}) is None
    assert parse_release_date({"coming_soon": True, "date": "Apr 26, 2027"}) is None
    assert parse_release_date(None) is None


def test_unknown_app_raises() -> None:
    with pytest.raises(SteamError, match="appdetails"):
        fetch_app_details(FakeSteam().http(), APP)
