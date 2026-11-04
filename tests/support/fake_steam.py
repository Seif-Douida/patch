"""An in-process stand-in for Steam's three endpoints, for pipeline tests (no network)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx2

from patchpulse.ingest.http import RateLimiter, RetryPolicy, SteamHttp
from patchpulse.ingest.patches import OFFICIAL_FEED

PAGE = 100


class FakeSteam:
    """Serves reviews by date range and cursor, news and prices; records every request."""

    def __init__(self) -> None:
        self.reviews: dict[int, list[dict[str, Any]]] = {}
        self.news: dict[int, list[dict[str, Any]]] = {}
        self.prices: dict[int, dict[str, Any] | None] = {}
        self.failing: set[int] = set()
        self.rate_limit_after: int | None = None  # answer HTTP 429 once this many reviews requests
        self.review_requests = 0
        self.page_size = PAGE
        # appid -> reviews served from a window before Steam answers empty pages (20 Oct 2026: it
        # ended three games' history windows long before the counts it reported for them).
        self.cut_short: dict[int, int] = {}
        self.prices_down = False
        self.requests: list[httpx2.Request] = []
        # Phase 4: appdetails without filters (name, genres, release_date) per game, and per-day
        # review requests to break: appid -> "no response" (no `response` object) or "no summary"
        # (no counts on the first page).
        self.details: dict[int, dict[str, Any]] = {}
        self.day_counts_broken: dict[int, str] = {}
        # The patch-history fix (4 Nov 2026): a news request with `enddate` answers this HTTP
        # status (503 down, 429 rate-limited), or `enddate` is ignored (the newest page every time).
        self.news_history_status: int | None = None
        self.ignore_enddate = False

    def add_review(self, appid: int, rid: int, created: datetime, *, voted_up: bool = True) -> None:
        stamp = int(created.timestamp())
        self.reviews.setdefault(appid, []).append(
            {
                "recommendationid": str(rid),
                "author": {
                    "steamid": f"765611980{rid:08d}",
                    "personaname": "Someone",
                    "num_games_owned": 3,
                    "num_reviews": 1,
                    "playtime_forever": 900,
                    "playtime_at_review": 300,
                    "playtime_last_two_weeks": 0,
                },
                "language": "english",
                "review": f"review {rid} with a few words",
                "timestamp_created": stamp,
                "timestamp_updated": stamp,
                "voted_up": voted_up,
                "votes_up": 0,
                "votes_funny": 0,
                "weighted_vote_score": "0.5",
                "comment_count": 0,
                "steam_purchase": True,
                "received_for_free": False,
                "written_during_early_access": False,
            }
        )

    def add_news(
        self,
        appid: int,
        gid: str,
        title: str,
        published: datetime,
        *,
        feedname: str = OFFICIAL_FEED,
        tags: Sequence[str] = ("patchnotes",),
    ) -> None:
        self.news.setdefault(appid, []).append(
            {
                "gid": gid,
                "title": title,
                "url": f"https://example.com/{gid}",
                "contents": "notes",
                "date": int(published.timestamp()),
                "feedname": feedname,
                "feedlabel": "Community Announcements" if feedname == OFFICIAL_FEED else feedname,
                "appid": appid,
                "tags": list(tags),
            }
        )

    def http(self) -> SteamHttp:
        return SteamHttp(
            httpx2.Client(transport=httpx2.MockTransport(self)),
            limiter=RateLimiter(min_interval_s=0.0),
            policy=RetryPolicy(max_attempts=1),
        )

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        path = urlsplit(str(request.url)).path
        params = {k: v[0] for k, v in parse_qs(urlsplit(str(request.url)).query).items()}
        if path == "/IUserReviewsService/GetAppReviews/v1/":
            query = json.loads(params["input_json"])
            return self._reviews(int(query["appid"]), query)
        if path.startswith("/appreviews/"):  # switched off by Valve on 22 Oct 2026
            return httpx2.Response(302, headers={"Location": "https://store.steampowered.com/"})
        if "GetNewsForApp" in path:
            return self._news(params)
        if "appdetails" in path and "filters" not in params:
            return self._details(params["appids"])
        if self.prices_down:
            return httpx2.Response(503)
        body: dict[str, Any] = {}
        for requested in params["appids"].split(","):
            overview = self.prices.get(int(requested))
            body[requested] = {
                "success": True,
                "data": {"price_overview": overview} if overview else [],
            }
        return httpx2.Response(200, json=body)

    def _news(self, params: dict[str, str]) -> httpx2.Response:
        """Like GetNewsForApp: newest first, `count` at most, `feeds` and `enddate` (inclusive)
        honoured when sent."""
        if "enddate" in params and self.news_history_status is not None:
            return httpx2.Response(self.news_history_status)
        items = self.news.get(int(params["appid"]), [])
        if "feeds" in params:
            items = [i for i in items if i["feedname"] in params["feeds"].split(",")]
        if "enddate" in params and not self.ignore_enddate:
            items = [i for i in items if i["date"] <= int(params["enddate"])]
        newest = sorted(items, key=lambda i: -int(i["date"]))[: int(params.get("count", 20))]
        return httpx2.Response(200, json={"appnews": {"newsitems": newest}})

    def _reviews(self, appid: int, query: dict[str, Any]) -> httpx2.Response:
        """Like GetAppReviews: the page inside `response`, no `success` field."""
        self.review_requests += 1
        if self.rate_limit_after is not None and self.review_requests > self.rate_limit_after:
            return httpx2.Response(429)
        if appid in self.failing:
            return httpx2.Response(500)
        start, end = int(query["date_range_start"]), int(query["date_range_end"])
        matching = sorted(
            (r for r in self.reviews.get(appid, []) if start <= r["timestamp_created"] <= end),
            key=lambda r: r["timestamp_created"],
            reverse=True,
        )
        broken = self.day_counts_broken.get(appid)
        if broken == "no response":
            return httpx2.Response(200, json={})
        cursor = str(query["cursor"])
        per_page = min(int(query.get("num_per_page", 20)), self.page_size)
        offset = 0 if cursor == "*" else int(cursor[1:])
        served = matching[: self.cut_short.get(appid, len(matching))]
        page = served[offset : offset + per_page]
        next_cursor = f"o{offset + len(page)}"
        # Like Steam: the window's counts on the first page only.
        summary: dict[str, int] = {"num_reviews": len(page)}
        if cursor == "*" and broken != "no summary":
            positive = sum(1 for r in matching if r["voted_up"])
            summary |= {
                "total_positive": positive,
                "total_negative": len(matching) - positive,
                "total_reviews": len(matching),
            }
        return httpx2.Response(
            200,
            json={"response": {"query_summary": summary, "reviews": page, "cursor": next_cursor}},
        )

    def _details(self, appid: str) -> httpx2.Response:
        detail = self.details.get(int(appid))
        entry = {"success": True, "data": detail} if detail else {"success": False}
        return httpx2.Response(200, json={appid: entry})
