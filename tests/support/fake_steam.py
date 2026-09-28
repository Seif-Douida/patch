"""An in-process stand-in for Steam's three endpoints, for pipeline tests (no network)."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx2

from patchpulse.ingest.http import RateLimiter, RetryPolicy, SteamHttp

PAGE = 100


class FakeSteam:
    """Serves reviews by date range and cursor, news and prices; records every request."""

    def __init__(self) -> None:
        self.reviews: dict[int, list[dict[str, Any]]] = {}
        self.news: dict[int, list[dict[str, Any]]] = {}
        self.prices: dict[int, dict[str, Any] | None] = {}
        self.failing: set[int] = set()
        self.requests: list[httpx2.Request] = []

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

    def add_news(self, appid: int, gid: str, title: str, published: datetime) -> None:
        self.news.setdefault(appid, []).append(
            {
                "gid": gid,
                "title": title,
                "url": f"https://example.com/{gid}",
                "contents": "notes",
                "date": int(published.timestamp()),
                "feedname": "steam_community_announcements",
                "feedlabel": "Community Announcements",
                "appid": appid,
                "tags": ["patchnotes"],
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
        if path.startswith("/appreviews/"):
            return self._reviews(int(path.rsplit("/", 1)[1]), params)
        if "GetNewsForApp" in path:
            appid = int(params["appid"])
            return httpx2.Response(200, json={"appnews": {"newsitems": self.news.get(appid, [])}})
        body: dict[str, Any] = {}
        for requested in params["appids"].split(","):
            overview = self.prices.get(int(requested))
            body[requested] = {
                "success": True,
                "data": {"price_overview": overview} if overview else [],
            }
        return httpx2.Response(200, json=body)

    def _reviews(self, appid: int, params: dict[str, str]) -> httpx2.Response:
        if appid in self.failing:
            return httpx2.Response(500)
        start, end = int(params["start_date"]), int(params["end_date"])
        matching = sorted(
            (r for r in self.reviews.get(appid, []) if start <= r["timestamp_created"] <= end),
            key=lambda r: r["timestamp_created"],
            reverse=True,
        )
        cursor = params["cursor"]
        offset = 0 if cursor == "*" else int(cursor[1:])
        page = matching[offset : offset + PAGE]
        next_cursor = f"o{offset + len(page)}"
        return httpx2.Response(
            200,
            json={"success": 1, "query_summary": {}, "reviews": page, "cursor": next_cursor},
        )
