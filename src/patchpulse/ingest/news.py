"""Patch notes and news from ISteamNews/GetNewsForApp (spec §6.1; no API key needed).

Only the game's official posts are asked for (`feeds=steam_community_announcements`): press
articles filled the 100 items and hid older patch posts, and nothing downstream used them (4 Nov
2026). `fetch_news_history` pages back with `enddate` to a game's history start."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from pydantic import ValidationError

from patchpulse.ingest.http import SteamHttp
from patchpulse.ingest.models import NewsItem, RawPage
from patchpulse.ingest.patches import OFFICIAL_FEED

NEWS_URL = "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"
NEWS_COUNT = 100


@dataclass(frozen=True)
class NewsFetch:
    raw: RawPage
    items: list[NewsItem]
    invalid: int


@dataclass(frozen=True)
class NewsHistory:
    fetches: list[NewsFetch]
    reached: bool  # a fetched post is dated before `until`


def fetch_news(
    http: SteamHttp, appid: int, *, count: int = NEWS_COUNT, before: datetime | None = None
) -> NewsFetch:
    """The latest `count` official posts with full contents (`maxlength=0`), up to `before`
    (inclusive) when it's given."""
    request: dict[str, str | int] = {
        "appid": appid,
        "count": count,
        "maxlength": 0,
        "feeds": OFFICIAL_FEED,
    }
    if before is not None:
        request["enddate"] = int(before.timestamp())
    payload, status = http.get_json(NEWS_URL, request)
    raw_items = (payload.get("appnews") or {}).get("newsitems") or []
    items: list[NewsItem] = []
    invalid = 0
    for raw_item in raw_items:
        try:
            items.append(NewsItem.from_steam(raw_item))
        except (ValidationError, TypeError):
            invalid += 1
    raw = RawPage("news", appid, request, status, payload, len(raw_items))
    return NewsFetch(raw, items, invalid)


def fetch_news_history(
    http: SteamHttp,
    appid: int,
    *,
    oldest: datetime,
    until: date,
    max_pages: int,
    count: int = NEWS_COUNT,
) -> NewsHistory:
    """Pages of official posts back from `oldest` (the oldest one stored) until one is dated
    before `until`, at most `max_pages`. Each page starts at the last page's oldest post,
    inclusive, so posts sharing that second aren't skipped; a page with nothing older than its
    start ends the history (Steam has no more, or ignores `enddate`). Raises SteamError."""
    fetches: list[NewsFetch] = []
    boundary = oldest
    for _ in range(max_pages):
        fetch = fetch_news(http, appid, count=count, before=boundary)
        fetches.append(fetch)
        if not fetch.items:
            break
        if any(item.published_at.date() < until for item in fetch.items):
            return NewsHistory(fetches, reached=True)
        page_oldest = min(item.published_at for item in fetch.items)
        if page_oldest >= boundary:
            break
        boundary = page_oldest
    return NewsHistory(fetches, reached=False)
