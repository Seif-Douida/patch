"""Patch notes and news from ISteamNews/GetNewsForApp (spec §6.1; no API key needed)."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from patchpulse.ingest.http import SteamHttp
from patchpulse.ingest.models import NewsItem, RawPage

NEWS_URL = "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"


@dataclass(frozen=True)
class NewsFetch:
    raw: RawPage
    items: list[NewsItem]
    invalid: int


def fetch_news(http: SteamHttp, appid: int, *, count: int = 100) -> NewsFetch:
    """The latest `count` news items with full contents (`maxlength=0`)."""
    request: dict[str, str | int] = {"appid": appid, "count": count, "maxlength": 0}
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
