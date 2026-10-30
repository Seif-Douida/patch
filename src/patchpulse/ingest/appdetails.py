"""A game's name, genres and release date from the store's appdetails (Phase 4, design D3).

One request per game, in English with US settings like the price stage. The release date is
`release_date.date` in the store's format ("Apr 26, 2024"; some regions answer "26 Apr, 2024");
a game not yet released, or a date without a day, gives None.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from patchpulse.ingest.http import SteamError, SteamHttp
from patchpulse.ingest.prices import APPDETAILS_URL, PRICE_COUNTRY

_DATE_FORMATS = ("%b %d, %Y", "%d %b, %Y", "%B %d, %Y", "%d %B, %Y")


@dataclass(frozen=True)
class AppDetails:
    appid: int
    name: str
    genres: tuple[str, ...]
    release_date: date | None


def parse_release_date(release: Any) -> date | None:
    if not isinstance(release, dict) or release.get("coming_soon"):
        return None
    text = str(release.get("date", "")).strip()
    for pattern in _DATE_FORMATS:
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    return None


def fetch_app_details(http: SteamHttp, appid: int) -> AppDetails:
    request: dict[str, str | int] = {"appids": appid, "cc": PRICE_COUNTRY, "l": "english"}
    payload, _ = http.get_json(APPDETAILS_URL, request)
    entry = payload.get(str(appid))
    data = entry.get("data") if isinstance(entry, dict) and entry.get("success") else None
    if not isinstance(data, dict):
        raise SteamError(f"appdetails {appid}: not found")
    genres = tuple(str(g["description"]) for g in data.get("genres") or [] if isinstance(g, dict))
    return AppDetails(
        appid, str(data.get("name", "")), genres, parse_release_date(data.get("release_date"))
    )
