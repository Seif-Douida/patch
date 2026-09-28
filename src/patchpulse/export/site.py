"""Build the static JSON the dashboard reads (schema version 1).

Files (paths relative to the site's data folder):

* `index.json`: `schema_version`, `data_version` (the pipeline run), `generated_at` (UTC, ISO),
  `attribution` ("Data from Steam"), and `games`: one entry per game, sorted by name, with
  `appid`, `name`, `genres`, `first_date`, `last_date` and `latest` (the last 30 days:
  `days`, `to`, `n`, `share`; null before the game has data).
* `games/{appid}.json`: the same header fields plus `appid`, `name`, `genres`, `series` (one
  point per day: `date`, `n`, `pos`, `share`, `lo`, `hi`, `sale`, `discount`) and `patches`
  (`date`, `title`, `url`, `type`, `treatment`).

Output is deterministic: sorted keys and rows, floats rounded to 4 places, UTF-8, a trailing
newline. The same data always gives byte-identical files, so an unchanged night is a no-op commit.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import Connection, text

SCHEMA_VERSION = 1
ATTRIBUTION = "Data from Steam"
LATEST_DAYS = 30
_PLACES = 4


@dataclass(frozen=True)
class GameRow:
    appid: int
    name: str
    genres: str  # "Action, RPG" as stored in core.dim_game


@dataclass(frozen=True)
class DayRow:
    appid: int
    date: date
    n_reviews: int
    n_positive: int
    pos_share: float | None
    pos_lo90: float | None
    pos_hi90: float | None
    discount_pct: int | None
    is_sale: bool


@dataclass(frozen=True)
class PatchRow:
    appid: int
    published_at: datetime
    title: str
    url: str
    patch_type: str
    is_treatment_candidate: bool


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, _PLACES)


def _dump(document: dict[str, Any]) -> bytes:
    text_ = json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return (text_ + "\n").encode("utf-8")


def _latest(series: Sequence[DayRow]) -> dict[str, Any] | None:
    if not series:
        return None
    last = series[-1].date
    window = [d for d in series if d.date > last - timedelta(days=LATEST_DAYS)]
    n = sum(d.n_reviews for d in window)
    positive = sum(d.n_positive for d in window)
    return {
        "days": LATEST_DAYS,
        "to": last.isoformat(),
        "n": n,
        "share": _round(positive / n) if n else None,
    }


def shape_site_export(
    games: Sequence[GameRow],
    days: Sequence[DayRow],
    patches: Sequence[PatchRow],
    *,
    data_version: int,
    generated_at: datetime,
) -> dict[str, bytes]:
    """Turn mart rows into site files: {relative path: file bytes}."""
    header = {
        "schema_version": SCHEMA_VERSION,
        "data_version": data_version,
        "generated_at": generated_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "attribution": ATTRIBUTION,
    }
    series_by_game: dict[int, list[DayRow]] = defaultdict(list)
    for row in sorted(days, key=lambda d: (d.appid, d.date)):
        series_by_game[row.appid].append(row)
    patches_by_game: dict[int, list[PatchRow]] = defaultdict(list)
    for patch in sorted(patches, key=lambda p: (p.appid, p.published_at, p.title)):
        patches_by_game[patch.appid].append(patch)

    files: dict[str, bytes] = {}
    listing: list[dict[str, Any]] = []
    for game in sorted(games, key=lambda g: (g.name, g.appid)):
        series = series_by_game.get(game.appid, [])
        genres = [genre.strip() for genre in game.genres.split(",") if genre.strip()]
        listing.append(
            {
                "appid": game.appid,
                "name": game.name,
                "genres": genres,
                "first_date": series[0].date.isoformat() if series else None,
                "last_date": series[-1].date.isoformat() if series else None,
                "latest": _latest(series),
            }
        )
        files[f"games/{game.appid}.json"] = _dump(
            {
                **header,
                "appid": game.appid,
                "name": game.name,
                "genres": genres,
                "series": [
                    {
                        "date": d.date.isoformat(),
                        "n": d.n_reviews,
                        "pos": d.n_positive,
                        "share": _round(d.pos_share),
                        "lo": _round(d.pos_lo90),
                        "hi": _round(d.pos_hi90),
                        "sale": d.is_sale,
                        "discount": d.discount_pct,
                    }
                    for d in series
                ],
                "patches": [
                    {
                        "date": p.published_at.date().isoformat(),
                        "title": p.title,
                        "url": p.url,
                        "type": p.patch_type,
                        "treatment": p.is_treatment_candidate,
                    }
                    for p in patches_by_game.get(game.appid, [])
                ],
            }
        )
    files["index.json"] = _dump({**header, "games": listing})
    return files


def build_site_export(
    connection: Connection, *, data_version: int, generated_at: datetime
) -> dict[str, bytes]:
    """Read the marts and shape the site files."""
    games = [
        GameRow(r.appid, r.name, r.genres)
        for r in connection.execute(text("SELECT appid, name, genres FROM core.dim_game"))
    ]
    days = [
        DayRow(
            r.appid,
            r.date,
            r.n_reviews,
            r.n_positive,
            r.pos_share,
            r.pos_lo90,
            r.pos_hi90,
            r.discount_pct,
            bool(r.is_sale),
        )
        for r in connection.execute(
            text(
                "SELECT appid, [date], n_reviews, n_positive, pos_share, pos_lo90, pos_hi90, "
                "discount_pct, is_sale FROM mart.daily_game_sentiment"
            )
        )
    ]
    patches = [
        PatchRow(
            r.game_key,
            r.published_at.replace(tzinfo=UTC),
            r.title,
            r.url,
            r.patch_type,
            bool(r.is_treatment_candidate),
        )
        for r in connection.execute(
            text(
                "SELECT game_key, published_at, title, url, patch_type, is_treatment_candidate "
                "FROM core.fact_patch"
            )
        )
    ]
    return shape_site_export(
        games, days, patches, data_version=data_version, generated_at=generated_at
    )
