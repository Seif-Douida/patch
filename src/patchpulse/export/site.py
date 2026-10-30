"""Build the static JSON the dashboard reads (schema version 2).

Files (paths relative to the site's data folder):

* `index.json`: `schema_version`, `data_version` (the pipeline run), `generated_at` (UTC, ISO),
  `attribution` ("Data from Steam"), `aspect_model` (the model that detects aspects: `family`,
  `version`, `test_macro_f1` on the gold test set, and `aspect_f1`, each aspect's F1 there, when
  the model records it), and `games`: one entry per game, sorted by name, with `appid`, `name`,
  `genres`, `first_date`, `last_date`, `latest` (the last 30 days: `days`, `to`, `n`, `share`; null
  before the game has data) and `top_complaint_30d` (over the same 30 days, the aspect mentioned
  by the largest share of scored negative reviews: `code`, `share`, `n` scored negative reviews;
  off-topic and aspects with an F1 under MIN_ASPECT_F1 don't count; null without any).
* `games/{appid}.json`: the same header fields plus `appid`, `name`, `genres`, `series` (one
  point per day: `date`, `n`, `pos`, `share`, `lo`, `hi`, `sale`, `discount`), `patches`
  (`date`, `title`, `url`, `type`, `treatment`) and `aspects`: `codes` (the taxonomy, in order)
  and `days`, one per day with scored reviews: `date`, `scored` ([negative, positive] reviews
  scored), `neg` and `pos` (per aspect, in `codes` order, how many of them mention it).

Without a model (no `PP_MODEL_DIR`), `aspect_model`, `top_complaint_30d` and `aspects` are null.
Schema 2 only adds fields, so a site that reads both versions treats them as optional.

Output is deterministic: sorted keys and rows, floats rounded to 4 places, UTF-8, a trailing
newline. The same data always gives byte-identical files, so an unchanged night is a no-op commit.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import Connection, text

from patchpulse.labeling.taxonomy import ASPECTS

log = logging.getLogger("patchpulse.export")

SCHEMA_VERSION = 2
ATTRIBUTION = "Data from Steam"
LATEST_DAYS = 30
_PLACES = 4
NOT_A_COMPLAINT = "off_topic"  # memes and jokes, not a complaint about the game
# An aspect the model detects with a gold-test F1 under this isn't named as a top complaint; the
# site marks it (web/lib/aspects.ts uses the same bar).
MIN_ASPECT_F1 = 0.5


@dataclass(frozen=True)
class GameRow:
    appid: int
    name: str
    genres: str  # "Action, RPG" as stored in core.dim_game
    role: str = "tracked"  # comparison games (role donor) are never exported


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


@dataclass(frozen=True)
class AspectModel:
    family: str  # s0, s1a or s1b
    version: str
    test_macro_f1: float | None  # on the gold test set, from the eval report
    aspect_f1: Mapping[str, float] = field(default_factory=dict)  # each aspect's, when recorded


@dataclass(frozen=True)
class AspectDayRow:
    appid: int
    date: date
    n_neg_scored: int
    n_pos_scored: int
    neg: tuple[int, ...]  # mentions per aspect among the scored negative reviews, ASPECTS order
    pos: tuple[int, ...]


@dataclass(frozen=True)
class Aspects:
    model: AspectModel
    days: Sequence[AspectDayRow]


def read_aspect_model(model_dir: Path | None) -> AspectModel | None:
    """The champion's name and gold-test score, from its model.json; None without a model.

    Display only: the scoring stage checks the model itself. An unreadable file loses the block
    for one night instead of failing the export.
    """
    if model_dir is None:
        return None
    try:
        record = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
        metrics = record.get("metrics", {})
        metric = metrics.get("test_macro_f1")
        return AspectModel(
            family=str(record["family"]),
            version=str(record["version"]),
            test_macro_f1=None if metric is None else float(metric),
            aspect_f1={
                code: float(metrics[f"test_f1_{code}"])
                for code in ASPECTS
                if f"test_f1_{code}" in metrics
            },
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        log.warning("aspect model unreadable in %s: %s", model_dir, error)
        return None


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


def _top_complaint(
    aspect_days: Sequence[AspectDayRow],
    latest: dict[str, Any] | None,
    aspect_f1: Mapping[str, float],
) -> dict[str, Any] | None:
    if latest is None:
        return None
    last = date.fromisoformat(latest["to"])
    window = [d for d in aspect_days if last - timedelta(days=LATEST_DAYS) < d.date <= last]
    n = sum(d.n_neg_scored for d in window)
    if n == 0:
        return None
    mentions = [sum(d.neg[i] for d in window) for i in range(len(ASPECTS))]
    candidates = [
        i
        for i, code in enumerate(ASPECTS)
        if code != NOT_A_COMPLAINT and aspect_f1.get(code, 1.0) >= MIN_ASPECT_F1 and mentions[i]
    ]
    if not candidates:
        return None
    top = max(candidates, key=lambda i: (mentions[i], -i))  # a tie goes to the taxonomy's order
    return {"code": ASPECTS[top], "share": _round(mentions[top] / n), "n": n}


def _aspect_series(aspect_days: Sequence[AspectDayRow]) -> dict[str, Any]:
    return {
        "codes": list(ASPECTS),
        "days": [
            {
                "date": d.date.isoformat(),
                "scored": [d.n_neg_scored, d.n_pos_scored],
                "neg": list(d.neg),
                "pos": list(d.pos),
            }
            for d in aspect_days
        ],
    }


def shape_site_export(
    games: Sequence[GameRow],
    days: Sequence[DayRow],
    patches: Sequence[PatchRow],
    *,
    data_version: int,
    generated_at: datetime,
    aspects: Aspects | None = None,
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
    aspects_by_game: dict[int, list[AspectDayRow]] = defaultdict(list)
    for aspect_day in sorted(aspects.days if aspects else [], key=lambda d: (d.appid, d.date)):
        aspects_by_game[aspect_day.appid].append(aspect_day)

    files: dict[str, bytes] = {}
    listing: list[dict[str, Any]] = []
    shown = (g for g in games if g.role != "donor")
    for game in sorted(shown, key=lambda g: (g.name, g.appid)):
        series = series_by_game.get(game.appid, [])
        aspect_days = aspects_by_game.get(game.appid, [])
        genres = [genre.strip() for genre in game.genres.split(",") if genre.strip()]
        latest = _latest(series)
        listing.append(
            {
                "appid": game.appid,
                "name": game.name,
                "genres": genres,
                "first_date": series[0].date.isoformat() if series else None,
                "last_date": series[-1].date.isoformat() if series else None,
                "latest": latest,
                "top_complaint_30d": (
                    _top_complaint(aspect_days, latest, aspects.model.aspect_f1)
                    if aspects
                    else None
                ),
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
                "aspects": _aspect_series(aspect_days) if aspects else None,
            }
        )
    model = None
    if aspects is not None:
        model = {
            "family": aspects.model.family,
            "version": aspects.model.version,
            "test_macro_f1": _round(aspects.model.test_macro_f1),
            "aspect_f1": {code: _round(f1) for code, f1 in aspects.model.aspect_f1.items()},
        }
    files["index.json"] = _dump({**header, "aspect_model": model, "games": listing})
    return files


def _aspect_days(connection: Connection) -> list[AspectDayRow]:
    """mart.daily_aspect_share as one row per game and day with scored reviews."""
    position = {code: i for i, code in enumerate(ASPECTS)}
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for r in connection.execute(
        text(
            "SELECT game_key, date_key, aspect_key, n_neg_mentions, n_pos_mentions, "
            "n_neg_scored, n_pos_scored FROM mart.daily_aspect_share "
            "WHERE n_neg_scored + n_pos_scored > 0"
        )
    ):
        day = grouped.setdefault(
            (r.game_key, r.date_key),
            {
                "scored": (r.n_neg_scored, r.n_pos_scored),
                "neg": [0] * len(ASPECTS),
                "pos": [0] * len(ASPECTS),
            },
        )
        i = position[r.aspect_key]  # dbt's seed is tested equal to ASPECTS
        day["neg"][i], day["pos"][i] = r.n_neg_mentions, r.n_pos_mentions
    return [
        AspectDayRow(
            appid=game_key,
            date=datetime.strptime(str(date_key), "%Y%m%d").date(),
            n_neg_scored=day["scored"][0],
            n_pos_scored=day["scored"][1],
            neg=tuple(day["neg"]),
            pos=tuple(day["pos"]),
        )
        for (game_key, date_key), day in grouped.items()
    ]


def build_site_export(
    connection: Connection,
    *,
    data_version: int,
    generated_at: datetime,
    aspect_model: AspectModel | None = None,
) -> dict[str, bytes]:
    """Read the marts and shape the site files; aspects only with a model."""
    games = [
        GameRow(r.appid, r.name, r.genres, r.role)
        for r in connection.execute(text("SELECT appid, name, genres, role FROM core.dim_game"))
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
    aspects = None if aspect_model is None else Aspects(aspect_model, _aspect_days(connection))
    return shape_site_export(
        games, days, patches, data_version=data_version, generated_at=generated_at, aspects=aspects
    )
