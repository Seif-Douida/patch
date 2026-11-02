"""Build the static JSON the dashboard reads (schema version 4).

Files (paths relative to the site's data folder):

* `index.json`: `schema_version`, `data_version` (the pipeline run), `generated_at` (UTC, ISO),
  `attribution` ("Data from Steam"), `aspect_model` (the model that detects aspects: `family`,
  `version`, `test_macro_f1` on the gold test set, and `aspect_f1`, each aspect's F1 there, when
  the model records it), and `games`: one entry per game, sorted by name, with `appid`, `name`,
  `genres`, `first_date`, `last_date`, `latest` (the last 30 days: `days`, `to`, `n`, `share`; null
  before the game has data) and `top_complaint_30d` (over the same 30 days, the aspect mentioned
  by the largest share of scored negative reviews: `code`, `share`, `n` scored negative reviews;
  off-topic and aspects with an F1 under MIN_ASPECT_F1 don't count; null without any), and
  `causal` (Phase 4): `default_method` (its or synth, from config/causal.yaml), `effects_version`,
  `benchmark_report` (the report's path in the repository; null until the benchmark has run) and
  `validated` (a method met ADR-007's rule; false until then), and `reports` (Phase 5): how many
  report files there are.
* `games/{appid}.json`: the same header fields plus `appid`, `name`, `genres`, `series` (one
  point per day: `date`, `n`, `pos`, `share`, `lo`, `hi`, `sale`, `discount`), `patches`
  (`date`, `title`, `url`, `type`, `treatment`, `effect`, `report`) and `aspects`: `codes` (the
  taxonomy, in order) and `days`, one per day with scored reviews: `date`, `scored` ([negative,
  positive] reviews scored), `neg` and `pos` (per aspect, in `codes` order, how many of them
  mention it).
* `reports/{bundle_id}.json` (schema 4): the same header fields plus `report`, the public copy of
  the bundle's report (agent/report.py `public_copy`: no quote spans, no review ids, withheld
  claims as counts per rule): the 14-day report once it exists, the 7-day one until then. Only
  reports a patch links to are written.

A patch's `effect` (schema 3) is null unless it's a treatment candidate. Otherwise it's its
bundle's (patches within 7 days of the first are estimated together): `start`, the bundle's first
day, and `horizons`, one per horizon (7 and 14 days): `days`, `ends` (the window's last day), the
estimated effect on the positive share in percentage points (`effect`, `lo90`, `hi90`, `method`,
`flags`) as estimate_patch_effect serves it (the default method, or interrupted time series with
the default's reason where the default has none), and `reason` when there's no estimate:
`window_not_complete`, `another_patch_within_14_days` (14 days only), `not_estimated_yet` (the
effects stage hasn't reached it), or the estimator's own words.

A patch's `report` (schema 4) is its bundle's id when that bundle has a report, else null.

Without a model (no `PP_MODEL_DIR`), `aspect_model`, `top_complaint_30d` and `aspects` are null.
Each schema only adds fields, so a site that reads two versions treats the new ones as optional.

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

from patchpulse.agent.report import Report, public_copy
from patchpulse.causal.bundles import LONG_HORIZON, Patch, bundle_patches, window_complete
from patchpulse.causal.config import CAUSAL_CONFIG_PATH, read_causal_config
from patchpulse.causal.estimate import EFFECTS_VERSION, HORIZONS, Method, Sample
from patchpulse.causal.its import POSITIVE_SHARE
from patchpulse.causal.store import latest_estimates, serve
from patchpulse.labeling.taxonomy import ASPECTS

log = logging.getLogger("patchpulse.export")

SCHEMA_VERSION = 4
ATTRIBUTION = "Data from Steam"
LATEST_DAYS = 30
_PLACES = 4
NOT_A_COMPLAINT = "off_topic"  # memes and jokes, not a complaint about the game
# An aspect the model detects with a gold-test F1 under this isn't named as a top complaint; the
# site marks it (web/lib/aspects.ts uses the same bar).
MIN_ASPECT_F1 = 0.5
BENCHMARK_REPORT = "reports/causal/benchmark.md"
# Why a treatment patch's horizon has no estimate when no stored row says.
WINDOW_NOT_COMPLETE = "window_not_complete"
ANOTHER_PATCH = "another_patch_within_14_days"
NOT_ESTIMATED_YET = "not_estimated_yet"


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
    patch_id: str


@dataclass(frozen=True)
class EffectRow:
    """The positive-share estimate served for one bundle and horizon (causal.store.serve)."""

    bundle_id: str
    horizon: int
    effect: float | None  # percentage points
    lo90: float | None
    hi90: float | None
    method: str
    flags: frozenset[str]
    reason: str | None


@dataclass(frozen=True)
class Effects:
    default_method: str
    effects_version: str
    benchmark_report: str | None
    validated: bool  # a method met ADR-007's rule (config/causal.yaml)
    rows: Sequence[EffectRow]


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


def _horizon(days: int, ends: date, reason: str, row: EffectRow | None) -> dict[str, Any]:
    """One horizon of a patch's `effect`: the served row, or `reason` without one."""
    return {
        "days": days,
        "ends": ends.isoformat(),
        "effect": _round(row.effect) if row else None,
        "lo90": _round(row.lo90) if row else None,
        "hi90": _round(row.hi90) if row else None,
        "method": row.method if row else None,
        "flags": sorted(row.flags) if row else [],
        "reason": row.reason if row else reason,
    }


def _bundle_ids(patches: Sequence[PatchRow]) -> dict[str, str]:
    """Each treatment candidate's bundle id (its bundle's first patch), by patch id."""
    candidates = [
        Patch(p.patch_id, p.appid, p.published_at.astimezone(UTC).date())
        for p in patches
        if p.is_treatment_candidate
    ]
    return {
        patch_id: bundle.bundle_id
        for bundle in bundle_patches(candidates)
        for patch_id in bundle.patch_ids
    }


def _effect_blocks(
    patches: Sequence[PatchRow], effects: Effects, last_complete_day: date
) -> dict[str, dict[str, Any]]:
    """Each treatment candidate's `effect`, by patch id: its bundle's, so every patch of a bundle
    shows the same estimate."""
    candidates = [
        Patch(p.patch_id, p.appid, p.published_at.astimezone(UTC).date())
        for p in patches
        if p.is_treatment_candidate
    ]
    served = {(r.bundle_id, r.horizon): r for r in effects.rows}
    blocks: dict[str, dict[str, Any]] = {}
    for bundle in bundle_patches(candidates):
        horizons = []
        for days in HORIZONS:
            ends = bundle.start + timedelta(days=days - 1)
            if days == LONG_HORIZON and not bundle.allow_14:
                horizons.append(_horizon(days, ends, ANOTHER_PATCH, None))
            elif not window_complete(bundle, days, last_complete_day):
                horizons.append(_horizon(days, ends, WINDOW_NOT_COMPLETE, None))
            else:
                row = served.get((bundle.bundle_id, days))
                horizons.append(_horizon(days, ends, NOT_ESTIMATED_YET, row))
        block = {"start": bundle.start.isoformat(), "horizons": horizons}
        for patch_id in bundle.patch_ids:
            blocks[patch_id] = block
    return blocks


def shape_site_export(
    games: Sequence[GameRow],
    days: Sequence[DayRow],
    patches: Sequence[PatchRow],
    *,
    data_version: int,
    generated_at: datetime,
    aspects: Aspects | None = None,
    effects: Effects | None = None,
    reports: Mapping[str, Report] | None = None,
) -> dict[str, bytes]:
    """Turn mart rows into site files: {relative path: file bytes}. Without `effects` (no
    estimates read), the index's `causal` and every patch's `effect` are null. `reports` are the
    stored reports each bundle shows, by bundle id; only their public copies are written."""
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

    last_complete_day = generated_at.astimezone(UTC).date() - timedelta(days=1)
    available = reports or {}

    files: dict[str, bytes] = {}
    listing: list[dict[str, Any]] = []
    shown = (g for g in games if g.role != "donor")
    for game in sorted(shown, key=lambda g: (g.name, g.appid)):
        series = series_by_game.get(game.appid, [])
        aspect_days = aspects_by_game.get(game.appid, [])
        game_patches = patches_by_game.get(game.appid, [])
        blocks = _effect_blocks(game_patches, effects, last_complete_day) if effects else {}
        linked = {
            patch_id: bundle_id
            for patch_id, bundle_id in _bundle_ids(game_patches).items()
            if bundle_id in available
        }
        for bundle_id in sorted(set(linked.values())):
            files[f"reports/{bundle_id}.json"] = _dump(
                {**header, "report": public_copy(available[bundle_id])}
            )
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
                        "effect": blocks.get(p.patch_id),
                        "report": linked.get(p.patch_id),
                    }
                    for p in game_patches
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
    causal = None
    if effects is not None:
        causal = {
            "default_method": effects.default_method,
            "effects_version": effects.effects_version,
            "benchmark_report": effects.benchmark_report,
            "validated": effects.validated,
        }
    shown_reports = sum(1 for name in files if name.startswith("reports/"))
    files["index.json"] = _dump(
        {
            **header,
            "aspect_model": model,
            "causal": causal,
            "games": listing,
            "reports": shown_reports,
        }
    )
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


def _effects(connection: Connection, config_path: Path) -> Effects:
    """The stored positive-share estimates, served as estimate_patch_effect serves them."""
    config = read_causal_config(config_path)
    stored = latest_estimates(
        connection,
        outcome=POSITIVE_SHARE,
        sample=Sample.ALL,
        methods={config.default_method, Method.ITS},
    )
    return Effects(
        default_method=config.default_method,
        effects_version=EFFECTS_VERSION,
        benchmark_report=BENCHMARK_REPORT if config.benchmarked else None,
        validated=config.validated,
        rows=[
            EffectRow(r.bundle_id, r.horizon, r.effect, r.lo90, r.hi90, r.method, r.flags, r.reason)
            for r in serve(stored, config.default_method, fallback=True)
        ],
    )


def build_site_export(
    connection: Connection,
    *,
    data_version: int,
    generated_at: datetime,
    aspect_model: AspectModel | None = None,
    causal_config_path: Path = CAUSAL_CONFIG_PATH,
) -> dict[str, bytes]:
    """Read the marts, app.patch_effect and app.report and shape the site files; aspects only
    with a model."""
    # Imported here: the report store fingerprints with the agent's tools.
    from patchpulse.agent.store import latest_reports, shown_reports

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
            str(r.patch_id),
        )
        for r in connection.execute(
            text(
                "SELECT patch_id, game_key, published_at, title, url, patch_type, "
                "is_treatment_candidate FROM core.fact_patch"
            )
        )
    ]
    aspects = None if aspect_model is None else Aspects(aspect_model, _aspect_days(connection))
    return shape_site_export(
        games,
        days,
        patches,
        data_version=data_version,
        generated_at=generated_at,
        aspects=aspects,
        effects=_effects(connection, causal_config_path),
        reports={b: row.report for b, row in shown_reports(latest_reports(connection)).items()},
    )
