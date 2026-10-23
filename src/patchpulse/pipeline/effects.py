"""The nightly effects stage (Phase 4, design D10): estimate each patch bundle whose window has
ended or whose inputs changed, and store its rows in app.patch_effect.

The inputs are the marts dbt has just built: tracked games' days (all reviews, established
players' reviews, and each aspect among scored negative reviews), comparison games' daily counts
from Steam, every game's treatment-candidate patches, Steam's sales and the discounts the price
snapshots have seen. A day is complete once it has ended in UTC, so a night reads through
yesterday. Bundles go newest first, one transaction each, until the stage's deadline; the next
night finds the rest by their fingerprints.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import Connection, Engine, bindparam, text

from patchpulse.causal.bundles import Patch, window_complete
from patchpulse.causal.estimate import (
    HORIZONS,
    EffectInputs,
    GameInfo,
    bundles_to_estimate,
    estimate_bundle,
    tracked_bundles,
)
from patchpulse.causal.series import from_counts
from patchpulse.causal.store import stored_fingerprints, write_estimates

log = logging.getLogger("patchpulse.pipeline.effects")

# Playtime at review of 10 hours or more (core.fact_review's buckets): established players.
ESTABLISHED_BUCKETS = ("10h_50h", "50h_200h", "200h_plus")
TRACKED_ROLES = ("tracked", "both")

DayCounts = dict[date, tuple[int, int]]


@dataclass(frozen=True)
class EffectsOutcome:
    estimated: int  # bundles estimated tonight
    written: int  # rows stored (only those whose fingerprint changed)
    unchanged: int  # bundles with a complete window whose rows were all current
    remaining: int  # bundles due but left for the next night by the deadline


def _day(date_key: int) -> date:
    return date(date_key // 10_000, date_key // 100 % 100, date_key % 100)


def _date_key(day: date) -> int:
    return day.year * 10_000 + day.month * 100 + day.day


def _by_game(rows: Iterable[Any]) -> dict[int, DayCounts]:
    """(appid, date_key, positive, total) rows as each game's counts per day."""
    counts: dict[int, DayCounts] = defaultdict(dict)
    for appid, date_key, positive, total in rows:
        counts[int(appid)][_day(int(date_key))] = (int(positive), int(total))
    return counts


def read_inputs(connection: Connection, *, today: date) -> EffectInputs:
    """Everything the estimators read, through yesterday (UTC)."""
    last = today - timedelta(days=1)
    through = {"last": _date_key(last)}
    games = {
        int(r.appid): GameInfo(
            int(r.appid),
            tuple(g.strip() for g in (r.genres or "").split(",") if g.strip()),
            str(r.role),
        )
        for r in connection.execute(text("SELECT appid, genres, role FROM core.dim_game"))
    }
    reviewed = _by_game(
        connection.execute(
            text(
                "SELECT game_key, date_key, n_positive, n_reviews FROM mart.daily_game_sentiment "
                "WHERE date_key <= :last"
            ),
            through,
        )
    )
    # A tracked game's days run from its first review to yesterday; a day without reviews is 0.
    spans = {
        appid: (min(days), last)
        for appid, days in reviewed.items()
        if appid in games and games[appid].role in TRACKED_ROLES
    }
    established = _by_game(
        connection.execute(
            text(
                "SELECT game_key, date_key, SUM(n_positive), SUM(n_reviews) "
                "FROM mart.daily_playtime_sentiment "
                "WHERE playtime_bucket IN :buckets AND date_key <= :last "
                "GROUP BY game_key, date_key"
            ).bindparams(bindparam("buckets", expanding=True)),
            {**through, "buckets": list(ESTABLISHED_BUCKETS)},
        )
    )
    aspects: dict[int, dict[str, DayCounts]] = defaultdict(lambda: defaultdict(dict))
    for r in connection.execute(
        text(
            "SELECT game_key, date_key, aspect_key, n_neg_mentions, n_neg_scored "
            "FROM mart.daily_aspect_share WHERE date_key <= :last"
        ),
        through,
    ):
        aspects[int(r.game_key)][str(r.aspect_key)][_day(int(r.date_key))] = (
            int(r.n_neg_mentions),
            int(r.n_neg_scored),
        )
    donors = _by_game(
        connection.execute(
            text(
                "SELECT game_key, date_key, n_positive, n_reviews "
                "FROM mart.daily_comparison_counts WHERE date_key <= :last"
            ),
            through,
        )
    )
    patches: dict[int, list[Patch]] = defaultdict(list)
    for r in connection.execute(
        text(
            "SELECT patch_id, game_key, date_key FROM core.fact_patch "
            "WHERE is_treatment_candidate = 1 ORDER BY game_key, date_key, patch_id"
        )
    ):
        patches[int(r.game_key)].append(Patch(str(r.patch_id), int(r.game_key), _day(r.date_key)))
    discounts: dict[int, set[date]] = defaultdict(set)
    known_from = today  # no price snapshot yet: every discount is unknown
    for r in connection.execute(
        text("SELECT game_key, date_key, discount_pct FROM core.fact_price_snapshot WHERE ok = 1")
    ):
        day = _day(int(r.date_key))
        known_from = min(known_from, day)
        if r.discount_pct:
            discounts[int(r.game_key)].add(day)
    sales = connection.execute(text("SELECT [date] FROM core.dim_date WHERE is_steam_sale = 1"))
    return EffectInputs(
        series={
            a: from_counts(a, first, end, reviewed[a], missing_is_zero=True)
            for a, (first, end) in spans.items()
        },
        established={
            a: from_counts(a, first, end, established.get(a, {}), missing_is_zero=True)
            for a, (first, end) in spans.items()
        },
        aspects={
            a: {
                code: from_counts(a, *spans[a], days, missing_is_zero=True)
                for code, days in by_code.items()
            }
            for a, by_code in aspects.items()
            if a in spans
        },
        donors={
            a: from_counts(a, min(days), max(days), days, missing_is_zero=False)
            for a, days in donors.items()
        },
        games=games,
        patches=dict(patches),
        sale_days=frozenset(r[0] for r in sales),
        discounts={a: frozenset(days) for a, days in discounts.items()},
        discounts_known_from=known_from,
        last_complete_day=last,
    )


def run_effects(
    engine: Engine,
    *,
    run_id: int,
    now: datetime,
    clock: Callable[[], float],
    deadline: float,
) -> EffectsOutcome:
    """Estimate the due bundles, newest first, each in its own transaction, until `deadline` (on
    `clock`); a bundle started before the deadline finishes."""
    began = time.monotonic()
    with engine.connect() as connection:
        inputs = read_inputs(connection, today=now.date())
        stored = stored_fingerprints(connection)
    due = bundles_to_estimate(inputs, stored)
    shortest = min(HORIZONS)
    complete = [
        b for b in tracked_bundles(inputs) if window_complete(b, shortest, inputs.last_complete_day)
    ]
    estimated = written = 0
    for bundle in due:
        if clock() >= deadline:
            break
        rows = estimate_bundle(bundle, inputs)
        with engine.begin() as connection:
            written += write_estimates(connection, rows, run_id=run_id, computed_at=now)
        estimated += 1
    outcome = EffectsOutcome(
        estimated=estimated,
        written=written,
        unchanged=len(complete) - len(due),
        remaining=len(due) - estimated,
    )
    log.info(
        "effects: %d bundles estimated, %d rows written, %d unchanged, %d left for the next "
        "night (%.1f s)",
        outcome.estimated,
        outcome.written,
        outcome.unchanged,
        outcome.remaining,
        time.monotonic() - began,
    )
    return outcome
