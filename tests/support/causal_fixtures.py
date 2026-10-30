"""A small world of games, series and patches for the causal tests (Phase 4)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date, timedelta

import numpy as np

from patchpulse.causal.bundles import Patch
from patchpulse.causal.estimate import EffectInputs, GameInfo
from patchpulse.causal.series import DailySeries

FIRST = date(2026, 1, 5)  # a Monday
DAYS = 190
PATCH_DAY = FIRST + timedelta(days=150)  # a Thursday, with 16 quiet weeks to calibrate on
LAST_DAY = FIRST + timedelta(days=DAYS - 1)
TREATED = 1
DONORS = tuple(range(101, 109))  # eight comparison games, Action
SIM_DONOR = 201


def day(offset: int) -> date:
    """A day counted from the patch."""
    return PATCH_DAY + timedelta(days=offset)


def steady(
    appid: int,
    *,
    per_day: int = 150,
    share: float = 0.8,
    seed: int = 0,
    first: date = FIRST,
    days: int = DAYS,
    shift: float = 0.0,
    common: np.ndarray | None = None,
) -> DailySeries:
    """Binomial days around `share`, plus `common` (a shared daily movement) and `shift` from the
    patch day on."""
    rng = np.random.default_rng(seed)
    p = np.full(days, share)
    if common is not None:
        p += common[(first - FIRST).days : (first - FIRST).days + days]
    offsets = np.array([(first + timedelta(days=i) - PATCH_DAY).days for i in range(days)])
    p[offsets >= 0] += shift
    totals = np.full(days, per_day, dtype=np.int64)
    positives = rng.binomial(totals, np.clip(p, 0.0, 1.0)).astype(np.int64)
    return DailySeries(appid, first, positives, totals, np.ones(days, dtype=bool))


def world(
    *,
    treated: DailySeries | None = None,
    donors: Mapping[int, DailySeries] | None = None,
    patches: Sequence[Patch] | None = None,
    sale_days: frozenset[date] = frozenset(),
    discounts: Mapping[int, frozenset[date]] | None = None,
    known_from: date = FIRST,
    last_day: date = LAST_DAY,
) -> EffectInputs:
    common = np.cumsum(np.random.default_rng(99).normal(0, 0.004, DAYS))
    if treated is None:
        treated = steady(TREATED, seed=1, shift=-0.05, common=common)
    if donors is None:
        donors = {
            a: steady(a, per_day=60, share=0.7 + (a % 5) * 0.03, seed=a, common=common)
            for a in DONORS
        }
    games = {TREATED: GameInfo(TREATED, ("Action",), "both")}
    games |= {a: GameInfo(a, ("Action",), "donor") for a in donors if a != SIM_DONOR}
    games[SIM_DONOR] = GameInfo(SIM_DONOR, ("Simulation",), "donor")
    aspect = DailySeries(
        TREATED, FIRST, (treated.totals // 10) // 3, treated.totals // 10, treated.observed
    )
    established = replace(treated, positives=treated.positives // 2, totals=treated.totals // 2)
    return EffectInputs(
        series={TREATED: treated},
        established={TREATED: established},
        aspects={TREATED: {"performance": aspect}},
        donors=dict(donors),
        games=games,
        patches={
            TREATED: list(patches) if patches is not None else [Patch("p1", TREATED, PATCH_DAY)]
        },
        sale_days=sale_days,
        discounts=dict(discounts or {}),
        discounts_known_from=known_from,
        last_complete_day=last_day,
    )
