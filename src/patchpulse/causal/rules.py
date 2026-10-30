"""Which comparison games qualify for a patch (design D3, D6).

A comparison game qualifies when it shares the treated game's genre group, had no patch of its own
from 28 days before to 13 days after the start (the twin's fit window and the longest horizon: a
patch while the twin is fitted distorts it; stricter than the spec's "±window", Seif 2026-10-21),
has data on every day of that span and a median of at least 20 reviews a day over it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from enum import StrEnum

import numpy as np

from patchpulse.causal.bundles import Bundle
from patchpulse.causal.series import DailySeries

FIT_DAYS = 28
POST_DAYS = 14
MIN_DONORS = 5
DONOR_MIN_MEDIAN = 20


class GenreGroup(StrEnum):
    ACTION_RPG = "action_rpg"
    SIM_STRATEGY = "sim_strategy"


def genre_group(genres: Iterable[str]) -> GenreGroup:
    """Simulation decides: Cities: Skylines II and Manor Lords are in; Baldur's Gate 3's Strategy
    alone isn't."""
    return GenreGroup.SIM_STRATEGY if "Simulation" in set(genres) else GenreGroup.ACTION_RPG


def own_fit_window_clear(bundle: Bundle, game_bundles: Sequence[Bundle]) -> bool:
    """No other bundle of the treated game starts in the 28 days before this one."""
    earliest = bundle.start - timedelta(days=FIT_DAYS)
    return not any(
        other.appid == bundle.appid
        and other.bundle_id != bundle.bundle_id
        and earliest <= other.start < bundle.start
        for other in game_bundles
    )


def donor_span(bundle: Bundle) -> tuple[date, date]:
    """The days a comparison game must be clean and complete on: the fit window and 14 days."""
    return bundle.start - timedelta(days=FIT_DAYS), bundle.start + timedelta(days=POST_DAYS - 1)


def patch_free_donors(
    bundle: Bundle,
    group: GenreGroup,
    candidates: Mapping[int, GenreGroup],
    patch_days: Mapping[int, Sequence[date]],
) -> list[int]:
    """The genre and patch rules alone, for the candidate check before any history exists."""
    first, last = donor_span(bundle)
    return sorted(
        appid
        for appid, candidate_group in candidates.items()
        if candidate_group == group
        and appid != bundle.appid
        and not any(first <= day <= last for day in patch_days.get(appid, ()))
    )


def eligible_donors(
    bundle: Bundle,
    group: GenreGroup,
    candidates: Mapping[int, GenreGroup],
    patch_days: Mapping[int, Sequence[date]],
    series: Mapping[int, DailySeries],
) -> list[int]:
    """The comparison games that qualify for this bundle, ordered by appid."""
    first, last = donor_span(bundle)
    chosen: list[int] = []
    for appid in patch_free_donors(bundle, group, candidates, patch_days):
        whole = series.get(appid)
        part = whole.span(first, last) if whole is not None else None
        if part is None or not part.observed.all():
            continue
        if float(np.median(part.totals)) < DONOR_MIN_MEDIAN:
            continue
        chosen.append(appid)
    return chosen
