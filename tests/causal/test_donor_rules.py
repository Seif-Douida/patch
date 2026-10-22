"""Which comparison games qualify for a patch (Phase 4 design D3, D6)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from patchpulse.causal.bundles import Bundle
from patchpulse.causal.rules import (
    DONOR_MIN_MEDIAN,
    FIT_DAYS,
    MIN_DONORS,
    POST_DAYS,
    GenreGroup,
    eligible_donors,
    genre_group,
    own_fit_window_clear,
    patch_free_donors,
)
from patchpulse.causal.series import DailySeries

D0 = date(2026, 1, 1)
START = D0 + timedelta(days=60)
TREATED = 1


def day(offset: int) -> date:
    return START + timedelta(days=offset)


def bundle(start: date = START, patch_id: str = "p") -> Bundle:
    return Bundle(appid=TREATED, start=start, patch_ids=(patch_id,), allow_14=True)


def steady(appid: int, per_day: int = 30, unobserved: int | None = None) -> DailySeries:
    observed = np.ones(121, dtype=bool)
    if unobserved is not None:
        observed[(day(unobserved) - D0).days] = False
    return DailySeries(
        appid=appid,
        first_day=D0,
        positives=np.full(121, per_day // 2, dtype=np.int64),
        totals=np.full(121, per_day, dtype=np.int64),
        observed=observed,
    )


def test_rule_constants() -> None:
    assert (FIT_DAYS, POST_DAYS, MIN_DONORS, DONOR_MIN_MEDIAN) == (28, 14, 5, 20)


def test_genre_group_puts_simulation_first() -> None:
    assert genre_group(["Simulation", "Strategy"]) is GenreGroup.SIM_STRATEGY  # Manor Lords
    assert genre_group(["Simulation"]) is GenreGroup.SIM_STRATEGY  # Cities: Skylines II
    assert genre_group(["Adventure", "RPG", "Strategy"]) is GenreGroup.ACTION_RPG  # BG3
    assert genre_group(["Action"]) is GenreGroup.ACTION_RPG


def test_own_fit_window_clear() -> None:
    this = bundle()

    assert own_fit_window_clear(this, [this])
    assert not own_fit_window_clear(this, [bundle(day(-28), "early"), this])
    assert own_fit_window_clear(this, [bundle(day(-29), "earlier"), this])
    assert own_fit_window_clear(this, [this, bundle(day(3), "later")])


def candidates() -> tuple[dict[int, GenreGroup], dict[int, list[date]], dict[int, DailySeries]]:
    action, sim = GenreGroup.ACTION_RPG, GenreGroup.SIM_STRATEGY
    groups = {appid: action for appid in (1, 2, 4, 5, 6, 7, 8, 9, 10)} | {3: sim}
    patch_days = {4: [day(-28)], 5: [day(13)], 6: [day(-29)], 7: [day(14)]}
    series = {appid: steady(appid) for appid in (1, 2, 3, 4, 5, 6, 7)}
    series[8] = steady(8, unobserved=5)
    series[9] = steady(9, per_day=19)
    return groups, patch_days, series  # 10 has no series at all


def test_eligibility_excludes_each_case() -> None:
    groups, patch_days, series = candidates()

    chosen = eligible_donors(bundle(), GenreGroup.ACTION_RPG, groups, patch_days, series)

    # 1 is the treated game, 3 another group, 4 and 5 patched inside the span, 8 has a day missing,
    # 9 a median of 19 and 10 no data; 6 and 7 patched just outside it.
    assert chosen == [2, 6, 7]


def test_patch_free_donors_skips_the_data_checks() -> None:
    groups, patch_days, _ = candidates()

    free = patch_free_donors(bundle(), GenreGroup.ACTION_RPG, groups, patch_days)

    assert free == [2, 6, 7, 8, 9, 10]
