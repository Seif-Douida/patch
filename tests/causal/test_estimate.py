"""Both methods put together for a patch: rows, flags, fallbacks and fingerprints (design D2-D7)."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest
from causal_fixtures import (
    DONORS,
    FIRST,
    PATCH_DAY,
    TREATED,
    day,
    steady,
    world,
)

from patchpulse.causal import estimate
from patchpulse.causal.bundles import Bundle, Patch
from patchpulse.causal.estimate import (
    EFFECTS_VERSION,
    EffectInputs,
    Estimate,
    EstimateKey,
    Method,
    Sample,
    bundles_to_estimate,
    estimate_bundle,
    fingerprint,
    flags_for,
    volume_spike,
)
from patchpulse.causal.its import POSITIVE_SHARE, VOLUME, aspect_outcome
from patchpulse.causal.series import DailySeries

BUNDLE = Bundle(TREATED, PATCH_DAY, ("p1",), True)


def assert_clean(rows: list[Estimate]) -> None:
    for row in rows:
        for value in (row.effect, row.lo90, row.hi90, row.p_value, row.pre_fit_rmse):
            assert value is None or math.isfinite(value), row
        if row.effect is None:
            assert row.reason, row
        assert row.version == EFFECTS_VERSION
        assert len(row.fingerprint) == 64


def find(
    rows: list[Estimate],
    outcome: str,
    method: Method,
    horizon: int = 14,
    sample: Sample = Sample.ALL,
) -> Estimate:
    matches = [
        r
        for r in rows
        if (r.outcome, r.method, r.horizon, r.sample) == (outcome, method, horizon, sample)
    ]
    assert len(matches) == 1, [(r.outcome, r.method, r.horizon, r.sample) for r in rows]
    return matches[0]


def test_a_full_bundle_gets_every_row() -> None:
    rows = estimate_bundle(BUNDLE, world())

    assert_clean(rows)
    # Per horizon: ITS for share, volume, one aspect and established players; synth for share and
    # volume.
    assert len(rows) == 12
    its = find(rows, POSITIVE_SHARE, Method.ITS)
    synth = find(rows, POSITIVE_SHARE, Method.SYNTH)
    # The world's shared drift rises 1.8 points above its own trend in the 14 days after the
    # patch: interrupted time series can't tell that from the patch (ADR-007), but its calibrated
    # interval still holds the planted -5; synthetic control sees the drift in the donors.
    assert its.lo90 is not None
    assert its.hi90 is not None
    assert its.lo90 <= -5.0 <= its.hi90
    assert synth.effect == pytest.approx(-5.0, abs=2.5)
    assert synth.lo90 is not None
    assert synth.hi90 is not None
    assert synth.p_value is not None
    assert len(synth.donors) >= 1
    assert find(rows, aspect_outcome("performance"), Method.ITS).effect is not None
    assert find(rows, POSITIVE_SHARE, Method.ITS, sample=Sample.ESTABLISHED).effect is not None
    assert {r.key for r in rows} == {
        (r.bundle_id, r.outcome, r.sample, r.horizon, r.method) for r in rows
    }


def test_estimates_handle_odd_series() -> None:
    gaps = steady(TREATED, seed=1, shift=-0.05)
    totals, positives = gaps.totals.copy(), gaps.positives.copy()
    totals[::9], positives[::9] = 0, 0  # days with no reviews
    holey = replace(gaps, totals=totals, positives=positives)
    assert find(estimate_bundle(BUNDLE, world(treated=holey)), POSITIVE_SHARE, Method.ITS).effect

    one_missing = steady(DONORS[0], per_day=60, seed=5)
    observed = one_missing.observed.copy()
    observed[(day(-3) - FIRST).days] = False  # a comparison game's day not fetched
    five = {a: steady(a, per_day=60, seed=a) for a in DONORS[1:5]}
    five[DONORS[0]] = replace(one_missing, observed=observed)
    rows = estimate_bundle(BUNDLE, world(donors=five))
    assert "too few comparison games" in (find(rows, POSITIVE_SHARE, Method.SYNTH).reason or "")

    launched = steady(TREATED, first=PATCH_DAY - timedelta(days=20), days=40)
    rows = estimate_bundle(BUNDLE, world(treated=launched))
    assert find(rows, POSITIVE_SHARE, Method.ITS).reason == "pre-period under 28 days with reviews"
    assert find(rows, POSITIVE_SHARE, Method.SYNTH).effect is None

    from_day_one = steady(TREATED, first=PATCH_DAY, days=40)
    rows = estimate_bundle(BUNDLE, world(treated=from_day_one))
    assert find(rows, POSITIVE_SHARE, Method.ITS).effect is None
    assert_clean(rows)

    assert estimate_bundle(BUNDLE, world(last_day=PATCH_DAY + timedelta(days=3))) == []

    perfect = steady(TREATED, seed=1)
    all_positive = replace(perfect, positives=perfect.totals.copy())
    assert_clean(estimate_bundle(BUNDLE, world(treated=all_positive)))

    thin = steady(TREATED, per_day=8, seed=1)
    rows = estimate_bundle(BUNDLE, world(treated=thin))
    assert (
        find(rows, POSITIVE_SHARE, Method.ITS).reason == "too few reviews (median under 10 a day)"
    )
    assert_clean(rows)


def test_each_flag() -> None:
    def flags(**kwargs: object) -> frozenset[str]:
        return flags_for(BUNDLE, 14, world(**kwargs))  # type: ignore[arg-type]

    assert "confounded" in flags(sale_days=frozenset({day(-3)}))
    assert "confounded" in flags(sale_days=frozenset({day(3)}))
    assert "confounded" not in flags(sale_days=frozenset({day(-4)}))
    assert "confounded" in flags(discounts={TREATED: frozenset({day(1)})})
    assert "discounts_unknown" in flags(known_from=day(-2))
    assert "discounts_unknown" not in flags(known_from=day(-3))

    spiky = steady(TREATED, seed=1)
    totals, positives = spiky.totals.copy(), spiky.positives.copy()
    totals[(day(2) - FIRST).days] *= 5
    positives[(day(2) - FIRST).days] *= 5
    assert "volume_spike" in flags(treated=replace(spiky, totals=totals, positives=positives))
    assert "volume_spike" not in flags()

    rows = estimate_bundle(BUNDLE, world(treated=steady(TREATED, per_day=15, seed=1)))
    assert "binned" in find(rows, POSITIVE_SHARE, Method.ITS).flags


def test_volume_spike_is_a_day_three_sds_above_the_month_before() -> None:
    def jumped(series: DailySeries, offset: int, factor: float) -> DailySeries:
        totals, positives = series.totals.copy(), series.positives.copy()
        i = series.index_of(day(offset))
        totals[i], positives[i] = int(totals[i] * factor), int(positives[i] * factor)
        return replace(series, totals=totals, positives=positives)

    base = steady(TREATED, seed=1)

    assert not volume_spike(base, PATCH_DAY, 14)
    assert volume_spike(jumped(base, 2, 5), PATCH_DAY, 14)
    assert volume_spike(jumped(base, 2, 5), PATCH_DAY, 7)
    assert not volume_spike(jumped(base, 10, 5), PATCH_DAY, 7)  # after the horizon
    assert not volume_spike(jumped(base, 2, 0.2), PATCH_DAY, 14)  # a drop isn't a spike
    late = steady(TREATED, first=PATCH_DAY - timedelta(days=5), days=40, seed=1)
    assert not volume_spike(jumped(late, 2, 5), PATCH_DAY, 14)  # under 7 days before


def test_synth_falls_back_with_its_reason() -> None:
    earlier = [Patch("p0", TREATED, day(-10)), Patch("p1", TREATED, PATCH_DAY)]
    rows = estimate_bundle(BUNDLE, world(patches=earlier))
    synth = find(rows, POSITIVE_SHARE, Method.SYNTH)
    assert synth.effect is None
    assert synth.reason == "the game had another patch in the 28 days before"
    assert find(rows, POSITIVE_SHARE, Method.ITS).effect is not None

    four = {a: steady(a, per_day=60, seed=a) for a in DONORS[:4]}
    rows = estimate_bundle(BUNDLE, world(donors=four))
    assert find(rows, POSITIVE_SHARE, Method.SYNTH).reason == (
        "too few comparison games without their own patch (4 of 5 needed)"
    )

    own = steady(TREATED, seed=1)
    weekly = 0.10 * np.sin(2 * np.pi * np.arange(len(own.totals)) / 7)
    patterned = replace(
        own, positives=np.round(np.clip(own.share() + weekly, 0, 1) * own.totals).astype(np.int64)
    )
    rows = estimate_bundle(BUNDLE, world(treated=patterned))
    assert find(rows, POSITIVE_SHARE, Method.SYNTH).reason == (
        "the comparison games didn't track the game before the patch"
    )


def test_seven_day_synth_waits_for_the_comparison_games_fourteen_days() -> None:
    def night(last: date) -> EffectInputs:
        """The inputs a night reads: every series ends at the last complete day."""

        def cut(series: DailySeries) -> DailySeries:
            part = series.span(series.first_day, last)
            assert part is not None
            return part

        full = world()
        return replace(
            full,
            series={a: cut(s) for a, s in full.series.items()},
            established={a: cut(s) for a, s in full.established.items()},
            aspects={a: {c: cut(s) for c, s in m.items()} for a, m in full.aspects.items()},
            donors={a: cut(s) for a, s in full.donors.items()},
            last_complete_day=last,
        )

    waiting = [
        find(estimate_bundle(BUNDLE, night(day(n))), POSITIVE_SHARE, Method.SYNTH, 7)
        for n in (8, 9)
    ]

    assert {row.effect for row in waiting} == {None}
    assert {row.reason for row in waiting} == {
        "the comparison games are checked over the 14 days after the patch, which end on "
        "17 Jun 2026"
    }
    assert waiting[0].fingerprint == waiting[1].fingerprint  # not recomputed every night
    ready = find(estimate_bundle(BUNDLE, night(day(13))), POSITIVE_SHARE, Method.SYNTH, 7)
    assert ready.effect is not None
    assert ready.fingerprint != waiting[0].fingerprint


def test_fourteen_days_only_when_allowed_and_complete() -> None:
    def horizons(rows: list[Estimate]) -> set[int]:
        return {r.horizon for r in rows}

    assert horizons(estimate_bundle(BUNDLE, world())) == {7, 14}
    assert horizons(estimate_bundle(replace(BUNDLE, allow_14=False), world())) == {7}
    assert horizons(estimate_bundle(BUNDLE, world(last_day=PATCH_DAY + timedelta(days=10)))) == {7}


def test_fingerprint_is_the_same_in_every_process() -> None:
    # Each night is a new process, and a set's order follows that process's string hashing.
    code = (
        "from patchpulse.causal.estimate import fingerprint; print(fingerprint('k', "
        "frozenset({'confounded', 'discounts_unknown', 'volume_spike'}), "
        "('p1', frozenset({'binned', 'volume_spike'}))))"
    )
    prints = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for seed in range(1, 7)
    }

    assert len(prints) == 1


def test_fingerprint_changes_with_any_input_and_the_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = fingerprint(np.arange(5), "share", 14)

    assert fingerprint(np.arange(5), "share", 14) == base
    assert fingerprint(np.arange(1, 6), "share", 14) != base
    assert fingerprint(np.arange(5), "share", 7) != base
    monkeypatch.setattr(estimate, "EFFECTS_VERSION", "e0")
    assert fingerprint(np.arange(5), "share", 14) != base


def test_rows_follow_their_own_window() -> None:
    series = steady(TREATED, seed=1, shift=-0.05)
    rows = {r.key: r.fingerprint for r in estimate_bundle(BUNDLE, world(treated=series))}

    def changed(offset: int) -> DailySeries:
        positives = series.positives.copy()
        positives[(day(offset) - FIRST).days] -= 1
        return replace(series, positives=positives)

    inside = {r.key: r.fingerprint for r in estimate_bundle(BUNDLE, world(treated=changed(-5)))}
    outside = {r.key: r.fingerprint for r in estimate_bundle(BUNDLE, world(treated=changed(20)))}
    key = EstimateKey(BUNDLE.bundle_id, POSITIVE_SHARE, Sample.ALL, 14, Method.ITS)
    assert inside[key] != rows[key]
    assert outside[key] == rows[key]  # 20 days after is past the 14-day horizon


def test_its_fingerprint_covers_the_calibration_history() -> None:
    series = steady(TREATED, seed=1, shift=-0.05)
    key = EstimateKey(BUNDLE.bundle_id, POSITIVE_SHARE, Sample.ALL, 14, Method.ITS)
    base = {r.key: r.fingerprint for r in estimate_bundle(BUNDLE, world(treated=series))}
    positives = series.positives.copy()
    positives[(day(-140) - FIRST).days] -= 1
    early = world(treated=replace(series, positives=positives))
    sale = world(treated=series, sale_days=frozenset({day(-100)}))

    # Both are long before the 56-day pre-period, inside the placebo dates' history.
    assert {r.key: r.fingerprint for r in estimate_bundle(BUNDLE, early)}[key] != base[key]
    assert {r.key: r.fingerprint for r in estimate_bundle(BUNDLE, sale)}[key] != base[key]


def test_bundles_to_estimate_skips_unchanged() -> None:
    inputs = world()
    stored = {r.key: r.fingerprint for r in estimate_bundle(BUNDLE, inputs)}

    assert bundles_to_estimate(inputs, stored) == []
    assert bundles_to_estimate(inputs, {}) == [BUNDLE]
    older = {k: v for k, v in stored.items() if k[3] == 7}
    assert bundles_to_estimate(inputs, older) == [BUNDLE]  # the 14-day rows are new


def test_volume_rows_are_percent_changes() -> None:
    rows = estimate_bundle(BUNDLE, world())

    volume = find(rows, VOLUME, Method.ITS)
    assert volume.effect is not None
    assert abs(volume.effect) < 15  # volume didn't change in this world
