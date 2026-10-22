"""Interrupted time series: the game's own history projected past the patch (design D5)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
import statsmodels.api as sm

from patchpulse.causal.its import (
    POSITIVE_SHARE,
    VOLUME,
    ItsResult,
    NoEstimate,
    aspect_outcome,
    fixed_b_critical,
    hac_cov,
    its_effect,
    newey_west_lag,
)
from patchpulse.causal.series import DailySeries

FIRST = date(2026, 3, 2)  # a Monday
START = FIRST + timedelta(days=56)


def simulate(
    *,
    per_day: int = 150,
    base: float = 0.80,
    shift: float = 0.0,
    earlier: tuple[date, float] | None = None,
    days: int = 70,
    first: date = FIRST,
    seed: int = 7,
) -> DailySeries:
    rng = np.random.default_rng(seed)
    swings = [0.02, 0.01, 0.0, -0.01, -0.02, 0.01, -0.01]
    positives, totals = [], []
    for offset in range(days):
        day = first + timedelta(days=offset)
        p = base + swings[day.weekday()]
        if earlier is not None and day >= earlier[0]:
            p += earlier[1]
        if day >= START:
            p += shift
        positives.append(rng.binomial(per_day, p))
        totals.append(per_day)
    return DailySeries(
        appid=1,
        first_day=first,
        positives=np.array(positives, dtype=np.int64),
        totals=np.array(totals, dtype=np.int64),
        observed=np.ones(days, dtype=bool),
    )


def test_outcome_names() -> None:
    assert (POSITIVE_SHARE, VOLUME, aspect_outcome("performance")) == (
        "positive_share",
        "volume",
        "aspect:performance",
    )


def test_hac_matches_statsmodels() -> None:
    rng = np.random.default_rng(1)
    n = 70
    x = np.column_stack([np.ones(n), np.arange(n), rng.normal(size=(n, 3))])
    y = x @ np.array([1.0, 0.01, 0.5, -0.2, 0.3]) + rng.normal(size=n)
    weights = rng.integers(20, 200, size=n).astype(float)
    reference = sm.WLS(y, x, weights=weights).fit(
        cov_type="HAC", cov_kwds={"maxlags": 7, "use_correction": True}
    )

    root = np.sqrt(weights)
    beta = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)[0]
    ours = hac_cov(x * root[:, None], (y - x @ beta) * root, lag=7)

    assert np.allclose(ours, reference.cov_params(), rtol=1e-8, atol=1e-12)


def test_recovers_a_planted_level_shift() -> None:
    result = its_effect(simulate(shift=-0.05), START, 14, [], "share")

    assert isinstance(result, ItsResult)
    assert result.effect == pytest.approx(-5.0, abs=1.5)
    assert result.lo90 <= -5.0 <= result.hi90
    assert not result.binned


def test_an_earlier_patch_in_the_pre_period_doesnt_bias_it() -> None:
    earlier = START - timedelta(days=20)
    series = simulate(shift=-0.05, earlier=(earlier, 0.04))

    result = its_effect(series, START, 14, [earlier], "share")

    assert isinstance(result, ItsResult)
    assert result.effect == pytest.approx(-5.0, abs=1.5)
    assert result.lo90 <= -5.0 <= result.hi90


def test_lag_rule_and_critical_values() -> None:
    assert [newey_west_lag(n) for n in (40, 70, 100, 300)] == [3, 3, 4, 5]
    assert fixed_b_critical(0.0) == pytest.approx(1.6449)
    assert fixed_b_critical(0.1) == pytest.approx(1.866, abs=0.001)
    assert fixed_b_critical(0.05) < fixed_b_critical(0.1) < fixed_b_critical(0.2)


def test_intervals_cover_about_90_percent() -> None:
    # Independent binomial days: the interval should hold a planted -5 about 90% of the time.
    covered = 0
    for seed in range(200):
        result = its_effect(simulate(shift=-0.05, seed=100 + seed), START, 14, [], "share")
        assert isinstance(result, ItsResult)
        covered += result.lo90 <= -5.0 <= result.hi90
    assert 0.85 <= covered / 200 <= 0.95


def test_volume_is_a_percent_change() -> None:
    effects, covered = [], 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        totals = np.concatenate([rng.poisson(200, 56), rng.poisson(260, 14)]).astype(np.int64)
        series = DailySeries(1, FIRST, totals // 2, totals, np.ones(70, dtype=bool))
        result = its_effect(series, START, 14, [], "volume")
        assert isinstance(result, ItsResult)
        effects.append(result.effect)
        covered += result.lo90 <= 30.0 <= result.hi90

    assert float(np.mean(effects)) == pytest.approx(30.0, abs=2.0)  # a planted 30% rise
    assert covered / 200 >= 0.85


def test_low_volume_days_are_binned() -> None:
    result = its_effect(simulate(per_day=12, shift=-0.05), START, 14, [], "share")

    assert isinstance(result, ItsResult)
    assert result.binned
    assert np.isfinite([result.effect, result.lo90, result.hi90]).all()


def test_short_pre_period_gives_a_reason() -> None:
    late = simulate(first=START - timedelta(days=20), days=34)

    assert its_effect(late, START, 14, [], "share") == NoEstimate(
        "pre-period under 28 days with reviews"
    )


def test_other_reasons() -> None:
    assert its_effect(simulate(days=60), START, 7, [], "share") == NoEstimate("window not complete")
    assert its_effect(simulate(per_day=8), START, 14, [], "share") == NoEstimate(
        "too few reviews (median under 10 a day)"
    )
    flat = DailySeries(1, FIRST, np.zeros(70, np.int64), np.zeros(70, np.int64), np.ones(70, bool))
    assert its_effect(flat, START, 14, [], "share") == NoEstimate(
        "too few reviews (median under 10 a day)"
    )
