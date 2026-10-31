"""Synthetic control: a blend of comparison games as the patched game's twin (design D6)."""

from __future__ import annotations

import time
from datetime import date, timedelta

import numpy as np
import pytest

from patchpulse.causal.its import NoEstimate
from patchpulse.causal.series import DailySeries
from patchpulse.causal.synth import (
    SynthFit,
    conformal_interval,
    fit_weights,
    placebo_p_value,
    synth_effect,
)

FIRST = date(2026, 3, 2)
START = FIRST + timedelta(days=28)
DAYS = 42  # 28 before the patch and 14 from it


def as_series(appid: int, shares: np.ndarray, totals: np.ndarray) -> DailySeries:
    positives = np.round(shares * totals).astype(np.int64)
    return DailySeries(appid, FIRST, positives, totals.astype(np.int64), np.ones(len(totals), bool))


def panel(
    seed: int, *, donors: int = 10, effect: float = -0.05, treated_n: int = 150, donor_n: int = 60
) -> tuple[DailySeries, dict[int, DailySeries]]:
    """Comparison games share a common daily movement; the treated game is a blend of two of them
    plus its own sampling noise, and the effect starts on the patch day."""
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 0.006, DAYS))
    levels = rng.uniform(0.70, 0.90, donors)
    loadings = rng.uniform(0.5, 1.5, donors)
    latent = levels[:, None] + loadings[:, None] * common[None, :]
    pool = {}
    for j in range(donors):
        totals = np.full(DAYS, donor_n)
        shares = rng.binomial(totals, np.clip(latent[j], 0.01, 0.99)) / totals
        pool[100 + j] = as_series(100 + j, shares, totals)
    truth = 0.6 * latent[0] + 0.4 * latent[1] + 0.03
    truth[28:] += effect
    totals = np.full(DAYS, treated_n)
    treated = as_series(1, rng.binomial(totals, np.clip(truth, 0, 1)) / totals, totals)
    return treated, pool


def test_exact_blend_recovers_its_weights() -> None:
    t = np.arange(DAYS)
    a, b, c = 0.8 + 0.02 * np.sin(t / 3), 0.7 + 0.03 * np.cos(t / 5), 0.75 + 0.01 * (t % 7)
    totals = np.full(DAYS, 1_000_000)
    treated = as_series(1, 0.6 * a + 0.4 * b + 0.1, totals)
    donors = {
        10: as_series(10, a, totals),
        11: as_series(11, b, totals),
        12: as_series(12, c, totals),
    }

    weights = fit_weights(0.6 * a[:28] + 0.4 * b[:28] + 0.1, np.column_stack([a, b, c])[:28])
    fit = synth_effect(treated, donors, START, 14, "share")

    assert weights == pytest.approx([0.6, 0.4, 0.0], abs=1e-4)
    assert isinstance(fit, SynthFit)
    assert fit.weights == pytest.approx({10: 0.6, 11: 0.4, 12: 0.0}, abs=1e-4)
    assert fit.effect == pytest.approx(0.0, abs=0.01)
    assert fit.fit_ok


def test_recovers_a_planted_effect() -> None:
    fits = [synth_effect(*panel(seed), START, 14, "share") for seed in range(200)]
    assert all(isinstance(fit, SynthFit) for fit in fits)
    effects = [fit.effect for fit in fits if isinstance(fit, SynthFit)]

    assert float(np.mean(effects)) == pytest.approx(-5.0, abs=0.5)  # unbiased; one panel's sd ~1.4
    assert sum(fit.fit_ok for fit in fits if isinstance(fit, SynthFit)) >= 180


def test_fit_check_fails_when_no_blend_tracks() -> None:
    treated, donors = panel(5, effect=0.0)
    weekly = 0.10 * np.sin(2 * np.pi * np.arange(DAYS) / 7)  # its own pattern, which no donor has
    own = as_series(1, np.clip(treated.share() + weekly, 0, 1), treated.totals)

    fit = synth_effect(own, donors, START, 14, "share")

    assert isinstance(fit, SynthFit)
    assert not fit.fit_ok


def placebo_ps(effect: float, seeds: range) -> np.ndarray:
    ps = []
    for seed in seeds:
        treated, donors = panel(seed, donors=20, effect=effect)
        fit = synth_effect(treated, donors, START, 14, "share")
        assert isinstance(fit, SynthFit)
        ps.append(placebo_p_value(treated, donors, START, 14, "share", fit))
    return np.array(ps)


def test_placebo_p_is_small_for_a_big_effect_and_not_for_none() -> None:
    # With no effect, p <= 0.1 should happen ~10% of the time; the ratio test is a little loose when
    # games differ in volume (17% on 100 such panels, 21 Oct), which the benchmark measures on real
    # data. A -10 effect should almost always stand out.
    assert (placebo_ps(0.0, range(60)) <= 0.1).mean() <= 0.25
    assert (placebo_ps(-0.10, range(60)) <= 0.1).mean() >= 0.8


def test_conformal_covers_a_planted_effect() -> None:
    began = time.monotonic()
    covered = 0
    for seed in range(100):
        treated, donors = panel(1000 + seed)
        interval = conformal_interval(treated, donors, START, 14, "share")
        assert not isinstance(interval, NoEstimate)
        covered += interval[0] <= -5.0 <= interval[1]

    assert 0.80 <= covered / 100 <= 0.98
    print(f"conformal: 100 intervals in {time.monotonic() - began:.1f} s")


def test_conformal_range_holds_its_estimate() -> None:
    # The ends are the first rejected grid points, so even a coarse grid keeps the estimate inside.
    for seed in range(10):
        treated, donors = panel(seed)
        fit = synth_effect(treated, donors, START, 14, "share")
        interval = conformal_interval(treated, donors, START, 14, "share", step=2.0)
        assert isinstance(fit, SynthFit)
        assert not isinstance(interval, NoEstimate)
        assert interval[0] <= fit.effect <= interval[1], (seed, fit.effect, interval)


def test_conformal_says_why_there_is_no_range() -> None:
    treated, donors = panel(5)  # an effect of -5 points

    wider = NoEstimate("interval wider than the grid")
    assert conformal_interval(treated, donors, START, 14, "share", span=2.0) == wider
    assert conformal_interval(treated, donors, START, 14, "share", span=6.0, step=2.0) == wider
    assert conformal_interval(treated, donors, START, 14, "share", level=0.0) == NoEstimate(
        "interval empty: no effect size passed the conformal test"
    )


def test_volume_is_a_percent_change() -> None:
    rng = np.random.default_rng(9)
    common = rng.normal(0, 0.05, DAYS)
    donors = {}
    for j in range(8):
        totals = rng.poisson(100 * np.exp(common + j * 0.1))
        donors[200 + j] = DailySeries(200 + j, FIRST, totals // 2, totals, np.ones(DAYS, bool))
    lift = np.where(np.arange(DAYS) >= 28, 1.3, 1.0)
    totals = rng.poisson(300 * np.exp(common) * lift)
    treated = DailySeries(1, FIRST, totals // 2, totals, np.ones(DAYS, bool))

    fit = synth_effect(treated, donors, START, 14, "volume")

    assert isinstance(fit, SynthFit)
    assert fit.effect == pytest.approx(30.0, abs=8.0)


def test_window_not_complete() -> None:
    treated, donors = panel(5)

    assert synth_effect(treated, donors, START + timedelta(days=5), 14, "share") == NoEstimate(
        "window not complete"
    )


def test_a_game_under_ten_reviews_a_day_gets_no_estimate() -> None:
    # Spec §9.1, as interrupted time series applies it: descriptive analysis only.
    thin, donors = panel(5, treated_n=8)
    enough, _ = panel(5, treated_n=10)

    for kind in ("share", "volume"):
        assert synth_effect(thin, donors, START, 14, kind) == NoEstimate(
            "too few reviews (median under 10 a day)"
        )
        assert isinstance(synth_effect(enough, donors, START, 14, kind), SynthFit)
