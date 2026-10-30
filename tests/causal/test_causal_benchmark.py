"""The semi-synthetic benchmark: known effects injected into real quiet stretches (design D12)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
from causal_fixtures import DONORS, FIRST, TREATED, steady, world

from patchpulse.causal.benchmark import (
    Choice,
    QuietWindow,
    Trial,
    choose_default,
    inject,
    placebo_in_space,
    quiet_windows,
    render_benchmark_report,
    run_trials,
    summarise,
)
from patchpulse.causal.its import ItsResult, calibrated_its_effect
from patchpulse.causal.series import DailySeries


def at(offset: int) -> date:
    return FIRST + timedelta(days=offset)


def test_quiet_windows_avoid_patches_sales_and_gaps() -> None:
    series = steady(TREATED, days=200)
    observed = series.observed.copy()
    observed[30] = False
    series = DailySeries(series.appid, FIRST, series.positives, series.totals, observed)

    starts = {
        w.start
        for w in quiet_windows({TREATED: series}, {TREATED: [at(100)]}, frozenset({at(150)}))
    }

    assert all(isinstance(s, date) for s in starts)
    assert at(73) in starts  # days 45-86: clear of the missing day, the patch and the sale
    assert at(58) not in starts  # its 28 days before reach the missing day 30
    assert at(59) in starts
    assert at(86) in starts
    assert at(87) not in starts  # the patch on day 100 lands in its 14 days after
    assert at(160) not in starts  # the sale on day 150 falls in its 28 days before
    assert at(179) in starts
    assert at(187) not in starts  # its 14 days after run past the series
    for start in starts:  # 14 more days before the fit window are patch-free too
        assert not at(100) - timedelta(days=13) <= start <= at(100) + timedelta(days=42)


def test_injection_moves_the_share_by_the_effect_on_average() -> None:
    big = steady(TREATED, per_day=10_000, days=42)
    rng = np.random.default_rng(0)

    step = inject(big, at(28), -5.0, "step", rng)
    ramp = inject(big, at(28), -5.0, "ramp", rng)

    drop_step = 100 * (big.share() - step.share())[28:]
    drop_ramp = 100 * (big.share() - ramp.share())[28:]
    assert float(drop_step.mean()) == pytest.approx(5.0, abs=0.2)
    assert float(drop_ramp[0]) == pytest.approx(5.0 / 7, abs=0.2)
    assert float(drop_ramp[6:].mean()) == pytest.approx(5.0, abs=0.2)


def test_injection_keeps_totals() -> None:
    base = steady(TREATED, days=42)

    hit = inject(base, at(28), -10.0, "step", np.random.default_rng(1))

    assert (hit.totals == base.totals).all()
    assert (hit.positives[:28] == base.positives[:28]).all()
    assert (hit.positives <= hit.totals).all()
    assert (hit.positives[28:] <= base.positives[28:]).all()
    assert inject(base, at(28), 0.0, "step", np.random.default_rng(1)).positives.tolist() == (
        base.positives.tolist()
    )


def trial(method: str, effect: float, estimate: float, lo: float, hi: float, **kw: object) -> Trial:
    values: dict[str, object] = {
        "appid": TREATED,
        "start": at(80),
        "effect_pts": effect,
        "truth": effect,
        "shape": "step",
        "tier": "over 150",
        "outcome": "positive_share",
        "horizon": 14,
        "method": method,
        "estimate": estimate,
        "lo90": lo,
        "hi90": hi,
        "p_value": None,
        "fallback": False,
    }
    values.update(kw)
    return Trial(**values)  # type: ignore[arg-type]


def test_metrics_on_hand_made_trials() -> None:
    trials = [
        trial("its", -5.0, -4.0, -6.0, -2.0),  # covers, detects
        trial("its", -5.0, -7.0, -8.0, -6.0),  # misses, detects
        trial("its", -5.0, -5.0, -9.0, 1.0),  # covers, doesn't detect
        trial("its", 0.0, 1.0, 0.5, 1.5),  # a false alarm
        trial("its", 0.0, 0.0, -1.0, 1.0),
    ]

    summary = summarise(trials)

    hit = summary[("its", "positive_share", 14, "over 150", -5.0, "step")]
    assert hit.bias == pytest.approx((1.0 - 2.0 + 0.0) / 3)
    assert hit.rmse == pytest.approx(np.sqrt((1 + 4 + 0) / 3))
    assert hit.coverage == pytest.approx(2 / 3)
    assert hit.power == pytest.approx(2 / 3)
    assert hit.fpr is None
    none = summary[("its", "positive_share", 14, "over 150", 0.0, "step")]
    assert none.fpr == pytest.approx(0.5)
    assert none.power is None
    assert (hit.n, none.n) == (3, 2)


def calibrated(method: str, *, miss: int, alarms: int, error: float) -> list[Trial]:
    """20 trials at -5 (missing `miss`) and 20 at 0 (raising `alarms`), estimates off by `error`."""
    trials = []
    for i in range(20):
        lo, hi = (-10.0, -9.0) if i < miss else (-8.0, -2.0)
        trials.append(trial(method, -5.0, -5.0 + error, lo, hi))
        zlo, zhi = (0.5, 1.0) if i < alarms else (-1.0, 1.0)
        trials.append(trial(method, 0.0, error, zlo, zhi))
    return trials


def test_choose_default_follows_the_rule() -> None:
    both = calibrated("its", miss=2, alarms=1, error=1.0) + calibrated(
        "synth", miss=2, alarms=1, error=0.5
    )
    assert choose_default(both).method == "synth"  # both eligible: the lower RMSE

    only_its = calibrated("its", miss=2, alarms=1, error=1.0) + calibrated(
        "synth",
        miss=6,
        alarms=1,
        error=0.1,  # coverage 70%: not eligible however accurate
    )
    assert choose_default(only_its).method == "its"

    noisy = calibrated("its", miss=2, alarms=4, error=0.1) + calibrated(  # 20% false alarms
        "synth",
        miss=0,
        alarms=0,
        error=1.0,  # 100% coverage: outside 85-95%
    )
    choice = choose_default(noisy)
    assert isinstance(choice, Choice)
    assert (choice.method, choice.validated) == ("its", False)
    assert "no method" in choice.reason


def test_trials_cover_each_cell_and_method() -> None:
    inputs = world()
    series = {TREATED: inputs.series[TREATED]}
    windows = [QuietWindow(TREATED, at(110)), QuietWindow(TREATED, at(115))]
    groups = {a: inputs.games[a].genres for a in [*DONORS, TREATED]}

    trials = run_trials(
        windows,
        series,
        inputs.donors,
        groups,
        {},
        sale_days=frozenset(),
        effects=(0.0, -5.0),
        shapes=("step",),
        per_cell=3,
        seed=7,
        horizons=(14,),
        outcomes=("positive_share",),
    )

    cells = {(t.method, t.effect_pts) for t in trials}
    assert cells == {("its", 0.0), ("its", -5.0), ("synth", 0.0), ("synth", -5.0)}
    assert len(trials) == 12
    assert all(t.estimate is None or np.isfinite(t.estimate) for t in trials)
    assert all(t.estimate is not None for t in trials if t.method == "its")  # 11-12 quiet weeks
    assert {t.truth for t in trials if t.effect_pts == -5.0} == {-5.0}


def test_benchmark_serves_the_calibrated_interval() -> None:
    inputs = world()
    series = inputs.series[TREATED]
    sale = frozenset({at(60)})  # rules out 3 of the 13 placebo dates, leaving 10

    trials = run_trials(
        [QuietWindow(TREATED, at(130))],
        {TREATED: series},
        inputs.donors,
        {a: inputs.games[a].genres for a in [*DONORS, TREATED]},
        {},
        effects=(0.0,),
        shapes=("step",),
        per_cell=1,
        seed=1,
        horizons=(14,),
        outcomes=("positive_share",),
        sale_days=sale,
    )

    served = calibrated_its_effect(series, at(130), 14, [], "share", busy_days=sale)
    assert isinstance(served, ItsResult)
    (its,) = [t for t in trials if t.method == "its"]
    assert (its.estimate, its.lo90, its.hi90) == (served.effect, served.lo90, served.hi90)


def test_ramp_truth_is_the_horizons_average() -> None:
    inputs = world()
    trials = run_trials(
        [QuietWindow(TREATED, at(110))],
        {TREATED: inputs.series[TREATED]},
        inputs.donors,
        {a: inputs.games[a].genres for a in [*DONORS, TREATED]},
        {},
        sale_days=frozenset(),
        effects=(-7.0,),
        shapes=("ramp",),
        per_cell=1,
        seed=1,
        horizons=(7, 14),
        outcomes=("positive_share",),
    )

    truths = {t.horizon: t.truth for t in trials}
    assert truths[7] == pytest.approx(-7.0 * 4 / 7)  # (1 + ... + 7) / 7 / 7 of -7
    assert truths[14] == pytest.approx(-7.0 * 11 / 14)


def test_placebo_in_space_runs_on_quiet_comparison_games() -> None:
    inputs = world()
    groups = {a: inputs.games[a].genres for a in DONORS}

    trials = placebo_in_space(inputs.donors, [at(120)], groups, {}, seed=3, sale_days=frozenset())

    assert {t.appid for t in trials} <= set(DONORS)
    assert {t.effect_pts for t in trials} == {0.0}
    assert {t.method for t in trials} == {"its", "synth"}


def test_report_states_the_rule_and_the_limitations() -> None:
    trials = calibrated("its", miss=2, alarms=1, error=1.0) + calibrated(
        "synth", miss=2, alarms=1, error=0.5
    )

    report = render_benchmark_report(
        summarise(trials),
        choose_default(trials),
        trials,
        [],
        {"seed": 7, "per_cell": 20, "data_version": 15, "effects_version": "e1"},
    )

    for rule_part in ("85%", "95%", "10%"):
        assert rule_part in report
    assert "Default method: synth" in report
    assert "series,and" not in report  # the rule's last sentence reads cleanly
    assert "intervals weren't validated" in report
    assert "calibrated on quiet dates in the game's own past" in report
    for limitation in ("Aspect", "rules classifier", "30 Sep 2026"):
        assert limitation in report
