"""The semi-synthetic benchmark: known effects injected into real quiet stretches (design D12)."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest
from causal_fixtures import DONORS, FIRST, TREATED, steady, world

from patchpulse.causal import benchmark
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
    spike_verdict,
    summarise,
)
from patchpulse.causal.its import ItsResult, NoEstimate, calibrated_its_effect
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
        "volume_spike": False,
        "pair": 0,
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
        trials.append(trial(method, -5.0, -5.0 + error, lo, hi, pair=2 * i))
        zlo, zhi = (0.5, 1.0) if i < alarms else (-1.0, 1.0)
        trials.append(trial(method, 0.0, error, zlo, zhi, pair=2 * i + 1))
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
    # Each injected series and horizon: one row per method, sharing a pair id.
    pairs: dict[int, list[str]] = {}
    for t in trials:
        pairs.setdefault(t.pair, []).append(t.method)
    assert sorted(sorted(methods) for methods in pairs.values()) == [["its", "synth"]] * 6
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


def test_benchmark_steps_at_earlier_bundles_as_the_nightly_estimates_do() -> None:
    inputs = world()
    series = inputs.series[TREATED]
    patches = [at(80), at(83)]  # one bundle (within 7 days) inside the 56-day pre-period

    trials = run_trials(
        [QuietWindow(TREATED, at(130))],
        {TREATED: series},
        inputs.donors,
        {a: inputs.games[a].genres for a in [*DONORS, TREATED]},
        {TREATED: patches},
        effects=(0.0,),
        shapes=("step",),
        per_cell=1,
        seed=1,
        horizons=(14,),
        outcomes=("positive_share",),
        sale_days=frozenset(),
    )

    busy = frozenset(patches)
    served = calibrated_its_effect(series, at(130), 14, [at(80)], "share", busy_days=busy)
    per_patch = calibrated_its_effect(series, at(130), 14, patches, "share", busy_days=busy)
    assert isinstance(served, ItsResult)
    assert isinstance(per_patch, ItsResult)
    assert served.effect != per_patch.effect
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


def test_trials_carry_the_sites_volume_spike_flag() -> None:
    inputs = world()
    base = inputs.series[TREATED]
    totals, positives = base.totals.copy(), base.positives.copy()
    i = base.index_of(at(112))  # the third day of the window starting at(110)
    totals[i], positives[i] = totals[i] * 5, positives[i] * 5
    jumped = replace(base, totals=totals, positives=positives)
    groups = {a: inputs.games[a].genres for a in [*DONORS, TREATED]}

    def flags(series: DailySeries) -> set[bool]:
        trials = run_trials(
            [QuietWindow(TREATED, at(110))],
            {TREATED: series},
            inputs.donors,
            groups,
            {},
            sale_days=frozenset(),
            effects=(0.0,),
            volume_effects=(-25.0,),  # thinned, the jump is still a spike
            shapes=("step",),
            per_cell=1,
            seed=7,
            horizons=(7, 14),
        )
        assert {t.outcome for t in trials} == {"positive_share", "volume"}
        return {t.volume_spike for t in trials}

    assert flags(jumped) == {True}
    assert flags(base) == {False}


def test_placebo_in_space_runs_on_quiet_comparison_games() -> None:
    inputs = world()
    groups = {a: inputs.games[a].genres for a in DONORS}

    trials = placebo_in_space(inputs.donors, [at(120)], groups, {}, seed=3, sale_days=frozenset())

    assert {t.appid for t in trials} <= set(DONORS)
    assert {t.effect_pts for t in trials} == {0.0}
    assert {t.method for t in trials} == {"its", "synth"}
    assert {t.volume_spike for t in trials} == {False}  # steady comparison games


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


def test_report_splits_the_metrics_by_volume_spike() -> None:
    calm = calibrated("its", miss=2, alarms=1, error=1.0)
    spiky = [
        replace(t, volume_spike=True) for t in calibrated("its", miss=10, alarms=10, error=3.0)
    ]
    trials = calm + spiky

    report = render_benchmark_report(
        summarise(trials), choose_default(trials), trials, [], {"seed": 7}
    )

    assert "## Windows with a volume spike" in report
    assert "50% of the trials" in report
    assert "| Positive share | its | no | 40 | +1.00 | 1.00 | 90% | 5% |" in report
    assert "| Positive share | its | yes | 40 | +3.00 | 3.00 | 50% | 50% |" in report
    # Phase 5's V4 reads this verdict (Seif's option C, 25 Oct): 10 of 20 spike trials at 0
    # raised an alarm, so causal wording stays blocked on spike windows.
    assert (
        "Causal wording on volume-spike windows (Phase 5): blocked. The default method raised "
        "false alarms on 50% of 20 spike-window trials at an injected 0 (allowed: at most 10% "
        "over at least 20 trials)." in report
    )


def test_synth_without_a_range_is_scored_as_served(monkeypatch: pytest.MonkeyPatch) -> None:
    # The site shows interrupted time series where synthetic control has no 90% range.
    no_range = NoEstimate("interval wider than the grid")
    monkeypatch.setattr(benchmark, "conformal_interval", lambda *args, **kwargs: no_range)
    inputs = world()
    groups = {a: inputs.games[a].genres for a in [*DONORS, TREATED]}

    trials = run_trials(
        [QuietWindow(TREATED, at(110))],
        {TREATED: inputs.series[TREATED]},
        inputs.donors,
        groups,
        {},
        sale_days=frozenset(),
        effects=(-5.0,),
        shapes=("step",),
        per_cell=1,
        seed=7,
        horizons=(14,),
        outcomes=("positive_share",),
    )

    its, synth = sorted(trials, key=lambda t: t.method)
    assert (synth.estimate, synth.lo90, synth.hi90) == (its.estimate, its.lo90, its.hi90)
    assert synth.fallback


def test_report_compares_the_methods_on_the_same_trials() -> None:
    # The rule uses each method's own trials with an estimate; interrupted time series has none
    # where the quiet history is too short, so the two are also compared on the shared trials.
    no_history = {0, 2, 4, 6}  # four of the -5 trials, two of them synthetic control's misses
    its = [
        replace(t, estimate=None, lo90=None, hi90=None) if t.pair in no_history else t
        for t in calibrated("its", miss=2, alarms=1, error=1.0)
    ]
    trials = its + calibrated("synth", miss=2, alarms=1, error=0.5)

    report = render_benchmark_report(
        summarise(trials), choose_default(trials), trials, [], {"seed": 7}
    )

    assert "## Both methods on the same trials" in report
    assert "36 positive-share trials where both methods had an estimate" in report
    assert "| its | 36 | 100% | 5% | 1.00 |" in report
    assert "| synth | 36 | 100% | 5% | 0.50 |" in report  # 90% coverage on its own 40


def test_spike_verdict_needs_twenty_trials_and_ten_percent() -> None:
    def at_zero(
        n: int,
        alarms: int,
        *,
        method: str = "synth",
        outcome: str = "positive_share",
        spike: bool = True,
    ) -> list[Trial]:
        bounds = [(0.2, 0.9) if i < alarms else (-1.0, 1.0) for i in range(n)]
        return [
            trial(method, 0.0, 0.5, lo, hi, outcome=outcome, volume_spike=spike)
            for lo, hi in bounds
        ]

    unscored = replace(trial("synth", 0.0, 0.5, -1.0, 1.0), estimate=None, lo90=None, hi90=None)
    noise = [
        *at_zero(5, 5, spike=False),  # no spike: not counted
        *[trial("synth", -5.0, -4.0, -6.0, -2.0, volume_spike=True) for _ in range(5)],  # not 0
        *at_zero(5, 5, method="its"),  # another method
        *at_zero(5, 5, outcome="volume"),  # another outcome
        replace(unscored, volume_spike=True),  # no estimate
    ]

    few = spike_verdict(at_zero(19, 0) + noise, "synth")
    ok = spike_verdict(at_zero(20, 2) + noise, "synth")
    high = spike_verdict(at_zero(20, 3) + noise, "synth")
    none = spike_verdict(noise, "synth")

    assert (few.trials, few.false_alarms, few.allowed) == (19, 0.0, False)
    assert ok.trials == 20
    assert ok.false_alarms == pytest.approx(0.10)
    assert ok.allowed
    assert high.trials == 20
    assert high.false_alarms == pytest.approx(0.15)
    assert not high.allowed
    assert (none.trials, none.false_alarms, none.allowed) == (0, None, False)
