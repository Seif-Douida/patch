"""The semi-synthetic benchmark (design D12, spec §9.4 and §11.3): known effects injected into real
quiet stretches of review history, so each method's error and interval behaviour can be measured.

- **Quiet windows:** 42-day stretches (28 before a start day, 14 from it) with no treatment patch
  of that game in them or in the 14 days before, no Steam sale, and data on every day.
- **Injection:** on the start day, each positive review turns negative with the probability that
  removes the effect from that day's share on average (a step, or a ramp reaching it on day 7);
  totals stay as they were, so the noise stays real. Volume effects thin each day's reviews.
- **Trials:** both methods on each injected window, as the nightly stage runs them (synthetic
  control falls back to interrupted time series when it would); the effect-0 trials on tracked
  games are the placebo-in-time runs. Placebo-in-space runs both methods on comparison games at
  the tracked games' real patch dates.
- **The default** follows ADR-007's rule, fixed before any run.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np

from patchpulse.causal.bundles import Bundle, Patch, bundle_patches
from patchpulse.causal.estimate import volume_spike
from patchpulse.causal.its import (
    POSITIVE_SHARE,
    VOLUME,
    ItsResult,
    Kind,
    NoEstimate,
    calibrated_its_effect,
)
from patchpulse.causal.rules import FIT_DAYS, MIN_DONORS, POST_DAYS, eligible_donors, genre_group
from patchpulse.causal.series import DailySeries
from patchpulse.causal.synth import SynthFit, conformal_interval, placebo_p_value, synth_effect

LINGER_DAYS = 14  # no patch this long before a window's fit days either
RAMP_DAYS = 7
SHARE_EFFECTS = (0.0, -2.0, -5.0, -10.0)  # points
VOLUME_EFFECTS = (0.0, -10.0, -25.0)  # percent
SHAPES = ("step", "ramp")
COVERAGE_RANGE = (0.85, 0.95)
MAX_FPR = 0.10


@dataclass(frozen=True)
class QuietWindow:
    appid: int
    start: date


@dataclass(frozen=True)
class Trial:
    appid: int
    start: date
    effect_pts: float  # injected: points (share) or percent (volume)
    truth: float  # the injected effect averaged over the horizon
    shape: str
    tier: str
    outcome: str
    horizon: int
    method: str  # "its", or "synth" as served (with its fallback to "its")
    estimate: float | None
    lo90: float | None  # None: no interval (wider than the grid)
    hi90: float | None
    p_value: float | None
    fallback: bool  # synthetic control fell back to interrupted time series
    volume_spike: bool  # the site's flag on the (injected) window: a real event the calendar missed


@dataclass(frozen=True)
class Metrics:
    bias: float | None
    rmse: float | None
    coverage: float | None
    fpr: float | None  # at an injected 0: intervals excluding 0
    power: float | None  # at a non-zero effect: intervals excluding 0
    n: int


@dataclass(frozen=True)
class Choice:
    method: str
    validated: bool
    reason: str
    table: dict[str, dict[str, float]]


def _clean(series: DailySeries, first: date, last: date) -> bool:
    part = series.span(first, last)
    return part is not None and bool(part.observed.all())


def quiet_windows(
    series: Mapping[int, DailySeries],
    patches: Mapping[int, Sequence[date]],
    sale_days: frozenset[date],
    *,
    pre: int = FIT_DAYS,
    post: int = POST_DAYS,
) -> list[QuietWindow]:
    windows: list[QuietWindow] = []
    for appid in sorted(series):
        data = series[appid]
        days = sorted(patches.get(appid, ()))
        start = data.first_day + timedelta(days=pre)
        while start + timedelta(days=post - 1) <= data.last_day:
            first, last = start - timedelta(days=pre), start + timedelta(days=post - 1)
            calm = first - timedelta(days=LINGER_DAYS)
            if (
                not any(calm <= d <= last for d in days)
                and not any(first <= d <= last for d in sale_days)
                and _clean(data, first, last)
            ):
                windows.append(QuietWindow(appid, start))
            start += timedelta(days=1)
    return windows


def _weights(shape: str, days: int) -> np.ndarray:
    if shape == "step":
        return np.ones(days)
    return np.minimum(1.0, np.arange(1, days + 1) / RAMP_DAYS)


def inject(
    series: DailySeries,
    start: date,
    effect_pts: float,
    shape: str,
    rng: np.random.Generator,
    *,
    days: int = POST_DAYS,
) -> DailySeries:
    """The series with an effect on the positive share from `start` for `days` days."""
    if effect_pts == 0:
        return series
    positives = series.positives.copy()
    offset = series.index_of(start)
    for k, weight in enumerate(_weights(shape, days)):
        i = offset + k
        if i >= len(positives) or series.totals[i] == 0:
            continue
        share = positives[i] / series.totals[i]
        change = abs(effect_pts) / 100 * weight
        if effect_pts < 0 and share > 0:
            positives[i] -= rng.binomial(positives[i], min(1.0, change / share))
        elif effect_pts > 0 and share < 1:
            negatives = series.totals[i] - positives[i]
            positives[i] += rng.binomial(negatives, min(1.0, change / (1 - share)))
    return replace(series, positives=positives)


def inject_volume(
    series: DailySeries,
    start: date,
    effect_pct: float,
    shape: str,
    rng: np.random.Generator,
    *,
    days: int = POST_DAYS,
) -> DailySeries:
    """The series with fewer reviews from `start`: each review kept with probability 1 + effect."""
    if effect_pct == 0:
        return series
    positives, totals = series.positives.copy(), series.totals.copy()
    offset = series.index_of(start)
    for k, weight in enumerate(_weights(shape, days)):
        i = offset + k
        if i >= len(totals):
            continue
        keep = max(0.0, 1 + effect_pct / 100 * weight)
        kept_positive = rng.binomial(positives[i], keep)
        kept_negative = rng.binomial(totals[i] - positives[i], keep)
        positives[i], totals[i] = kept_positive, kept_positive + kept_negative
    return replace(series, positives=positives, totals=totals)


def _truth(effect: float, shape: str, horizon: int, outcome: str) -> float:
    mean = float(_weights(shape, horizon).mean())
    if outcome == VOLUME:  # the average multiplicative change, as the methods report it
        return 100 * (float(np.exp(np.mean(np.log1p(effect / 100 * _weights(shape, horizon))))) - 1)
    return effect * mean


def _tier(series: DailySeries, start: date) -> str:
    median = series.median_total(start - timedelta(days=FIT_DAYS), start + timedelta(POST_DAYS - 1))
    return "under 50" if median < 50 else "50-150" if median <= 150 else "over 150"


Row = tuple[str, float | None, float | None, float | None, float | None, bool]


def _estimate(
    treated: DailySeries,
    pool: Mapping[int, DailySeries],
    groups: Mapping[int, Sequence[str]],
    patch_days: Mapping[int, Sequence[date]],
    start: date,
    horizon: int,
    kind: Kind,
    sale_days: frozenset[date],
) -> list[Row]:
    """(method, estimate, lo, hi, p-value, fallback) for interrupted time series and for synthetic
    control as served (falling back to interrupted time series when it would)."""
    own = patch_days.get(treated.appid, ())
    # Steps at the earlier bundles' starts, as the nightly estimates fit them.
    bundles = bundle_patches([Patch(f"p{i}", treated.appid, d) for i, d in enumerate(own)])
    earlier = [b.start for b in bundles if b.start < start]
    busy = frozenset(own) | sale_days
    its = calibrated_its_effect(treated, start, horizon, earlier, kind, busy_days=busy)
    values = (its.effect, its.lo90, its.hi90) if isinstance(its, ItsResult) else (None, None, None)
    its_row: Row = ("its", *values, None, False)
    synth_row: Row = ("synth", *values, None, True)
    bundle = Bundle(treated.appid, start, ("trial",), True)
    candidates = {a: genre_group(g) for a, g in groups.items() if a in pool}
    group = genre_group(groups[treated.appid])
    chosen = eligible_donors(bundle, group, candidates, patch_days, pool)
    if len(chosen) >= MIN_DONORS:
        donors = {a: pool[a] for a in chosen}
        fit = synth_effect(treated, donors, start, horizon, kind)
        if isinstance(fit, SynthFit) and fit.fit_ok:
            interval = conformal_interval(treated, donors, start, horizon, kind)
            # Without a range the site serves the fallback instead.
            if not isinstance(interval, NoEstimate):
                p_value = placebo_p_value(treated, donors, start, horizon, kind, fit)
                synth_row = ("synth", fit.effect, *interval, p_value, False)
    return [its_row, synth_row]


def run_trials(
    windows: Sequence[QuietWindow],
    series: Mapping[int, DailySeries],
    donors: Mapping[int, DailySeries],
    groups: Mapping[int, Sequence[str]],
    patch_days: Mapping[int, Sequence[date]],
    *,
    sale_days: frozenset[date],
    effects: Sequence[float] = SHARE_EFFECTS,
    volume_effects: Sequence[float] = VOLUME_EFFECTS,
    shapes: Sequence[str] = SHAPES,
    per_cell: int = 200,
    seed: int,
    horizons: Sequence[int] = (7, 14),
    outcomes: Sequence[str] = (POSITIVE_SHARE, VOLUME),
) -> list[Trial]:
    """`per_cell` windows drawn (with replacement, seeded) for every outcome, effect and shape;
    both methods at each horizon on the same injected series."""
    rng = np.random.default_rng(seed)
    trials: list[Trial] = []
    for outcome in outcomes:
        kind: Kind = "share" if outcome == POSITIVE_SHARE else "volume"
        for effect in effects if outcome == POSITIVE_SHARE else volume_effects:
            for shape in shapes:
                for index in rng.integers(0, len(windows), size=per_cell):
                    window = windows[int(index)]
                    base = series[window.appid]
                    hit = (
                        inject(base, window.start, effect, shape, rng)
                        if kind == "share"
                        else inject_volume(base, window.start, effect, shape, rng)
                    )
                    pool = {a: s for a, s in {**donors, **series}.items() if a != window.appid}
                    tier = _tier(base, window.start)
                    for horizon in horizons:
                        rows = _estimate(
                            hit, pool, groups, patch_days, window.start, horizon, kind, sale_days
                        )
                        spike = volume_spike(hit, window.start, horizon)
                        for method, estimate, lo, hi, p_value, fallback in rows:
                            trials.append(
                                Trial(
                                    window.appid,
                                    window.start,
                                    effect,
                                    _truth(effect, shape, horizon, outcome),
                                    shape,
                                    tier,
                                    outcome,
                                    horizon,
                                    method,
                                    estimate,
                                    lo,
                                    hi,
                                    p_value,
                                    fallback,
                                    spike,
                                )
                            )
    return trials


def placebo_in_space(
    donors: Mapping[int, DailySeries],
    patch_dates: Sequence[date],
    groups: Mapping[int, Sequence[str]],
    patch_days: Mapping[int, Sequence[date]],
    *,
    seed: int,
    sale_days: frozenset[date],
    horizon: int = POST_DAYS,
) -> list[Trial]:
    """Both methods on each comparison game that is quiet at a tracked patch's date; no injection,
    so any interval excluding 0 is a false alarm on real data."""
    del seed  # deterministic: no draws; kept for a uniform signature
    trials: list[Trial] = []
    for start in sorted(set(patch_dates)):
        for appid in sorted(donors):
            data = donors[appid]
            first = start - timedelta(days=FIT_DAYS + LINGER_DAYS)
            last = start + timedelta(days=horizon - 1)
            if any(first <= d <= last for d in patch_days.get(appid, ())):
                continue
            if not _clean(data, start - timedelta(days=FIT_DAYS), last):
                continue
            pool = {a: s for a, s in donors.items() if a != appid}
            spike = volume_spike(data, start, horizon)
            for method, estimate, lo, hi, p_value, fallback in _estimate(
                data, pool, groups, patch_days, start, horizon, "share", sale_days
            ):
                trials.append(
                    Trial(
                        appid,
                        start,
                        0.0,
                        0.0,
                        "none",
                        _tier(data, start),
                        POSITIVE_SHARE,
                        horizon,
                        method,
                        estimate,
                        lo,
                        hi,
                        p_value,
                        fallback,
                        spike,
                    )
                )
    return trials


def _covers(trial: Trial, value: float) -> bool:
    lo = -math.inf if trial.lo90 is None else trial.lo90
    hi = math.inf if trial.hi90 is None else trial.hi90
    return lo <= value <= hi


def _metrics(trials: Sequence[Trial]) -> Metrics:
    scored = [t for t in trials if t.estimate is not None]
    errors = np.array([t.estimate - t.truth for t in scored if t.estimate is not None])
    effect = trials[0].effect_pts
    excludes = [not _covers(t, 0.0) for t in scored]
    return Metrics(
        bias=float(errors.mean()) if len(errors) else None,
        rmse=float(np.sqrt((errors**2).mean())) if len(errors) else None,
        coverage=float(np.mean([_covers(t, t.truth) for t in scored])) if scored else None,
        fpr=float(np.mean(excludes)) if effect == 0 and scored else None,
        power=float(np.mean(excludes)) if effect != 0 and scored else None,
        n=len(scored),
    )


def summarise(
    trials: Sequence[Trial],
) -> dict[tuple[str, str, int, str, float, str], Metrics]:
    """Metrics per (method, outcome, horizon, tier, effect, shape)."""
    groups: dict[tuple[str, str, int, str, float, str], list[Trial]] = defaultdict(list)
    for t in trials:
        groups[(t.method, t.outcome, t.horizon, t.tier, t.effect_pts, t.shape)].append(t)
    return {key: _metrics(group) for key, group in sorted(groups.items())}


def choose_default(trials: Sequence[Trial]) -> Choice:
    """ADR-007's rule, on the positive-share trials with an estimate, every tier and horizon
    pooled: coverage of the non-zero effects within 85-95% and a false-alarm rate at 0 of at most
    10% make a method eligible; the eligible one with the lower RMSE is the default; with none,
    interrupted time series, unvalidated."""
    table: dict[str, dict[str, float]] = {}
    for method in sorted({t.method for t in trials}):
        mine = [
            t
            for t in trials
            if t.method == method and t.outcome == POSITIVE_SHARE and t.estimate is not None
        ]
        hits = [t for t in mine if t.effect_pts != 0]
        nulls = [t for t in mine if t.effect_pts == 0]
        errors = [t.estimate - t.truth for t in mine if t.estimate is not None]
        table[method] = {
            "coverage": float(np.mean([_covers(t, t.truth) for t in hits])) if hits else math.nan,
            "fpr": float(np.mean([not _covers(t, 0.0) for t in nulls])) if nulls else math.nan,
            "rmse": float(np.sqrt(np.mean(np.square(errors)))) if errors else math.nan,
        }
    eligible = [
        m
        for m, row in table.items()
        if COVERAGE_RANGE[0] <= row["coverage"] <= COVERAGE_RANGE[1] and row["fpr"] <= MAX_FPR
    ]
    if not eligible:
        return Choice(
            "its", False, "no method met the rule: interrupted time series, unvalidated", table
        )
    best = min(eligible, key=lambda m: table[m]["rmse"])
    return Choice(best, True, f"eligible: {', '.join(eligible)}; lowest RMSE: {best}", table)


def _fmt(value: float | None, percent: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{100 * value:.0f}%" if percent else f"{value:+.2f}"


Pooled = tuple[int, float | None, float | None, float | None, float | None]


def _pooled(trials: Sequence[Trial]) -> Pooled:
    """(trials with an estimate, bias, RMSE, coverage of the non-zero effects, false alarms at 0),
    every effect, shape, tier and horizon pooled."""
    scored = [t for t in trials if t.estimate is not None]
    errors = np.array([t.estimate - t.truth for t in scored if t.estimate is not None])
    hits = [t for t in scored if t.effect_pts != 0]
    nulls = [t for t in scored if t.effect_pts == 0]
    return (
        len(scored),
        float(errors.mean()) if len(errors) else None,
        float(np.sqrt((errors**2).mean())) if len(errors) else None,
        float(np.mean([_covers(t, t.truth) for t in hits])) if hits else None,
        float(np.mean([not _covers(t, 0.0) for t in nulls])) if nulls else None,
    )


def _spike_section(trials: Sequence[Trial]) -> list[str]:
    share = float(np.mean([t.volume_spike for t in trials]))
    lines = [
        "## Windows with a volume spike",
        "",
        f"{100 * share:.0f}% of the trials ran on a window with a volume spike: the site's "
        "`volume_spike` flag, a day in the horizon with reviews 3 SD above the 28 days before "
        "(log scale). Quiet windows avoid patches and Steam's seasonal sales, so these are events "
        "neither list holds: a game's own discount before 30 Sep 2026, an anniversary, an update "
        "the classifier didn't mark as a patch. The rule above pools them, as the nightly "
        "estimates meet them; the table shows how much less a flagged estimate can be trusted. "
        "Bias and RMSE in points (positive share) or percent (volume).",
        "",
        "| Outcome | Method | Volume spike | Trials | Bias | RMSE | Coverage | False alarms at 0 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for outcome in sorted({t.outcome for t in trials}):
        for method in sorted({t.method for t in trials}):
            for spike in (False, True):
                group = [
                    t
                    for t in trials
                    if (t.outcome, t.method, t.volume_spike) == (outcome, method, spike)
                ]
                if not group:
                    continue
                n, bias, rmse, coverage, fpr = _pooled(group)
                lines.append(
                    f"| {outcome.replace('_', ' ').capitalize()} | {method} | "
                    f"{'yes' if spike else 'no'} | {n} | {_fmt(bias)} | "
                    f"{'-' if rmse is None else f'{rmse:.2f}'} | {_fmt(coverage, True)} | "
                    f"{_fmt(fpr, True)} |"
                )
    lines.append("")
    return lines


def render_benchmark_report(
    summary: Mapping[tuple[str, str, int, str, float, str], Metrics],
    choice: Choice,
    trials: Sequence[Trial],
    placebos: Sequence[Trial],
    meta: Mapping[str, object],
) -> str:
    lines = [
        "# Phase 4 causal benchmark",
        "",
        "Known effects injected into real quiet stretches of review history (design D12): each",
        "method's error and 90% interval behaviour, measured where the truth is known.",
        "",
        "| Run | |",
        "|---|---|",
        *(f"| {k} | {v} |" for k, v in meta.items()),
        "",
        "## The rule (ADR-007, fixed before the first run)",
        "",
        "Among methods whose 90% intervals contain the injected effect in 85% to 95% of the",
        "positive-share trials (non-zero effects) and whose false-alarm rate at an injected 0",
        "is at most 10%, the default is the one with the lower RMSE. With none, interrupted time",
        "series, and the method note says its intervals weren't validated.",
        "",
        "| Method | Coverage | False alarms at 0 | RMSE (points) |",
        "|---|---|---|---|",
        *(
            f"| {m} | {_fmt(r['coverage'], True)} | {_fmt(r['fpr'], True)} | {r['rmse']:.2f} |"
            for m, r in choice.table.items()
        ),
        "",
        f"Default method: {choice.method} ({choice.reason}).",
        "",
    ]
    fallbacks = [t for t in trials if t.method == "synth"]
    if fallbacks:
        rate = np.mean([t.fallback for t in fallbacks])
        lines += [
            f"Synthetic control fell back to interrupted time series in {100 * rate:.0f}% "
            "of its trials (another patch in its fit window, fewer than 5 qualifying "
            "comparison games, or a twin that didn't track).",
            "",
        ]
    its_trials = [t for t in trials if t.method == "its"]
    if its_trials:
        missing = np.mean([t.estimate is None for t in its_trials])
        lines += [
            "Interrupted time series' intervals are calibrated on quiet dates in the game's own "
            f"past (ADR-007 amendment 1); {100 * missing:.0f}% of its trials had no estimate "
            "(too little quiet history to calibrate, or too few reviews).",
            "",
        ]
    if trials:
        lines += _spike_section(trials)
    for outcome in sorted({key[1] for key in summary}):
        unit = "points" if outcome == POSITIVE_SHARE else "percent"
        lines += [
            f"## {outcome.replace('_', ' ').capitalize()} ({unit})",
            "",
            "| Method | Horizon | Tier | Effect | Shape | n | Bias | RMSE | Coverage | False "
            "alarms | Power |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for (method, out, horizon, tier, effect, shape), m in summary.items():
            if out != outcome:
                continue
            lines.append(
                f"| {method} | {horizon} | {tier} | {effect:+g} | {shape} | {m.n} | "
                f"{_fmt(m.bias)} | {_fmt(m.rmse)} | {_fmt(m.coverage, True)} | "
                f"{_fmt(m.fpr, True)} | {_fmt(m.power, True)} |"
            )
        lines.append("")
    if placebos:
        lines += [
            "## Placebo-in-space (comparison games at real patch dates, no injection)",
            "",
            "| Method | Trials | False alarms |",
            "|---|---|---|",
        ]
        for method in sorted({t.method for t in placebos}):
            mine = [t for t in placebos if t.method == method and t.estimate is not None]
            alarms = np.mean([not _covers(t, 0.0) for t in mine]) if mine else math.nan
            lines.append(f"| {method} | {len(mine)} | {_fmt(float(alarms), True)} |")
        lines.append("")
    lines += [
        "## Limitations",
        "",
        "- The effect-0 trials on tracked games are the placebo-in-time runs: fake patch dates "
        "where none happened.",
        "- Aspect shares among negative reviews use the same interrupted time series but aren't "
        "benchmarked separately; their counts are thinner, so their intervals are wider.",
        "- Patch dates come from a rules classifier (rules-v1): a missed patch can leave a "
        '"quiet" window or a comparison game contaminated.',
        "- Game-specific discounts are known only since 30 Sep 2026; quiet windows avoid Steam's "
        "seasonal sales but not earlier game discounts (the volume-spike table shows their "
        "weight).",
        "- Injected effects are level shifts or 7-day ramps; effects that fade are not tested.",
        "",
    ]
    return "\n".join(lines)
