"""Interrupted time series (design D5, spec §9.2 M1): the game's own history, projected past the
patch.

Weighted least squares on the daily outcome over up to 56 days before the patch and the horizon's
days after it: an intercept, a linear trend, day-of-week effects, a step for each earlier patch
inside the pre-period, and the step at the patch, whose coefficient is the effect. The interval
comes from Newey-West (HAC) errors, since neighbouring days are alike: the lag by Newey and West's
own rule (3 for ~70 rows) and fixed-b critical values (Kiefer & Vogelsang 2005), which correct the
too-narrow intervals HAC gives on short series. Measured 21 Oct on 500 simulated series of 70 days
with independent noise: 0.90 coverage, against 0.80 with a 7-day lag and Student-t quantiles.

On real review series that interval is still too narrow (Task 11's dry run: 79% coverage), so the
served estimate is `calibrated_its_effect` (ADR-007 amendment 1): the same fit at quiet dates in
the game's own past, and the interval's critical value the conformal quantile of their |t|.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import numpy as np

from patchpulse.causal.series import DailySeries, FloatArray, bin_span, needs_bins, too_thin

POSITIVE_SHARE = "positive_share"
VOLUME = "volume"
Kind = Literal["share", "volume"]

PRE_DAYS = 56
MIN_PRE_DAYS = 28
LEVEL = 0.90
# Placebo dates for the calibration (ADR-007 amendment 1): weekly, so each shares the patch's
# weekday; a year back at most; no patch, sale or discount from a week before one to its horizon's
# end; at least 9, the fewest for which the 90% conformal rank exists.
PLACEBO_STEP_DAYS = 7
PLACEBO_LOOKBACK_DAYS = 364
PLACEBO_CALM_DAYS = 7
MIN_PLACEBOS = 9
CALIBRATION_DAYS = PLACEBO_LOOKBACK_DAYS + PRE_DAYS  # how far before the patch an estimate reads


def aspect_outcome(code: str) -> str:
    """An aspect's share among scored negative reviews, as an outcome name."""
    return f"aspect:{code}"


@dataclass(frozen=True)
class NoEstimate:
    reason: str


@dataclass(frozen=True)
class ItsResult:
    """The effect and its interval: percentage points for a share, a percent change for volume."""

    effect: float
    lo90: float
    hi90: float
    se: float  # of the step, in points (share) or log points x 100 (volume)
    n_pre: int  # rows before the patch
    binned: bool
    critical: float  # the interval's half-width over se
    placebos: int  # placebo dates the critical value came from; 0 for the fixed-b value


def newey_west_lag(n: int) -> int:
    """Newey and West's (1994) rule of thumb for the Bartlett kernel: floor(4 (n / 100)^(2/9))."""
    return int(4 * (n / 100) ** (2 / 9))


def fixed_b_critical(b: float) -> float:
    """The two-sided 90% critical value of a Bartlett HAC t statistic with bandwidth ratio b
    (Kiefer & Vogelsang 2005's cubic fit). Checked 21 Oct against a 20,000-draw simulation: within
    0.05 above it at b = 0.05-0.2, so slightly wider (safer) than the simulated value."""
    return 1.6449 + 2.1859 * b + 0.3142 * b**2 - 0.3427 * b**3


def hac_cov(design: FloatArray, residuals: FloatArray, lag: int) -> FloatArray:
    """Newey-West covariance of least-squares coefficients: Bartlett weights, `lag` lags and the
    n / (n - k) correction (statsmodels' `cov_type="HAC"` with `use_correction=True`). For weighted
    least squares pass the design and residuals already scaled by the square root of the weights."""
    n, k = design.shape
    scores = design * residuals[:, None]
    meat = scores.T @ scores
    for distance in range(1, lag + 1):
        weight = 1.0 - distance / (lag + 1)
        cross = scores[distance:].T @ scores[:-distance]
        meat += weight * (cross + cross.T)
    bread = np.linalg.inv(design.T @ design)
    covariance: FloatArray = bread @ meat @ bread * (n / (n - k))
    return covariance


@dataclass(frozen=True)
class _Rows:
    days: list[date]  # each row's first day (a bin's start when binned)
    offsets: FloatArray  # the row's middle, in days from the patch: the trend
    y: FloatArray
    weights: FloatArray


def _day_rows(series: DailySeries, first: date, end: date, start: date, kind: Kind) -> _Rows:
    part = series.require_span(first, end)
    days: list[date] = []
    y: list[float] = []
    weights: list[float] = []
    for index in range(len(part.totals)):
        total = int(part.totals[index])
        if not part.observed[index] or (kind == "share" and total == 0):
            continue
        days.append(first + timedelta(days=index))
        if kind == "share":
            y.append(int(part.positives[index]) / total)
            weights.append(float(total))
        else:
            y.append(float(np.log1p(total)))
            weights.append(1.0)
    offsets = np.array([(day - start).days for day in days], dtype=np.float64)
    return _Rows(days, offsets, np.array(y), np.array(weights))


def _bin_rows(series: DailySeries, first: date, end: date, start: date) -> _Rows:
    binned = bin_span(series, anchor=start, start=first, end=end)
    keep = binned.totals > 0
    days = [day for day, kept in zip(binned.starts, keep, strict=True) if kept]
    sizes = binned.sizes[keep]
    offsets = np.array([(day - start).days for day in days], dtype=np.float64) + (sizes - 1) / 2
    y = binned.positives[keep] / binned.totals[keep]
    return _Rows(days, offsets, y.astype(np.float64), binned.totals[keep].astype(np.float64))


def _design(rows: _Rows, steps: Sequence[date], start: date, weekdays: bool) -> FloatArray:
    columns = [np.ones(len(rows.days)), rows.offsets]
    if weekdays:
        weekday = np.array([day.weekday() for day in rows.days])
        columns += [(weekday == d).astype(np.float64) for d in range(6) if (weekday == d).any()]
    columns += [np.array([day >= step for day in rows.days], dtype=np.float64) for step in steps]
    columns.append(np.array([day >= start for day in rows.days], dtype=np.float64))
    return np.column_stack(columns)


def _pre_days_with_data(series: DailySeries, first: date, start: date, kind: Kind) -> int:
    part = series.require_span(first, start - timedelta(days=1))
    data = part.observed & (part.totals > 0) if kind == "share" else part.observed
    return int(data.sum())


@dataclass(frozen=True)
class _Fit:
    step: float  # a share's fraction or log(1 + reviews)
    se: float
    lags: int
    n: int
    n_pre: int
    binned: bool


def _fit(
    series: DailySeries,
    start: date,
    horizon: int,
    earlier_starts: Sequence[date],
    kind: Kind,
    *,
    pre_days: int = PRE_DAYS,
    min_pre_days: int = MIN_PRE_DAYS,
    lag: int | None = None,
) -> _Fit | NoEstimate:
    end = start + timedelta(days=horizon - 1)
    if end > series.last_day:
        return NoEstimate("window not complete")
    first = max(series.first_day, start - timedelta(days=pre_days))
    if first >= start:
        return NoEstimate("pre-period under 28 days with reviews")
    if too_thin(series, first, end):
        return NoEstimate("too few reviews (median under 10 a day)")
    if _pre_days_with_data(series, first, start, kind) < min_pre_days:
        return NoEstimate("pre-period under 28 days with reviews")

    binned = kind == "share" and needs_bins(series, first, end)
    if binned:
        rows = _bin_rows(series, first, end, start)
    else:
        rows = _day_rows(series, first, end, start, kind)
    steps = sorted(step for step in set(earlier_starts) if first < step < start)
    x = _design(rows, steps, start, weekdays=not binned)
    n, k = x.shape
    root = np.sqrt(rows.weights)
    xw, yw = x * root[:, None], rows.y * root
    if n <= k or np.linalg.matrix_rank(xw) < k:
        return NoEstimate("singular design")
    beta = np.linalg.lstsq(xw, yw, rcond=None)[0]
    lags = newey_west_lag(n) if lag is None else lag
    covariance = hac_cov(xw, yw - xw @ beta, lags)
    n_pre = sum(day < start for day in rows.days)
    return _Fit(float(beta[-1]), float(np.sqrt(covariance[-1, -1])), lags, n, n_pre, binned)


def _result(fit: _Fit, kind: Kind, critical: float, placebos: int) -> ItsResult:
    bounds = (fit.step, fit.step - critical * fit.se, fit.step + critical * fit.se)
    if kind == "share":
        effect, lo, hi = (100 * v for v in bounds)
    else:
        effect, lo, hi = (100 * float(np.expm1(v)) for v in bounds)
    return ItsResult(
        effect=effect,
        lo90=lo,
        hi90=hi,
        se=100 * fit.se,
        n_pre=fit.n_pre,
        binned=fit.binned,
        critical=critical,
        placebos=placebos,
    )


def its_effect(
    series: DailySeries,
    start: date,
    horizon: int,
    earlier_starts: Sequence[date],
    kind: Kind,
    *,
    pre_days: int = PRE_DAYS,
    min_pre_days: int = MIN_PRE_DAYS,
    lag: int | None = None,
    level: float = LEVEL,
) -> ItsResult | NoEstimate:
    """The patch's effect over `horizon` days from `start` (the patch day included), with the
    uncalibrated fixed-b interval: the building block of `calibrated_its_effect`. `lag` defaults to
    Newey and West's rule for the number of rows; only the spec's 90% level is supported."""
    if level != LEVEL:
        raise ValueError("the fixed-b critical values are for the 90% level only")
    fit = _fit(
        series,
        start,
        horizon,
        earlier_starts,
        kind,
        pre_days=pre_days,
        min_pre_days=min_pre_days,
        lag=lag,
    )
    if isinstance(fit, NoEstimate):
        return fit
    return _result(fit, kind, fixed_b_critical((fit.lags + 1) / fit.n), 0)


def placebo_starts(
    start: date, horizon: int, busy_days: Collection[date], *, first_day: date
) -> list[date]:
    """The quiet dates in the game's past to rerun the fit at: every 7 days back from `start`, at
    most a year and not before `first_day`, whose horizon ends before `start`, with no busy day
    (a patch, sale or discount) from a week before the date to the horizon's end."""
    oldest = max(first_day, start - timedelta(days=PLACEBO_LOOKBACK_DAYS))
    dates: list[date] = []
    candidate = start - timedelta(days=PLACEBO_STEP_DAYS)
    while candidate >= oldest:
        end = candidate + timedelta(days=horizon - 1)
        calm_from = candidate - timedelta(days=PLACEBO_CALM_DAYS)
        if end < start and not any(calm_from <= day <= end for day in busy_days):
            dates.append(candidate)
        candidate -= timedelta(days=PLACEBO_STEP_DAYS)
    return sorted(dates)


def conformal_critical(t_stats: Sequence[float], level: float = LEVEL) -> float | None:
    """The ceil(level (m + 1))-th smallest |t| of m placebo t statistics: with the real date's t
    exchangeable with theirs, the interval covers at least `level`. None when the rank exceeds m.
    """
    rank = math.ceil(level * (len(t_stats) + 1))
    if rank > len(t_stats):
        return None
    return sorted(abs(t) for t in t_stats)[rank - 1]


def calibrated_its_effect(
    series: DailySeries,
    start: date,
    horizon: int,
    earlier_starts: Sequence[date],
    kind: Kind,
    *,
    busy_days: Collection[date],
) -> ItsResult | NoEstimate:
    """The served estimate (ADR-007 amendment 1): the fit's step with a critical value from the
    same fit at the game's quiet past dates, which read nothing from `start` on."""
    fit = _fit(series, start, horizon, earlier_starts, kind)
    if isinstance(fit, NoEstimate):
        return fit
    t_stats: list[float] = []
    for placebo in placebo_starts(start, horizon, busy_days, first_day=series.first_day):
        other = _fit(series, placebo, horizon, earlier_starts, kind)
        if isinstance(other, _Fit) and other.se > 0:
            t_stats.append(other.step / other.se)
    critical = conformal_critical(t_stats)
    if critical is None:
        return NoEstimate(
            f"too little quiet history to check the interval ({len(t_stats)} of {MIN_PLACEBOS} "
            "placebo dates)"
        )
    return _result(fit, kind, critical, len(t_stats))
