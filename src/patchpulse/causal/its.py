"""Interrupted time series (design D5, spec §9.2 M1): the game's own history, projected past the
patch.

Weighted least squares on the daily outcome over up to 56 days before the patch and the horizon's
days after it: an intercept, a linear trend, day-of-week effects, a step for each earlier patch
inside the pre-period, and the step at the patch, whose coefficient is the effect. The interval
comes from Newey-West (HAC) errors, since neighbouring days are alike: the lag by Newey and West's
own rule (3 for ~70 rows) and fixed-b critical values (Kiefer & Vogelsang 2005), which correct the
too-narrow intervals HAC gives on short series. Measured 21 Oct on 500 simulated series of 70 days
with independent noise: 0.90 coverage, against 0.80 with a 7-day lag and Student-t quantiles.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    """The patch's effect over `horizon` days from `start` (the patch day included). `lag` defaults
    to Newey and West's rule for the number of rows; only the spec's 90% level is supported."""
    if level != LEVEL:
        raise ValueError("the fixed-b critical values are for the 90% level only")
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
    step, se = float(beta[-1]), float(np.sqrt(covariance[-1, -1]))
    half = fixed_b_critical((lags + 1) / n) * se
    if kind == "share":
        effect, lo, hi = 100 * step, 100 * (step - half), 100 * (step + half)
    else:
        effect, lo, hi = (100 * float(np.expm1(v)) for v in (step, step - half, step + half))
    n_pre = sum(day < start for day in rows.days)
    return ItsResult(effect=effect, lo90=lo, hi90=hi, se=100 * se, n_pre=n_pre, binned=binned)
