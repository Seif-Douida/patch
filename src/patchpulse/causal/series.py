"""A game's daily review counts, the unit every estimator works on (design D2).

A tracked game's day with no reviews is an observed 0. A comparison game's day that wasn't fetched
is unobserved: its counts are 0 and it never counts as data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class DailySeries:
    appid: int
    first_day: date
    positives: IntArray
    totals: IntArray
    observed: BoolArray

    def __post_init__(self) -> None:
        if not len(self.positives) == len(self.totals) == len(self.observed):
            raise ValueError("positives, totals and observed must have the same length")
        if (self.positives < 0).any() or (self.totals < 0).any():
            raise ValueError("review counts must not be negative")
        if (self.positives > self.totals).any():
            raise ValueError("positive reviews must not exceed the day's reviews")

    @property
    def last_day(self) -> date:
        return self.first_day + timedelta(days=len(self.totals) - 1)

    def index_of(self, day: date) -> int:
        index = (day - self.first_day).days
        if not 0 <= index < len(self.totals):
            raise ValueError(f"{day} is outside {self.first_day}..{self.last_day}")
        return index

    def span(self, start: date, end: date) -> DailySeries | None:
        """The days from `start` to `end`, both included; None unless the series holds them all."""
        if end < start or start < self.first_day or end > self.last_day:
            return None
        i, j = self.index_of(start), self.index_of(end) + 1
        return DailySeries(
            self.appid, start, self.positives[i:j], self.totals[i:j], self.observed[i:j]
        )

    def share(self) -> FloatArray:
        """The positive share per day; NaN on a day with no reviews or no data."""
        share = np.full(len(self.totals), np.nan)
        known = self.observed & (self.totals > 0)
        share[known] = self.positives[known] / self.totals[known]
        return share

    def median_total(self, start: date, end: date) -> float:
        """The median daily reviews over the observed days of a span (0 when none is observed)."""
        part = self.require_span(start, end)
        values = part.totals[part.observed]
        return float(np.median(values)) if len(values) else 0.0

    def require_span(self, start: date, end: date) -> DailySeries:
        part = self.span(start, end)
        if part is None:
            raise ValueError(f"{start}..{end} is outside {self.first_day}..{self.last_day}")
        return part


def from_counts(
    appid: int,
    first: date,
    last: date,
    counts: Mapping[date, tuple[int, int]],
    *,
    missing_is_zero: bool,
) -> DailySeries:
    """The days from `first` to `last` from each day's (positive, total) counts. A day without
    counts is an observed 0 when `missing_is_zero` (a tracked game's day without reviews), else
    unobserved (a comparison game's day not fetched). Counts outside the span are left out."""
    days = (last - first).days + 1
    positives = np.zeros(days, dtype=np.int64)
    totals = np.zeros(days, dtype=np.int64)
    observed = np.full(days, missing_is_zero)
    for day, (positive, total) in counts.items():
        if first <= day <= last:
            i = (day - first).days
            positives[i], totals[i], observed[i] = positive, total, True
    return DailySeries(appid, first, positives, totals, observed)


@dataclass(frozen=True, eq=False)
class Binned:
    """Consecutive bins of days: each bin's first day in the span, its length and its counts."""

    starts: tuple[date, ...]
    sizes: IntArray
    positives: IntArray
    totals: IntArray


def bin_span(series: DailySeries, anchor: date, start: date, end: date, width: int = 3) -> Binned:
    """Bins of `width` days counted from `anchor` (the patch day starts a bin); the first and last
    bins may be partial. Unobserved days add no reviews."""
    part = series.require_span(start, end)
    starts: list[date] = []
    sizes: list[int] = []
    positives: list[int] = []
    totals: list[int] = []
    current: int | None = None
    for offset in range(len(part.totals)):
        day = start + timedelta(days=offset)
        key = (day - anchor).days // width
        if key != current:
            current = key
            starts.append(day)
            sizes.append(0)
            positives.append(0)
            totals.append(0)
        sizes[-1] += 1
        if part.observed[offset]:
            positives[-1] += int(part.positives[offset])
            totals[-1] += int(part.totals[offset])
    return Binned(
        tuple(starts),
        np.array(sizes, dtype=np.int64),
        np.array(positives, dtype=np.int64),
        np.array(totals, dtype=np.int64),
    )


def needs_bins(series: DailySeries, start: date, end: date, min_count: int = 20) -> bool:
    """Spec §9.1: game-days with fewer than 20 reviews go into 3-day bins."""
    part = series.require_span(start, end)
    return bool((part.totals[part.observed] < min_count).any())


def too_thin(series: DailySeries, start: date, end: date, min_median: int = 10) -> bool:
    """Spec §9.1: a median under 10 reviews a day gets descriptive analysis only."""
    return series.median_total(start, end) < min_median
