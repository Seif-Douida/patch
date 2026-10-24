"""Daily series, the unit every estimator works on (Phase 4 design D2)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
import pytest

from patchpulse.causal.series import DailySeries, bin_span, needs_bins, too_thin

D0 = date(2026, 1, 1)


def day(offset: int) -> date:
    return D0 + timedelta(days=offset)


def series(
    positives: Sequence[int], totals: Sequence[int], observed: Sequence[bool] | None = None
) -> DailySeries:
    return DailySeries(
        appid=1,
        first_day=D0,
        positives=np.array(positives, dtype=np.int64),
        totals=np.array(totals, dtype=np.int64),
        observed=np.ones(len(totals), dtype=bool) if observed is None else np.array(observed),
    )


def test_series_validates_its_arrays() -> None:
    with pytest.raises(ValueError, match="same length"):
        series([1, 2], [3])
    with pytest.raises(ValueError, match="same length"):
        series([1, 2], [3, 3], observed=[True])
    with pytest.raises(ValueError, match="negative"):
        series([-1], [3])
    with pytest.raises(ValueError, match="exceed"):
        series([4], [3])


def test_share_is_nan_on_empty_or_unobserved_days() -> None:
    share = series([3, 0, 2], [4, 0, 2], observed=[True, True, False]).share()

    assert share[0] == 0.75
    assert np.isnan(share[1])  # no reviews that day
    assert np.isnan(share[2])  # a comparison game's day that wasn't fetched


def test_span_inside_and_outside() -> None:
    whole = series(list(range(10)), [10] * 10)

    part = whole.span(day(2), day(4))

    assert part is not None
    assert (part.first_day, list(part.positives), part.last_day) == (day(2), [2, 3, 4], day(4))
    assert whole.last_day == day(9)
    assert whole.index_of(day(3)) == 3
    assert whole.span(day(-1), day(1)) is None
    assert whole.span(day(8), day(10)) is None
    with pytest.raises(ValueError, match="outside"):
        whole.index_of(day(10))


def test_bins_start_on_the_patch_day() -> None:
    whole = series([1] * 11, [2] * 11)

    binned = bin_span(whole, anchor=day(4), start=day(0), end=day(10))

    # Bins of 3 counted from the patch day: a partial first bin (day 0), then 1-3, 4-6, 7-9, and a
    # partial last bin (day 10).
    assert binned.starts == (day(0), day(1), day(4), day(7), day(10))
    assert list(binned.sizes) == [1, 3, 3, 3, 1]
    assert list(binned.positives) == [1, 3, 3, 3, 1]
    assert list(binned.totals) == [2, 6, 6, 6, 2]


def test_needs_bins_and_too_thin_ignore_unobserved_days() -> None:
    lumpy = series([0] * 4, [25, 30, 19, 0], observed=[True, True, True, False])
    thin = series([0] * 3, [9, 9, 12])
    enough = series([0] * 4, [10, 10, 9, 0], observed=[True, True, True, False])

    assert needs_bins(lumpy, day(0), day(3))  # 19 is under 20
    assert not needs_bins(lumpy, day(0), day(1))
    assert too_thin(thin, day(0), day(2))  # median 9
    assert not too_thin(enough, day(0), day(3))  # median 10 over the observed days
