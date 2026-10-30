"""Who reviewed against how they felt: a shift-share split of a change in positive share
(design D8)."""

from __future__ import annotations

import numpy as np
import pytest

from patchpulse.causal.decompose import BUCKETS, decompose


def mix(**counts: tuple[int, int]) -> dict[str, tuple[int, int]]:
    """Bucket -> (positive reviews, reviews), with the bucket names' leading digits spelled out."""
    names = {
        "under_2h": "lt_2h",
        "h2_10": "2h_10h",
        "h10_50": "10h_50h",
        "h50_200": "50h_200h",
        "h200_plus": "200h_plus",
    }
    return {names[k]: v for k, v in counts.items()}


def test_buckets_are_fact_reviews() -> None:
    assert BUCKETS == ("lt_2h", "2h_10h", "10h_50h", "50h_200h", "200h_plus", "unknown")


def test_parts_sum_to_the_total() -> None:
    rng = np.random.default_rng(4)
    for _ in range(50):
        sides = []
        for _ in range(2):
            totals = rng.integers(0, 300, len(BUCKETS))
            totals[0] = max(totals[0], 1)
            positives = rng.binomial(totals, rng.uniform(0.3, 0.95, len(BUCKETS)))
            rows = zip(BUCKETS, positives, totals, strict=True)
            sides.append({b: (int(p), int(n)) for b, p, n in rows})
        baseline, window = sides

        parts = decompose(baseline, window, resamples=50)

        assert parts.composition + parts.within + parts.interaction == pytest.approx(parts.total)
        share = [sum(p for p, _ in s.values()) / sum(n for _, n in s.values()) for s in sides]
        assert parts.total == pytest.approx(100 * (share[1] - share[0]))


def test_pure_composition_change() -> None:
    # Same feelings per group; more newcomers, who are harsher.
    baseline = mix(under_2h=(50, 100), h200_plus=(270, 300))
    window = mix(under_2h=(150, 300), h200_plus=(90, 100))

    parts = decompose(baseline, window, resamples=50)

    assert parts.within == pytest.approx(0.0)
    assert parts.interaction == pytest.approx(0.0)
    assert parts.composition == pytest.approx(parts.total)
    assert parts.total < 0


def test_pure_within_change() -> None:
    baseline = mix(under_2h=(80, 100), h10_50=(160, 200))
    window = mix(under_2h=(60, 100), h10_50=(120, 200))

    parts = decompose(baseline, window, resamples=50)

    assert parts.composition == pytest.approx(0.0)
    assert parts.interaction == pytest.approx(0.0)
    assert parts.within == pytest.approx(parts.total)
    assert parts.total == pytest.approx(-20.0)


def test_intervals_cover_the_point_estimate() -> None:
    baseline = mix(under_2h=(400, 600), h2_10=(800, 1000), h10_50=(900, 1000))
    window = mix(under_2h=(500, 900), h2_10=(700, 1000), h10_50=(850, 1000))

    parts = decompose(baseline, window)
    again = decompose(baseline, window)

    for name in ("total", "composition", "within", "interaction"):
        lo, hi = parts.intervals[name]
        assert lo <= getattr(parts, name) <= hi, name
    assert parts.intervals == again.intervals  # seeded
    assert (parts.baseline_n, parts.window_n) == (2600, 2900)


def test_empty_bucket_on_one_side() -> None:
    baseline = mix(under_2h=(80, 100))
    window = mix(under_2h=(80, 100), h200_plus=(95, 100))

    parts = decompose(baseline, window, resamples=50)

    # The new group's arrival is composition; nobody in it changed their mind.
    assert parts.within == pytest.approx(0.0)
    assert parts.composition == pytest.approx(parts.total)
    assert np.isfinite([parts.total, parts.composition, parts.within, parts.interaction]).all()


def test_no_reviews_on_a_side_is_refused() -> None:
    with pytest.raises(ValueError, match="no reviews"):
        decompose({}, mix(under_2h=(1, 2)))
