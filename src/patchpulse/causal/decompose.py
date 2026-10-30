"""Who reviewed against how they felt (design D8, spec §9.3): a shift-share split of a change in the
positive share between a baseline window and a window, across playtime-at-review buckets.

total = composition + within + interaction, exactly:
- composition, sum of (w1 - w0) s0: different groups reviewing, each feeling as before;
- within, sum of w0 (s1 - s0): the same mix of groups, feeling differently;
- interaction, sum of (w1 - w0)(s1 - s0).
w is a bucket's share of the reviews and s its positive share, in points. A bucket empty on one
side takes the other side's share, so its arrival or departure counts as composition only.
Intervals resample the counts (multinomial buckets, binomial positives), seeded.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from patchpulse.causal.series import FloatArray

# fact_review.playtime_bucket's values, plus reviews whose playtime Steam didn't send.
BUCKETS = ("lt_2h", "2h_10h", "10h_50h", "50h_200h", "200h_plus", "unknown")
BucketCounts = Mapping[str, tuple[int, int]]  # bucket -> (positive reviews, reviews)
PARTS = ("total", "composition", "within", "interaction")


@dataclass(frozen=True)
class Decomposition:
    total: float
    composition: float
    within: float
    interaction: float
    intervals: dict[str, tuple[float, float]]
    baseline_n: int
    window_n: int


def _parts(p0: FloatArray, n0: FloatArray, p1: FloatArray, n1: FloatArray) -> dict[str, FloatArray]:
    """The four parts, in points, over the last axis (buckets); leading axes are resamples."""
    w0 = n0 / n0.sum(axis=-1, keepdims=True)
    w1 = n1 / n1.sum(axis=-1, keepdims=True)
    raw0 = np.divide(p0, n0, out=np.zeros_like(p0), where=n0 > 0)
    raw1 = np.divide(p1, n1, out=np.zeros_like(p1), where=n1 > 0)
    s0 = np.where(n0 > 0, raw0, np.where(n1 > 0, raw1, 0.0))
    s1 = np.where(n1 > 0, raw1, s0)
    composition = 100 * ((w1 - w0) * s0).sum(axis=-1)
    within = 100 * (w0 * (s1 - s0)).sum(axis=-1)
    interaction = 100 * ((w1 - w0) * (s1 - s0)).sum(axis=-1)
    return {
        "total": composition + within + interaction,
        "composition": composition,
        "within": within,
        "interaction": interaction,
    }


def _resample(
    positives: FloatArray, totals: FloatArray, resamples: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    count = int(totals.sum())
    counts = rng.multinomial(count, totals / count, size=resamples).astype(np.float64)
    shares = np.divide(positives, totals, out=np.zeros_like(positives), where=totals > 0)
    drawn = rng.binomial(counts.astype(np.int64), shares).astype(np.float64)
    return drawn, counts


def decompose(
    baseline: BucketCounts,
    window: BucketCounts,
    *,
    resamples: int = 2000,
    seed: int = 20261021,
    level: float = 0.90,
) -> Decomposition:
    buckets = list(BUCKETS) + sorted((set(baseline) | set(window)) - set(BUCKETS))
    p0 = np.array([baseline.get(b, (0, 0))[0] for b in buckets], dtype=np.float64)
    n0 = np.array([baseline.get(b, (0, 0))[1] for b in buckets], dtype=np.float64)
    p1 = np.array([window.get(b, (0, 0))[0] for b in buckets], dtype=np.float64)
    n1 = np.array([window.get(b, (0, 0))[1] for b in buckets], dtype=np.float64)
    if n0.sum() == 0:
        raise ValueError("no reviews in the baseline window")
    if n1.sum() == 0:
        raise ValueError("no reviews in the window")
    point = _parts(p0, n0, p1, n1)
    rng = np.random.default_rng(seed)
    draws = _parts(*_resample(p0, n0, resamples, rng), *_resample(p1, n1, resamples, rng))
    tail = (1 - level) / 2 * 100
    intervals = {
        name: (
            float(np.percentile(draws[name], tail)),
            float(np.percentile(draws[name], 100 - tail)),
        )
        for name in PARTS
    }
    return Decomposition(
        total=float(point["total"]),
        composition=float(point["composition"]),
        within=float(point["within"]),
        interaction=float(point["interaction"]),
        intervals=intervals,
        baseline_n=int(n0.sum()),
        window_n=int(n1.sum()),
    )
