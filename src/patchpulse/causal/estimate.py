"""Both methods put together for a patch (design D2-D7, D10): the rows the nightly stage stores.

For each bundle and each complete horizon (7 days; 14 when no other patch lands in days 8-14):
- interrupted time series for the positive share, volume, each aspect's share among negative
  reviews, and the positive share among established players (10+ hours at review), its interval
  calibrated on quiet dates in the game's past (ADR-007 amendment 1);
- synthetic control for the positive share and volume, or a row saying why not (another patch of
  the game in its fit window, fewer than 5 qualifying comparison games, a twin that didn't track).

Each row carries the flags (a sale or discount nearby, discounts unknown, a volume spike, binned
days) and a fingerprint of exactly what it was computed from, so a row is recomputed only when its
inputs change, or when EFFECTS_VERSION does.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from typing import NamedTuple

import numpy as np

from patchpulse.causal.bundles import Bundle, Patch, bundle_patches, window_complete
from patchpulse.causal.its import (
    CALIBRATION_DAYS,
    POSITIVE_SHARE,
    VOLUME,
    Kind,
    NoEstimate,
    aspect_outcome,
    calibrated_its_effect,
)
from patchpulse.causal.rules import (
    FIT_DAYS,
    MIN_DONORS,
    POST_DAYS,
    eligible_donors,
    genre_group,
    own_fit_window_clear,
)
from patchpulse.causal.series import DailySeries
from patchpulse.causal.synth import conformal_interval, placebo_p_value, synth_effect

EFFECTS_VERSION = "e2"  # bump when the maths or a threshold changes: every row is recomputed
# e2 (22 Oct 2026): interrupted time series' interval calibrated on the game's own past.
HORIZONS = (7, 14)
SALE_WINDOW_DAYS = 3
SPIKE_SDS = 3.0


class Method(StrEnum):
    ITS = "its"
    SYNTH = "synth"


class Sample(StrEnum):
    ALL = "all"
    ESTABLISHED = "established"


@dataclass(frozen=True)
class GameInfo:
    appid: int
    genres: tuple[str, ...]
    role: str


@dataclass(frozen=True, eq=False)
class EffectInputs:
    series: Mapping[int, DailySeries]  # tracked and `both` games: all reviews
    established: Mapping[int, DailySeries]  # their reviews with 10+ hours at review
    aspects: Mapping[int, Mapping[str, DailySeries]]  # negatives mentioning it / scored negatives
    donors: Mapping[int, DailySeries]  # comparison games: daily counts
    games: Mapping[int, GameInfo]
    patches: Mapping[int, Sequence[Patch]]  # treatment candidates, every role
    sale_days: frozenset[date]
    discounts: Mapping[int, frozenset[date]]
    discounts_known_from: date
    last_complete_day: date


class EstimateKey(NamedTuple):
    bundle_id: str
    outcome: str
    sample: str
    horizon: int
    method: str


@dataclass(frozen=True)
class Estimate:
    bundle_id: str
    patch_ids: tuple[str, ...]
    appid: int
    outcome: str
    sample: str
    horizon: int
    method: str
    effect: float | None
    lo90: float | None
    hi90: float | None
    p_value: float | None
    pre_fit_rmse: float | None
    donors: tuple[tuple[int, float], ...]
    flags: frozenset[str]
    reason: str | None
    fingerprint: str
    version: str

    @property
    def key(self) -> EstimateKey:
        return EstimateKey(self.bundle_id, self.outcome, self.sample, self.horizon, self.method)


def fingerprint(*parts: object) -> str:
    """SHA-256 over the parts and EFFECTS_VERSION: arrays by dtype, shape and bytes, tuples and
    lists item by item, sets in sorted order, anything else by repr. Each night is a new process
    and a set's iteration order follows the process's string hashing, so sets are sorted."""
    digest = hashlib.sha256(EFFECTS_VERSION.encode())
    for part in parts:
        for chunk in _chunks(part):
            digest.update(chunk)
    return digest.hexdigest()


def _chunks(part: object) -> Iterator[bytes]:
    if isinstance(part, np.ndarray):
        yield f"array{part.dtype}{part.shape}".encode()
        yield np.ascontiguousarray(part).tobytes()
    elif isinstance(part, set | frozenset):
        yield f"set{len(part)}".encode()
        for item in sorted(part, key=repr):
            yield from _chunks(item)
    elif isinstance(part, tuple | list):
        yield f"seq{len(part)}".encode()
        for item in part:
            yield from _chunks(item)
    else:
        yield repr(part).encode()
    yield b"|"


def _clipped(series: DailySeries, first: date, last: date) -> DailySeries | None:
    start, end = max(first, series.first_day), min(last, series.last_day)
    return series.span(start, end)


def _arrays(series: DailySeries | None) -> tuple[object, ...]:
    if series is None:
        return (None,)
    return (series.first_day, series.positives, series.totals, series.observed)


def flags_for(bundle: Bundle, horizon: int, inputs: EffectInputs) -> frozenset[str]:
    """The bundle's flags for a horizon: `confounded`, `discounts_unknown`, `volume_spike`."""
    flags: set[str] = set()
    near = {
        bundle.start + timedelta(days=k) for k in range(-SALE_WINDOW_DAYS, SALE_WINDOW_DAYS + 1)
    }
    if near & inputs.sale_days or near & inputs.discounts.get(bundle.appid, frozenset()):
        flags.add("confounded")
    if bundle.start - timedelta(days=SALE_WINDOW_DAYS) < inputs.discounts_known_from:
        flags.add("discounts_unknown")
    series = inputs.series.get(bundle.appid)
    if series is not None and volume_spike(series, bundle.start, horizon):
        flags.add("volume_spike")
    return frozenset(flags)


def volume_spike(series: DailySeries, start: date, horizon: int) -> bool:
    """A day in the horizon with reviews 3 SD above the 28 days before (log scale), given at least
    7 days of data before. The benchmark splits its trials by this same test."""
    pre = _clipped(series, start - timedelta(days=FIT_DAYS), start - timedelta(1))
    post = _clipped(series, start, start + timedelta(days=horizon - 1))
    if pre is None or post is None or pre.observed.sum() < 7:
        return False
    before = np.log1p(pre.totals[pre.observed])
    after = np.log1p(post.totals[post.observed])
    # At least Poisson noise on the log scale: a perfectly steady volume still has a scale.
    floor = 1.0 / np.sqrt(1.0 + float(pre.totals[pre.observed].mean()))
    sd = max(float(before.std()), floor)
    return bool((after > before.mean() + SPIKE_SDS * sd).any())


@dataclass(frozen=True)
class _Plan:
    key: EstimateKey
    fingerprint: str
    run: Callable[[], Estimate]


def tracked_bundles(inputs: EffectInputs) -> list[Bundle]:
    """The bundles of every game the site tracks (comparison games get none)."""
    return bundle_patches([p for appid in inputs.series for p in inputs.patches.get(appid, [])])


def _row(
    bundle: Bundle,
    key: EstimateKey,
    print_: str,
    flags: frozenset[str],
    *,
    effect: float | None = None,
    lo90: float | None = None,
    hi90: float | None = None,
    p_value: float | None = None,
    pre_fit_rmse: float | None = None,
    donors: tuple[tuple[int, float], ...] = (),
    extra_flags: frozenset[str] = frozenset(),
    reason: str | None = None,
) -> Estimate:
    if effect is not None and not math.isfinite(effect):
        effect, reason = None, "non-finite result"
    return Estimate(
        bundle_id=bundle.bundle_id,
        patch_ids=bundle.patch_ids,
        appid=bundle.appid,
        outcome=key.outcome,
        sample=key.sample,
        horizon=key.horizon,
        method=key.method,
        effect=effect,
        lo90=_finite(lo90),
        hi90=_finite(hi90),
        p_value=_finite(p_value),
        pre_fit_rmse=_finite(pre_fit_rmse),
        donors=donors,
        flags=flags | extra_flags,
        reason=reason,
        fingerprint=print_,
        version=EFFECTS_VERSION,
    )


def _finite(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _its_plan(
    bundle: Bundle,
    horizon: int,
    outcome: str,
    sample: Sample,
    series: DailySeries,
    kind: Kind,
    earlier: list[date],
    flags: frozenset[str],
    busy: frozenset[date],
) -> _Plan:
    key = EstimateKey(bundle.bundle_id, outcome, sample, horizon, Method.ITS)
    # The placebo dates' fits read up to CALIBRATION_DAYS before the patch.
    history = bundle.start - timedelta(days=CALIBRATION_DAYS)
    window = _clipped(series, history, bundle.start + timedelta(days=horizon - 1))
    busy_before = sorted(d for d in busy if history <= d < bundle.start)
    print_ = fingerprint(
        key, bundle.patch_ids, sorted(earlier), flags, busy_before, *_arrays(window)
    )

    def run() -> Estimate:
        result = calibrated_its_effect(series, bundle.start, horizon, earlier, kind, busy_days=busy)
        if isinstance(result, NoEstimate):
            return _row(bundle, key, print_, flags, reason=result.reason)
        extra = frozenset({"binned"}) if result.binned else frozenset()
        return _row(
            bundle,
            key,
            print_,
            flags,
            effect=result.effect,
            lo90=result.lo90,
            hi90=result.hi90,
            extra_flags=extra,
        )

    return _Plan(key, print_, run)


def _synth_plan(
    bundle: Bundle,
    horizon: int,
    outcome: str,
    kind: Kind,
    inputs: EffectInputs,
    clear: bool,
    flags: frozenset[str],
) -> _Plan:
    key = EstimateKey(bundle.bundle_id, outcome, Sample.ALL, horizon, Method.SYNTH)
    treated = inputs.series[bundle.appid]
    first = bundle.start - timedelta(days=FIT_DAYS)
    last = bundle.start + timedelta(days=POST_DAYS - 1)
    # A comparison game qualifies on its days to start + 13 whatever the horizon, so until they
    # have ended the row only says when; its fingerprint holds still until then.
    ready = last <= inputs.last_complete_day
    pool: dict[int, DailySeries] = {}
    chosen: list[int] = []
    if not clear:
        print_ = fingerprint(key, bundle.patch_ids, "another patch in the fit window", flags)
    elif not ready:
        print_ = fingerprint(key, bundle.patch_ids, "waiting for", last, flags)
    else:
        pool = {a: s for a, s in {**inputs.donors, **inputs.series}.items() if a != bundle.appid}
        groups = {a: genre_group(inputs.games[a].genres) for a in pool if a in inputs.games}
        patch_days = {a: [p.day for p in inputs.patches.get(a, [])] for a in groups}
        group = genre_group(inputs.games[bundle.appid].genres)
        chosen = eligible_donors(bundle, group, groups, patch_days, pool)
        donor_arrays = [(a, *_arrays(pool[a].span(first, last))) for a in chosen]
        print_ = fingerprint(
            key, bundle.patch_ids, flags, *_arrays(_clipped(treated, first, last)), donor_arrays
        )

    def run() -> Estimate:
        if not clear:
            return _row(
                bundle,
                key,
                print_,
                flags,
                reason="the game had another patch in the 28 days before",
            )
        if not ready:
            return _row(
                bundle,
                key,
                print_,
                flags,
                reason=(
                    "the comparison games are checked over the 14 days after the patch, which "
                    f"end on {last.day} {last:%b %Y}"
                ),
            )
        if len(chosen) < MIN_DONORS:
            return _row(
                bundle,
                key,
                print_,
                flags,
                reason=(
                    f"too few comparison games without their own patch ({len(chosen)} of "
                    f"{MIN_DONORS} needed)"
                ),
            )
        donors = {a: pool[a] for a in chosen}
        fit = synth_effect(treated, donors, bundle.start, horizon, kind)
        if isinstance(fit, NoEstimate):
            return _row(bundle, key, print_, flags, reason=fit.reason)
        weights = tuple((a, round(w, 4)) for a, w in sorted(fit.weights.items()) if w >= 1e-4)
        if not fit.fit_ok:
            return _row(
                bundle,
                key,
                print_,
                flags,
                pre_fit_rmse=fit.pre_rmse,
                donors=weights,
                reason="the comparison games didn't track the game before the patch",
            )
        interval = conformal_interval(treated, donors, bundle.start, horizon, kind)
        p_value = placebo_p_value(treated, donors, bundle.start, horizon, kind, fit)
        no_range = interval if isinstance(interval, NoEstimate) else None
        lo, hi = (None, None) if isinstance(interval, NoEstimate) else interval
        return _row(
            bundle,
            key,
            print_,
            flags,
            effect=fit.effect,
            lo90=lo,
            hi90=hi,
            p_value=p_value,
            pre_fit_rmse=fit.pre_rmse,
            donors=weights,
            reason=no_range.reason if no_range else None,
        )

    return _Plan(key, print_, run)


def _plans(bundle: Bundle, inputs: EffectInputs, game_bundles: Sequence[Bundle]) -> list[_Plan]:
    series = inputs.series[bundle.appid]
    earlier = [b.start for b in game_bundles if b.start < bundle.start]
    clear = own_fit_window_clear(bundle, game_bundles)
    # A placebo date must be clear of the game's patches, Steam sales and the game's discounts.
    busy = (
        frozenset(p.day for p in inputs.patches.get(bundle.appid, []))
        | inputs.sale_days
        | inputs.discounts.get(bundle.appid, frozenset())
    )
    plans: list[_Plan] = []
    for horizon in HORIZONS:
        if horizon == 14 and not bundle.allow_14:
            continue
        if not window_complete(bundle, horizon, inputs.last_complete_day):
            continue
        flags = flags_for(bundle, horizon, inputs)
        outcomes: list[tuple[str, Sample, DailySeries, Kind]] = [
            (POSITIVE_SHARE, Sample.ALL, series, "share"),
            (VOLUME, Sample.ALL, series, "volume"),
        ]
        for code, aspect in sorted(inputs.aspects.get(bundle.appid, {}).items()):
            outcomes.append((aspect_outcome(code), Sample.ALL, aspect, "share"))
        established = inputs.established.get(bundle.appid)
        if established is not None:
            outcomes.append((POSITIVE_SHARE, Sample.ESTABLISHED, established, "share"))
        for outcome, sample, data, kind in outcomes:
            plans.append(
                _its_plan(bundle, horizon, outcome, sample, data, kind, earlier, flags, busy)
            )
        for outcome, kind in ((POSITIVE_SHARE, "share"), (VOLUME, "volume")):
            plans.append(_synth_plan(bundle, horizon, outcome, kind, inputs, clear, flags))
    return plans


def _game_bundles(bundle: Bundle, inputs: EffectInputs) -> list[Bundle]:
    return [b for b in tracked_bundles(inputs) if b.appid == bundle.appid]


def estimate_bundle(bundle: Bundle, inputs: EffectInputs) -> list[Estimate]:
    """Every row for the bundle's complete horizons."""
    return [plan.run() for plan in _plans(bundle, inputs, _game_bundles(bundle, inputs))]


def bundles_to_estimate(inputs: EffectInputs, stored: Mapping[EstimateKey, str]) -> list[Bundle]:
    """The tracked bundles with a complete horizon whose rows are new or whose inputs changed,
    newest first."""
    bundles = tracked_bundles(inputs)
    by_game: dict[int, list[Bundle]] = {}
    for bundle in bundles:
        by_game.setdefault(bundle.appid, []).append(bundle)
    due = [
        bundle
        for bundle in bundles
        if any(
            stored.get(p.key) != p.fingerprint
            for p in _plans(bundle, inputs, by_game[bundle.appid])
        )
    ]
    return sorted(due, key=lambda b: (b.start, b.appid), reverse=True)
