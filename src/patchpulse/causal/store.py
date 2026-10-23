"""app.patch_effect, the estimates the nightly effects stage stores, and Phase 5's two readers
(design D10, spec §9.4).

Rows are append-only: a key (bundle, outcome, sample, horizon, method) gets a new row only when its
fingerprint changes (new inputs or a new EFFECTS_VERSION), so its latest row is its current
estimate and `as_of` reads an earlier night's. `patch_ids` and `donors` are stored as JSON and
`flags` as a comma-separated list; the effects and intervals are REAL (single precision).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Connection, func, insert, select, text

from patchpulse.causal.config import CAUSAL_CONFIG_PATH, read_causal_config
from patchpulse.causal.decompose import BucketCounts, Decomposition, decompose
from patchpulse.causal.estimate import Estimate, EstimateKey, Method, Sample
from patchpulse.db import tables


@dataclass(frozen=True)
class StoredEstimate:
    """One row of app.patch_effect: an Estimate with the run that stored it."""

    bundle_id: str
    patch_ids: tuple[str, ...]
    appid: int
    outcome: str
    sample: str
    horizon: int
    method: str
    run_id: int
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
    computed_at: datetime

    @property
    def key(self) -> EstimateKey:
        return EstimateKey(
            self.bundle_id, self.outcome, Sample(self.sample), self.horizon, Method(self.method)
        )


_KEY = ("bundle_id", "outcome", "sample", "horizon", "method")


def _latest(*where: Any, as_of: int | None = None) -> Any:
    """The latest row per key among the rows matching `where` (and run_id <= as_of)."""
    table = tables.patch_effect
    rank = (
        func.row_number()
        .over(partition_by=[table.c[k] for k in _KEY], order_by=table.c.run_id.desc())
        .label("rank")
    )
    conditions = list(where)
    if as_of is not None:
        conditions.append(table.c.run_id <= as_of)
    ranked = select(table, rank).where(*conditions).subquery()
    return select(ranked).where(ranked.c.rank == 1)


def stored_fingerprints(connection: Connection) -> dict[EstimateKey, str]:
    """The latest row's fingerprint for every key stored."""
    return {
        EstimateKey(r.bundle_id, r.outcome, Sample(r.sample), r.horizon, Method(r.method)): (
            r.fingerprint
        )
        for r in connection.execute(_latest())
    }


def _row(estimate: Estimate, run_id: int, computed_at: datetime) -> dict[str, Any]:
    return {
        "bundle_id": estimate.bundle_id,
        "patch_ids": json.dumps(list(estimate.patch_ids)),
        "appid": estimate.appid,
        "outcome": estimate.outcome,
        "sample": str(estimate.sample),
        "horizon": estimate.horizon,
        "method": str(estimate.method),
        "run_id": run_id,
        "effect": estimate.effect,
        "lo90": estimate.lo90,
        "hi90": estimate.hi90,
        "p_value": estimate.p_value,
        "pre_fit_rmse": estimate.pre_fit_rmse,
        "donors": json.dumps([[appid, weight] for appid, weight in estimate.donors]),
        "flags": ",".join(sorted(estimate.flags)),
        "reason": estimate.reason,
        "fingerprint": estimate.fingerprint,
        "version": estimate.version,
        "computed_at": computed_at.astimezone(UTC).replace(tzinfo=None),
    }


def write_estimates(
    connection: Connection, estimates: Iterable[Estimate], *, run_id: int, computed_at: datetime
) -> int:
    """Insert the estimates whose fingerprint differs from their key's latest row; returns how
    many were written."""
    latest = stored_fingerprints(connection)
    new = [e for e in estimates if latest.get(e.key) != e.fingerprint]
    if new:
        connection.execute(insert(tables.patch_effect), [_row(e, run_id, computed_at) for e in new])
    return len(new)


def _stored(row: Any) -> StoredEstimate:
    return StoredEstimate(
        bundle_id=row.bundle_id,
        patch_ids=tuple(json.loads(row.patch_ids)),
        appid=row.appid,
        outcome=row.outcome,
        sample=row.sample,
        horizon=row.horizon,
        method=row.method,
        run_id=row.run_id,
        effect=row.effect,
        lo90=row.lo90,
        hi90=row.hi90,
        p_value=row.p_value,
        pre_fit_rmse=row.pre_fit_rmse,
        donors=tuple((int(a), float(w)) for a, w in json.loads(row.donors)),
        flags=frozenset(f for f in row.flags.split(",") if f),
        reason=row.reason,
        fingerprint=row.fingerprint,
        version=row.version,
        computed_at=row.computed_at.replace(tzinfo=UTC),
    )


def _bundle_of(connection: Connection, patch_id: str, as_of: int | None) -> str | None:
    """The bundle holding `patch_id` in the latest run that stored it (a bundle's id is its first
    patch's, so a patch added before it later changes the bundle)."""
    found = connection.execute(
        text(
            "SELECT TOP 1 bundle_id FROM app.patch_effect "
            "WHERE EXISTS (SELECT 1 FROM OPENJSON(patch_ids) WHERE value = :patch_id) "
            "AND (:as_of IS NULL OR run_id <= :as_of) ORDER BY run_id DESC"
        ),
        {"patch_id": patch_id, "as_of": as_of},
    ).first()
    return None if found is None else str(found.bundle_id)


def estimate_patch_effect(
    connection: Connection,
    patch_id: str,
    outcome: str = "positive_share",
    method: str | None = None,
    *,
    sample: str = "all",
    as_of: int | None = None,
    config_path: Path = CAUSAL_CONFIG_PATH,
) -> list[StoredEstimate]:
    """The estimates for the bundle holding `patch_id`, one per horizon, as of run `as_of` (the
    latest by default). `method=None` means config/causal.yaml's default; where the default has no
    estimate, the interrupted time series row is returned instead, carrying the default's reason."""
    bundle_id = _bundle_of(connection, patch_id, as_of)
    if bundle_id is None:
        return []
    chosen = method or read_causal_config(config_path).default_method
    table = tables.patch_effect
    rows = [
        _stored(r)
        for r in connection.execute(
            _latest(
                table.c.bundle_id == bundle_id,
                table.c.outcome == outcome,
                table.c.sample == sample,
                table.c.method.in_({chosen, Method.ITS.value}),
                as_of=as_of,
            )
        )
    ]
    by_key = {(r.horizon, r.method): r for r in rows}
    served: list[StoredEstimate] = []
    for horizon in sorted({r.horizon for r in rows}):
        default = by_key.get((horizon, chosen))
        its = by_key.get((horizon, Method.ITS.value))
        if default is None:
            continue
        if method is None and default.effect is None and its is not None and its.effect is not None:
            served.append(replace(its, reason=default.reason))
        else:
            served.append(default)
    return served


def _bucket_counts(connection: Connection, appid: int, window: tuple[date, date]) -> BucketCounts:
    first, last = window
    if first > last:
        raise ValueError(f"a window's first day {first} is after its last {last}")
    rows = connection.execute(
        text(
            "SELECT playtime_bucket, SUM(n_positive) AS positive, SUM(n_reviews) AS total "
            "FROM mart.daily_playtime_sentiment WHERE game_key = :appid "
            "AND date_key BETWEEN :first AND :last GROUP BY playtime_bucket"
        ),
        {"appid": appid, "first": _date_key(first), "last": _date_key(last)},
    )
    return {r.playtime_bucket: (int(r.positive), int(r.total)) for r in rows}


def _date_key(day: date) -> int:
    return day.year * 10_000 + day.month * 100 + day.day


def decompose_change(
    connection: Connection,
    appid: int,
    window: tuple[date, date],
    baseline_window: tuple[date, date],
) -> Decomposition:
    """Who reviewed against how they felt (spec §9.3): the change in a game's positive share from
    `baseline_window` to `window` (first and last days, both included), split by playtime at
    review, from mart.daily_playtime_sentiment."""
    return decompose(
        _bucket_counts(connection, appid, baseline_window),
        _bucket_counts(connection, appid, window),
    )
