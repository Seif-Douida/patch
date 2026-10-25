"""The agent's data bundle (design D2): everything the core tools read, as aggregates.

The nightly `bundle` stage builds it from the marts and app.patch_effect after the effects stage,
for the tracked games, with days up to yesterday; `pp-api`, the nightly reports and the tests read
it instead of SQL. It never carries review text or author fields: every field name is on
ALLOWED_FIELDS and `dump_bundle` refuses anything else.

The file is gzip JSON with sorted keys and no timestamp in the gzip header, so the same bundle
always gives the same bytes. Its `schema` is BUNDLE_SCHEMA; a reader refuses any other. Estimate
rows are Phase 4's StoredEstimate as stored, so `causal.store.serve` takes them unchanged.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import Connection, text

from patchpulse.causal import bundles as patch_bundles
from patchpulse.causal.config import CAUSAL_CONFIG_PATH, CausalConfig, read_causal_config
from patchpulse.causal.store import StoredEstimate, all_latest_estimates

BUNDLE_SCHEMA = 1


@dataclass(frozen=True)
class DayRow:
    """A game's day: its reviews and positive reviews, a Steam sale, and the game's discount in
    percent (None where no price snapshot was taken that day)."""

    date: date
    positives: int
    totals: int
    sale: bool
    discount: int | None


@dataclass(frozen=True)
class AspectRow:
    """A game's day with scored reviews, for one aspect: how many of the scored negative and
    positive reviews mention it."""

    date: date
    aspect: str
    neg_mentions: int
    neg_scored: int
    pos_mentions: int
    pos_scored: int


@dataclass(frozen=True)
class BucketRow:
    """A game's day, for one playtime bucket at review (decompose_change's input)."""

    date: date
    bucket: str
    positives: int
    totals: int


@dataclass(frozen=True)
class PatchRow:
    """An official Steam post with its classifier label; `bundle_id` is its estimate bundle's
    (treatment candidates only, bundled as the effects stage bundles them)."""

    patch_id: str
    gid: str  # Steam's news item id
    published_at: datetime  # UTC
    day: date
    title: str
    patch_type: str
    bundle_id: str | None


@dataclass(frozen=True)
class GameData:
    appid: int
    name: str
    genres: str  # as stored in core.dim_game: "Action, RPG"
    days: tuple[DayRow, ...]
    aspects: tuple[AspectRow, ...]
    buckets: tuple[BucketRow, ...]
    patches: tuple[PatchRow, ...]
    estimates: tuple[StoredEstimate, ...]


@dataclass(frozen=True)
class Bundle:
    data_version: int  # the night's run id
    built_on: date
    causal: CausalConfig
    games: tuple[GameData, ...]


def _names(*types: type) -> frozenset[str]:
    return frozenset(f.name for t in types for f in fields(t))


ALLOWED_FIELDS = _names(
    Bundle, CausalConfig, GameData, DayRow, AspectRow, BucketRow, PatchRow, StoredEstimate
) | {"schema"}


def _plain(value: Any) -> Any:
    """A JSON-able copy: dataclasses become objects (their fields checked against the
    allowlist), dates ISO strings, sets sorted lists, tuples lists."""
    if is_dataclass(value) and not isinstance(value, type):
        out: dict[str, Any] = {}
        for f in fields(value):
            if f.name not in ALLOWED_FIELDS:
                raise ValueError(f"the bundle may not carry the field {f.name!r}")
            out[f.name] = _plain(getattr(value, f.name))
        return out
    if isinstance(value, date):  # datetimes too
        return value.isoformat()
    if isinstance(value, frozenset | set):
        return sorted(_plain(v) for v in value)
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    return value


def dump_bundle(bundle: Bundle) -> bytes:
    document = {"schema": BUNDLE_SCHEMA, **_plain(bundle)}
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return gzip.compress(raw.encode("utf-8"), compresslevel=9, mtime=0)


def _estimate(row: dict[str, Any]) -> StoredEstimate:
    return StoredEstimate(
        **{
            **row,
            "patch_ids": tuple(row["patch_ids"]),
            "donors": tuple((int(a), float(w)) for a, w in row["donors"]),
            "flags": frozenset(row["flags"]),
            "computed_at": datetime.fromisoformat(row["computed_at"]),
        }
    )


def _game(g: dict[str, Any]) -> GameData:
    return GameData(
        appid=g["appid"],
        name=g["name"],
        genres=g["genres"],
        days=tuple(DayRow(**{**d, "date": date.fromisoformat(d["date"])}) for d in g["days"]),
        aspects=tuple(
            AspectRow(**{**a, "date": date.fromisoformat(a["date"])}) for a in g["aspects"]
        ),
        buckets=tuple(
            BucketRow(**{**b, "date": date.fromisoformat(b["date"])}) for b in g["buckets"]
        ),
        patches=tuple(
            PatchRow(
                **{
                    **p,
                    "published_at": datetime.fromisoformat(p["published_at"]),
                    "day": date.fromisoformat(p["day"]),
                }
            )
            for p in g["patches"]
        ),
        estimates=tuple(_estimate(e) for e in g["estimates"]),
    )


def load_bundle(data: bytes) -> Bundle:
    document = json.loads(gzip.decompress(data))
    schema = document.get("schema") if isinstance(document, dict) else None
    if schema != BUNDLE_SCHEMA:
        raise ValueError(f"unknown bundle schema {schema!r}; this code reads {BUNDLE_SCHEMA}")
    return Bundle(
        data_version=document["data_version"],
        built_on=date.fromisoformat(document["built_on"]),
        causal=CausalConfig(**document["causal"]),
        games=tuple(_game(g) for g in document["games"]),
    )


def _day(key: int) -> date:
    return date(key // 10_000, key // 100 % 100, key % 100)


def _key(day: date) -> int:
    return day.year * 10_000 + day.month * 100 + day.day


def build_bundle(
    connection: Connection,
    *,
    run_id: int,
    today: date,
    causal_config_path: Path = CAUSAL_CONFIG_PATH,
) -> Bundle:
    """The bundle for the night `run_id`: the tracked games' marts up to yesterday, their patches
    and the latest row of every stored estimate key, with config/causal.yaml."""
    through = {"last": _key(today - timedelta(days=1))}
    games = {
        int(r.appid): (str(r.name), str(r.genres))
        for r in connection.execute(
            text("SELECT appid, name, genres FROM core.dim_game WHERE role <> 'donor'")
        )
    }
    days: dict[int, list[DayRow]] = {a: [] for a in games}
    for r in connection.execute(
        text(
            "SELECT game_key, date_key, n_positive, n_reviews, is_sale, discount_pct "
            "FROM mart.daily_game_sentiment WHERE date_key <= :last"
        ),
        through,
    ):
        if int(r.game_key) in days:
            discount = None if r.discount_pct is None else int(r.discount_pct)
            days[int(r.game_key)].append(
                DayRow(
                    date=_day(r.date_key),
                    positives=int(r.n_positive),
                    totals=int(r.n_reviews),
                    sale=bool(r.is_sale),
                    discount=discount,
                )
            )
    aspects: dict[int, list[AspectRow]] = {a: [] for a in games}
    for r in connection.execute(
        text(
            "SELECT game_key, date_key, aspect_key, n_neg_mentions, n_neg_scored, "
            "n_pos_mentions, n_pos_scored FROM mart.daily_aspect_share "
            "WHERE date_key <= :last AND n_neg_scored + n_pos_scored > 0"
        ),
        through,
    ):
        if int(r.game_key) in aspects:
            aspects[int(r.game_key)].append(
                AspectRow(
                    _day(r.date_key),
                    str(r.aspect_key),
                    int(r.n_neg_mentions),
                    int(r.n_neg_scored),
                    int(r.n_pos_mentions),
                    int(r.n_pos_scored),
                )
            )
    buckets: dict[int, list[BucketRow]] = {a: [] for a in games}
    for r in connection.execute(
        text(
            "SELECT game_key, date_key, playtime_bucket, n_positive, n_reviews "
            "FROM mart.daily_playtime_sentiment WHERE date_key <= :last"
        ),
        through,
    ):
        if int(r.game_key) in buckets:
            buckets[int(r.game_key)].append(
                BucketRow(
                    date=_day(r.date_key),
                    bucket=str(r.playtime_bucket),
                    positives=int(r.n_positive),
                    totals=int(r.n_reviews),
                )
            )
    posts = [
        r
        for r in connection.execute(
            text(
                "SELECT patch_id, game_key, gid, published_at, date_key, title, patch_type, "
                "is_treatment_candidate FROM core.fact_patch"
            )
        )
        if int(r.game_key) in games
    ]
    bundle_of = {
        patch_id: b.bundle_id
        for b in patch_bundles.bundle_patches(
            [
                patch_bundles.Patch(str(r.patch_id), int(r.game_key), _day(r.date_key))
                for r in posts
                if r.is_treatment_candidate
            ]
        )
        for patch_id in b.patch_ids
    }
    patches: dict[int, list[PatchRow]] = {a: [] for a in games}
    for r in posts:
        patches[int(r.game_key)].append(
            PatchRow(
                patch_id=str(r.patch_id),
                gid=str(r.gid),
                published_at=r.published_at.replace(tzinfo=UTC),
                day=_day(r.date_key),
                title=str(r.title),
                patch_type=str(r.patch_type),
                bundle_id=bundle_of.get(str(r.patch_id)),
            )
        )
    estimates: dict[int, list[StoredEstimate]] = {a: [] for a in games}
    for e in all_latest_estimates(connection):
        if e.appid in estimates:
            estimates[e.appid].append(e)
    return Bundle(
        data_version=run_id,
        built_on=today,
        causal=read_causal_config(causal_config_path),
        games=tuple(
            GameData(
                appid=appid,
                name=name,
                genres=genres,
                days=tuple(sorted(days[appid], key=lambda d: d.date)),
                aspects=tuple(sorted(aspects[appid], key=lambda a: (a.date, a.aspect))),
                buckets=tuple(sorted(buckets[appid], key=lambda b: (b.date, b.bucket))),
                patches=tuple(sorted(patches[appid], key=lambda p: (p.published_at, p.patch_id))),
                estimates=tuple(
                    sorted(
                        estimates[appid],
                        key=lambda e: (e.bundle_id, e.outcome, e.sample, e.horizon, e.method),
                    )
                ),
            )
            for appid, (name, genres) in sorted(games.items())
        ),
    )
