"""The nightly effects stage (Phase 4 Task 13, design D10), on the SQL Server container.

The module builds the marts once with the real dbt project from fixture reviews, comparison games'
daily counts, prices and patches, all in 2022, before the sale calendar's first sale. Each test
starts from an empty app.patch_effect; the refill test's extra review changes nothing the other
tests read.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import SecretStr
from sqlalchemy import Engine, insert, text

from patchpulse.causal.bundles import Patch
from patchpulse.causal.estimate import GameInfo, estimate_bundle
from patchpulse.causal.its import POSITIVE_SHARE, VOLUME, aspect_outcome
from patchpulse.config import Settings
from patchpulse.db import tables
from patchpulse.db.engine import make_engine
from patchpulse.db.migrate import upgrade_to_head
from patchpulse.db.provision import WRITER_USER, provision_writer
from patchpulse.ingest.games import Game, Role
from patchpulse.ingest.load import (
    upsert_classifications,
    upsert_games,
    upsert_news,
    upsert_prices,
    upsert_reviews,
)
from patchpulse.ingest.models import NewsItem, PriceSnapshot, ReviewRecord
from patchpulse.ingest.patches import OFFICIAL_FEED, classify
from patchpulse.pipeline import effects
from patchpulse.pipeline.dbt import run_dbt
from patchpulse.pipeline.effects import EffectsOutcome, read_inputs, run_effects

pytestmark = pytest.mark.integration

GAME = 9_100_001  # tracked, Action
DONORS = tuple(range(9_100_101, 9_100_107))  # six comparison games, Action
PATCHED_DONOR = DONORS[-1]  # its own patch keeps it out of the first bundle's twins
FIRST = date(2022, 3, 1)  # the game's first review; the sale calendar starts in 2023
TODAY = date(2022, 9, 1)
LAST = TODAY - timedelta(days=1)  # the night's last complete day
NOW = datetime(2022, 9, 1, 3, tzinfo=UTC)
P1 = date(2022, 7, 20)  # its own bundle: 7 and 14 days complete
P2, P3 = date(2022, 8, 25), date(2022, 8, 28)  # one bundle: 7 days complete, 14 not yet
DONOR_PATCH = date(2022, 7, 10)
B1, B2 = "7001", "7002"  # a bundle's id is its first patch's
PRICES_FROM = date(2022, 7, 1)
DISCOUNT_DAY = date(2022, 8, 24)  # the day before P2: its bundle is flagged
REFILLED = date(2022, 8, 10)  # after the first bundle's 14 days, in the second's fit
WRITER_PASSWORD = "Writer-Effects-1"


def days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=k) for k in range((last - first).days + 1)]


def review(rid: int, created: datetime, *, voted_up: bool, minutes: int) -> ReviewRecord:
    return ReviewRecord(
        recommendation_id=rid,
        appid=GAME,
        author_hash=f"{rid:064d}",
        author_num_games_owned=None,
        author_num_reviews=3,
        author_playtime_forever=minutes,
        author_playtime_at_review=minutes,
        author_playtime_last_two_weeks=0,
        language="english",
        review_text="A review",
        timestamp_created=created,
        timestamp_updated=created,
        voted_up=voted_up,
        votes_up=0,
        votes_funny=0,
        weighted_vote_score=0.5,
        comment_count=0,
        steam_purchase=True,
        received_for_free=False,
        written_during_early_access=False,
    )


def reviews() -> list[ReviewRecord]:
    """10 to 16 reviews a day; 80% positive, then 35% from the first patch on. Every other review
    has 10+ hours at review (an established player)."""
    rng = np.random.default_rng(20220901)
    records: list[ReviewRecord] = []
    for day in days(FIRST, LAST):
        share = 0.8 if day < P1 else 0.35
        for k in range(int(rng.integers(10, 17))):
            created = datetime(day.year, day.month, day.day, 12, k, tzinfo=UTC)
            voted_up = bool(rng.random() < share)
            records.append(
                review(len(records) + 1, created, voted_up=voted_up, minutes=700 if k % 2 else 30)
            )
    return records


def donor_counts() -> list[dict[str, Any]]:
    rng = np.random.default_rng(20220902)
    rows: list[dict[str, Any]] = []
    for appid in DONORS:
        for day in days(FIRST, LAST):
            positive = int(rng.binomial(40, 0.8))
            rows.append(
                {
                    "appid": appid,
                    "day": day,
                    "positive": positive,
                    "negative": 40 - positive,
                    "fetched_run_id": 1,
                    "fetched_at": datetime(2022, 9, 1, 3),
                }
            )
    return rows


def news(gid: str, appid: int, day: date) -> NewsItem:
    return NewsItem(
        gid=gid,
        appid=appid,
        title=f"Patch 1.{gid}",
        url=f"https://store.steampowered.com/news/{gid}",
        contents="Fixes",
        published_at=datetime(day.year, day.month, day.day, 17, tzinfo=UTC),
        feedname=OFFICIAL_FEED,
        feedlabel="Community Announcements",
    )


def game(appid: int, role: Role) -> Game:
    return Game(
        appid=appid, name=f"Game {appid}", genres=["Action"], role=role, backfill_start=FIRST
    )


@dataclass(frozen=True)
class World:
    engine: Engine
    writer: Settings
    target: Callable[[], Path]

    def build(self) -> None:
        result = run_dbt(["build"], settings=self.writer, target_path=self.target())
        assert result.ok, result.failures


@pytest.fixture(scope="module")
def world(new_database: Callable[[], Settings], tmp_path_factory: pytest.TempPathFactory) -> World:
    admin_settings = new_database()
    engine = make_engine(admin_settings)
    upgrade_to_head(engine)
    items = [
        news("7001", GAME, P1),
        news("7002", GAME, P2),
        news("7003", GAME, P3),
        news("7004", PATCHED_DONOR, DONOR_PATCH),
    ]
    with engine.begin() as connection:
        provision_writer(connection, password=WRITER_PASSWORD)
        games = [game(GAME, Role.TRACKED), *(game(appid, Role.DONOR) for appid in DONORS)]
        upsert_games(connection, games, run_id=1)
        upsert_reviews(connection, reviews(), run_id=1)
        connection.execute(insert(tables.steam_review_daily), donor_counts())
        upsert_news(connection, items, run_id=1)
        upsert_classifications(connection, {i.gid: classify(i) for i in items}, run_id=1)
        for day in days(PRICES_FROM, LAST):
            discount = 25 if day == DISCOUNT_DAY else 0
            price = PriceSnapshot(
                currency="USD",
                initial=3999,
                final=3999 * (100 - discount) // 100,
                discount_percent=discount,
            )
            upsert_prices(connection, {GAME: price}, snapshot_date=day, run_id=1)
    writer = admin_settings.model_copy(
        update={"db_user": WRITER_USER, "db_password": SecretStr(WRITER_PASSWORD)}
    )
    built = World(engine, writer, lambda: tmp_path_factory.mktemp("dbt"))
    built.build()
    return built


@pytest.fixture
def marts(world: World) -> World:
    with world.engine.begin() as connection:
        connection.execute(text("DELETE FROM app.patch_effect"))
    return world


def night(world: World, run_id: int) -> EffectsOutcome:
    """A night with time to spare."""
    return run_effects(world.engine, run_id=run_id, now=NOW, clock=lambda: 0.0, deadline=60.0)


def stored(engine: Engine) -> list[Any]:
    with engine.connect() as connection:
        return list(connection.execute(text("SELECT * FROM app.patch_effect")))


def aspect_codes(engine: Engine) -> list[str]:
    with engine.connect() as connection:
        return sorted(r.code for r in connection.execute(text("SELECT code FROM core.dim_aspect")))


def keys_for(engine: Engine, bundle_id: str, horizon: int) -> set[tuple[str, str, str, int, str]]:
    """Every row a complete horizon gets: interrupted time series for the positive share, volume,
    each aspect and established players; synthetic control for the positive share and volume."""
    its = [POSITIVE_SHARE, VOLUME, *(aspect_outcome(code) for code in aspect_codes(engine))]
    return {
        *((bundle_id, outcome, "all", horizon, "its") for outcome in its),
        (bundle_id, POSITIVE_SHARE, "established", horizon, "its"),
        (bundle_id, POSITIVE_SHARE, "all", horizon, "synth"),
        (bundle_id, VOLUME, "all", horizon, "synth"),
    }


def key(row: Any) -> tuple[str, str, str, int, str]:
    return (row.bundle_id, row.outcome, row.sample, row.horizon, row.method)


def test_read_inputs_maps_the_marts(marts: World) -> None:
    with marts.engine.connect() as connection:
        inputs = read_inputs(connection, today=TODAY)
        positive, total = connection.execute(
            text(
                "SELECT SUM(CAST(voted_up AS int)), COUNT(*) FROM core.fact_review "
                f"WHERE game_key = {GAME}"
            )
        ).one()

    assert set(inputs.series) == {GAME}  # comparison games are no series of their own
    series = inputs.series[GAME]
    assert (series.first_day, series.last_day) == (FIRST, LAST)  # today is never complete
    assert series.observed.all()
    assert (int(series.positives.sum()), int(series.totals.sum())) == (positive, total)
    established = inputs.established[GAME]
    assert (established.first_day, established.last_day) == (FIRST, LAST)
    assert 0 < established.totals.sum() < series.totals.sum()
    assert sorted(inputs.aspects[GAME]) == aspect_codes(marts.engine)
    for aspect in inputs.aspects[GAME].values():  # nothing is scored: no negatives to share
        assert (aspect.first_day, aspect.last_day, int(aspect.totals.sum())) == (FIRST, LAST, 0)
    assert set(inputs.donors) == set(DONORS)
    donor = inputs.donors[DONORS[0]]
    assert (donor.first_day, donor.last_day) == (FIRST, LAST)
    assert (donor.totals == 40).all()
    assert donor.observed.all()
    assert inputs.games[GAME] == GameInfo(GAME, ("Action",), "tracked")
    assert inputs.games[PATCHED_DONOR].role == "donor"
    assert list(inputs.patches[GAME]) == [
        Patch("7001", GAME, P1),
        Patch("7002", GAME, P2),
        Patch("7003", GAME, P3),
    ]
    assert list(inputs.patches[PATCHED_DONOR]) == [Patch("7004", PATCHED_DONOR, DONOR_PATCH)]
    assert min(inputs.sale_days) == date(2023, 3, 16)  # the calendar's first sale
    assert inputs.discounts == {GAME: frozenset({DISCOUNT_DAY})}
    assert inputs.discounts_known_from == PRICES_FROM
    assert inputs.last_complete_day == LAST


def test_first_night_estimates_every_complete_bundle(
    marts: World, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="patchpulse.pipeline")

    outcome = night(marts, run_id=1)

    # The first bundle's 7 and 14 days, the second's 7 (its 14 days end after tonight).
    assert outcome == EffectsOutcome(estimated=2, written=45, unchanged=0, remaining=0, failed=0)
    rows = stored(marts.engine)
    assert {key(r) for r in rows} == (
        keys_for(marts.engine, B1, 7)
        | keys_for(marts.engine, B1, 14)
        | keys_for(marts.engine, B2, 7)
    )
    assert {r.run_id for r in rows} == {1}
    by_key = {key(r): r for r in rows}
    drop = by_key[(B1, POSITIVE_SHARE, "all", 7, "its")]
    assert drop.hi90 < 0  # the planted fall from 80% to 35% positive, clear of 0
    twin = by_key[(B1, POSITIVE_SHARE, "all", 7, "synth")]
    donors = {appid for appid, _ in json.loads(twin.donors)}
    assert donors  # five comparison games qualified, so the fit ran
    assert PATCHED_DONOR not in donors  # its own patch is in the fit window
    assert json.loads(by_key[(B2, POSITIVE_SHARE, "all", 7, "its")].patch_ids) == [B2, "7003"]
    assert all("confounded" in r.flags for r in rows if r.bundle_id == B2)  # the discount
    waiting = by_key[(B2, POSITIVE_SHARE, "all", 7, "synth")]
    assert (waiting.effect, waiting.reason) == (
        None,
        "the comparison games are checked over the 14 days after the patch, which end on "
        "7 Sep 2022",
    )
    assert not any("confounded" in r.flags for r in rows if r.bundle_id == B1)
    (line,) = [r.getMessage() for r in caplog.records if r.name == "patchpulse.pipeline.effects"]
    assert line.startswith("effects: 2 bundles estimated, 45 rows written")


def test_second_night_writes_nothing_new(marts: World) -> None:
    first = night(marts, run_id=1)
    second = night(marts, run_id=2)

    assert first.written == 45
    assert second == EffectsOutcome(estimated=0, written=0, unchanged=2, remaining=0, failed=0)
    assert {r.run_id for r in stored(marts.engine)} == {1}


def test_a_refilled_history_day_recomputes_its_bundle(marts: World) -> None:
    # Review focus 3: the refill adds a review to a day the second bundle reads and the first
    # doesn't. Only the second bundle's rows that read that day get a new row; the rest stay.
    night(marts, run_id=1)
    late = datetime(REFILLED.year, REFILLED.month, REFILLED.day, 18, tzinfo=UTC)
    with marts.engine.begin() as connection:
        upsert_reviews(connection, [review(900_001, late, voted_up=True, minutes=30)], run_id=2)
    marts.build()

    outcome = night(marts, run_id=3)

    assert outcome == EffectsOutcome(estimated=1, written=2, unchanged=1, remaining=0, failed=0)
    rows = stored(marts.engine)
    assert {key(r) for r in rows if r.run_id == 3} == {
        (B2, POSITIVE_SHARE, "all", 7, "its"),
        (B2, VOLUME, "all", 7, "its"),
    }  # its synthetic control waits for day 13 and doesn't read the day yet
    assert sum(r.run_id == 1 for r in rows) == 45  # the earlier rows stay readable


def test_budget_stops_and_the_next_night_resumes(marts: World) -> None:
    def clock() -> float:  # the night's time runs out once a bundle is stored
        return 100.0 if stored(marts.engine) else 0.0

    first = run_effects(marts.engine, run_id=1, now=NOW, clock=clock, deadline=50.0)

    assert first == EffectsOutcome(estimated=1, written=15, unchanged=0, remaining=1, failed=0)
    assert {r.bundle_id for r in stored(marts.engine)} == {B2}  # newest first

    second = night(marts, run_id=2)

    assert second == EffectsOutcome(estimated=1, written=30, unchanged=1, remaining=0, failed=0)
    assert {(r.bundle_id, r.run_id) for r in stored(marts.engine)} == {(B2, 1), (B1, 2)}


def test_a_failing_bundle_is_a_warning_and_the_others_go_on(
    marts: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review focus 4: newest first, so one bundle failing every night would block all the rest.
    real = estimate_bundle

    def fails_on_b2(bundle: Any, inputs: Any) -> Any:
        if bundle.bundle_id == B2:
            raise ValueError("a series the estimators choke on")
        return real(bundle, inputs)

    monkeypatch.setattr(effects, "estimate_bundle", fails_on_b2)

    outcome = night(marts, run_id=1)

    assert outcome == EffectsOutcome(estimated=1, written=30, unchanged=0, remaining=0, failed=1)
    assert {r.bundle_id for r in stored(marts.engine)} == {B1}
    with marts.engine.connect() as connection:
        warnings = connection.execute(
            text(
                "SELECT severity, appid, detail FROM ops.data_quality_result "
                "WHERE check_name = 'effects_bundle_failed'"
            )
        ).all()
    assert [(w.severity, w.appid) for w in warnings] == [("warn", GAME)]
    assert warnings[0].detail == f"bundle {B2}: ValueError: a series the estimators choke on"
