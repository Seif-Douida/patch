"""The bundle built from the marts (plan Task 2) under `dbt build` on the SQL Server container.

Two tracked games and a comparison game: reviews over three days and today, aspect scores, two
patches in one bundle and an event post, a price snapshot, and estimates stored over two runs. The
bundle must hold exactly the marts' rows up to yesterday and the latest estimate of each key, for
the tracked games only, and never a review's text.
"""

from __future__ import annotations

import gzip
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from pydantic import SecretStr
from sqlalchemy import Engine, insert, text

from patchpulse.agent.bundle import Bundle, build_bundle, dump_bundle
from patchpulse.causal.config import CausalConfig, read_causal_config, write_causal_config
from patchpulse.causal.estimate import EFFECTS_VERSION, Estimate, Method, Sample
from patchpulse.causal.its import POSITIVE_SHARE, VOLUME, aspect_outcome
from patchpulse.causal.store import latest_estimates, write_estimates
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
from patchpulse.ingest.patches import classify
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.pipeline.dbt import run_dbt

pytestmark = pytest.mark.integration

TODAY = datetime.now(UTC).replace(hour=1, minute=0, second=0, microsecond=0)
DAY1, DAY2, DAY3 = (TODAY - timedelta(days=d) for d in (6, 5, 4))
GAME_A, GAME_B, DONOR = 553850, 949230, 1086940
WRITER_PASSWORD = "Writer-Bundle-1"
SCORER = "s0-20261013T000000Z"


def text_of(rid: int) -> str:
    return f"Review {rid} says the frame pacing stutters badly since patch day, refunded {rid}"


def review(
    rid: int, appid: int, created: datetime, *, voted_up: bool, minutes: int | None
) -> ReviewRecord:
    return ReviewRecord(
        recommendation_id=rid,
        appid=appid,
        author_hash=f"{rid:064d}",
        author_num_games_owned=10,
        author_num_reviews=2,
        author_playtime_forever=900,
        author_playtime_at_review=minutes,
        author_playtime_last_two_weeks=0,
        language="english",
        review_text=text_of(rid),
        timestamp_created=created,
        timestamp_updated=created,
        voted_up=voted_up,
        votes_up=1,
        votes_funny=0,
        weighted_vote_score=0.5,
        comment_count=0,
        steam_purchase=True,
        received_for_free=False,
        written_during_early_access=False,
    )


REVIEWS = [
    review(1, GAME_A, DAY1, voted_up=True, minutes=60),
    review(2, GAME_A, DAY1, voted_up=True, minutes=900),
    review(3, GAME_A, DAY1, voted_up=False, minutes=30),
    review(4, GAME_A, DAY1, voted_up=False, minutes=None),
    review(5, GAME_A, DAY3, voted_up=False, minutes=13000),
    review(6, GAME_A, TODAY, voted_up=False, minutes=60),  # today: not in the bundle
    review(7, GAME_B, DAY2, voted_up=True, minutes=400),
]


def score(rid: int, created: datetime, positives: set[str]) -> dict[str, Any]:
    return {
        "recommendation_id": rid,
        "timestamp_updated": created.replace(tzinfo=None),
        "model_version": SCORER,
        "model_name": "aspect-classifier",
        "scored_run_id": 1,
        "scored_at": TODAY.replace(tzinfo=None),
        **{f"p_{code}": 0.9 if code in positives else 0.1 for code in ASPECTS},
        **{f"l_{code}": code in positives for code in ASPECTS},
    }


def news(gid: str, appid: int, title: str, published: datetime) -> NewsItem:
    return NewsItem(
        gid=gid,
        appid=appid,
        title=title,
        url=f"https://example.com/{gid}",
        contents="",
        published_at=published,
        feedname="steam_community_announcements",
        feedlabel="",
        tags=["patchnotes"] if "atch" in title else [],
    )


NEWS = [
    news("100", GAME_A, "Patch 7.1.1", DAY1 + timedelta(hours=15)),
    news("102", GAME_A, "Patch 7.1.2", DAY3 + timedelta(hours=9)),  # joins bundle 100
    news("103", GAME_A, "Halloween event begins", DAY2),
    news("900", DONOR, "Patch 2.0.4", DAY2),  # a comparison game's: not in the bundle
]


def estimate(
    horizon: int,
    method: Method,
    effect: float | None,
    *,
    outcome: str = POSITIVE_SHARE,
    sample: Sample = Sample.ALL,
    salt: str = "",
) -> Estimate:
    synth = method is Method.SYNTH
    return Estimate(
        bundle_id="100",
        patch_ids=("100", "102"),
        appid=GAME_A,
        outcome=outcome,
        sample=sample,
        horizon=horizon,
        method=method,
        effect=effect,
        lo90=None if effect is None else effect - 2.25,
        hi90=None if effect is None else effect + 1.5,
        p_value=0.125 if synth else None,
        pre_fit_rmse=0.75 if synth else None,
        donors=((DONOR, 1.0),) if synth else (),
        flags=frozenset({"volume_spike"}),
        reason=None if effect is not None else "interval wider than the grid",
        fingerprint=f"{horizon}{method}{outcome}{sample}{salt}".ljust(64, "0")[:64],
        version=EFFECTS_VERSION,
    )


@dataclass
class Built:
    admin: Engine
    bundle: Bundle
    config: CausalConfig

    def rows(self, sql: str, **params: Any) -> list[Any]:
        with self.admin.connect() as connection:
            return list(connection.execute(text(sql), params))


@pytest.fixture(scope="module")
def built(new_database: Callable[[], Settings], tmp_path_factory: pytest.TempPathFactory) -> Built:
    admin_settings = new_database()
    admin = make_engine(admin_settings)
    upgrade_to_head(admin)
    start = (TODAY - timedelta(days=30)).date()
    games = [
        Game(appid=GAME_A, name="Helldivers 2", genres=["Action"], role=Role.TRACKED,
             backfill_start=start),
        Game(appid=GAME_B, name="Cities: Skylines II", genres=["Simulation"], role=Role.BOTH,
             backfill_start=start),
        Game(appid=DONOR, name="Comparison", genres=["Action"], role=Role.DONOR,
             backfill_start=start),
    ]  # fmt: skip
    with admin.begin() as connection:
        provision_writer(connection, password=WRITER_PASSWORD)
        upsert_games(connection, games, run_id=1)
        upsert_reviews(connection, REVIEWS, run_id=1)
        upsert_news(connection, NEWS, run_id=1)
        upsert_classifications(connection, {i.gid: classify(i) for i in NEWS}, run_id=1)
        upsert_prices(
            connection,
            {GAME_A: PriceSnapshot(currency="USD", initial=3999, final=1999, discount_percent=50)},
            snapshot_date=DAY1.date(),
            run_id=1,
        )
        connection.execute(
            insert(tables.review_aspect_score),
            [
                score(1, DAY1, {"performance"}),
                score(2, DAY1, {"performance", "content"}),
                score(3, DAY1, {"performance", "stability_bugs"}),
                score(4, DAY1, set()),
                score(6, TODAY, {"performance"}),
                score(7, DAY2, {"content"}),
            ],
        )
    writer = admin_settings.model_copy(
        update={"db_user": WRITER_USER, "db_password": SecretStr(WRITER_PASSWORD)}
    )
    result = run_dbt(["build"], settings=writer, target_path=tmp_path_factory.mktemp("dbt"))
    assert result.ok, result.failures

    config = CausalConfig("synth", EFFECTS_VERSION, "cd" * 32, validated=True, spike_causal_ok=True)
    config_path = tmp_path_factory.mktemp("causal") / "causal.yaml"
    write_causal_config(config, config_path)
    first = [
        estimate(7, Method.ITS, -3.0),
        estimate(7, Method.SYNTH, -4.0),
        estimate(14, Method.SYNTH, None),
        estimate(7, Method.ITS, -9.0, sample=Sample.ESTABLISHED),
        estimate(7, Method.ITS, 12.0, outcome=VOLUME),
        estimate(7, Method.ITS, 6.5, outcome=aspect_outcome("performance")),
    ]
    later = [estimate(7, Method.ITS, -2.0, salt="new inputs")]  # supersedes the first ITS row
    with admin.begin() as connection:
        write_estimates(connection, first, run_id=1, computed_at=TODAY - timedelta(days=1))
        write_estimates(connection, later, run_id=2, computed_at=TODAY)
        bundle = build_bundle(
            connection, run_id=2, today=TODAY.date(), causal_config_path=config_path
        )
    return Built(admin, bundle, read_causal_config(config_path))


def by_game(rows: list[Any], *values: str) -> Mapping[int, set[tuple[Any, ...]]]:
    grouped: dict[int, set[tuple[Any, ...]]] = defaultdict(set)
    for r in rows:
        grouped[int(r.game_key)].add(tuple(getattr(r, v) for v in values))
    return grouped


def day_of(key: int) -> date:
    return datetime.strptime(str(key), "%Y%m%d").date()


def test_bundle_matches_the_marts(built: Built) -> None:
    bundle = built.bundle
    yesterday = int((TODAY - timedelta(days=1)).strftime("%Y%m%d"))
    assert (bundle.data_version, bundle.built_on, bundle.causal) == (2, TODAY.date(), built.config)
    assert [(g.appid, g.name, g.genres) for g in bundle.games] == [
        (GAME_A, "Helldivers 2", "Action"),
        (GAME_B, "Cities: Skylines II", "Simulation"),
    ]
    games = {g.appid: g for g in bundle.games}

    days = by_game(
        built.rows(
            "SELECT game_key, [date], n_positive, n_reviews, is_sale, discount_pct "
            "FROM mart.daily_game_sentiment WHERE date_key <= :last",
            last=yesterday,
        ),
        "date",
        "n_positive",
        "n_reviews",
        "is_sale",
        "discount_pct",
    )
    for appid, game in games.items():
        assert {(d.date, d.positives, d.totals, d.sale, d.discount) for d in game.days} == {
            (day, positives, totals, bool(sale), discount)
            for day, positives, totals, sale, discount in days[appid]
        }
    a_days = {d.date: d for d in games[GAME_A].days}
    assert (a_days[DAY1.date()].positives, a_days[DAY1.date()].totals) == (2, 4)
    assert a_days[DAY1.date()].discount == 50
    assert TODAY.date() not in a_days  # days up to yesterday

    aspects = by_game(
        built.rows(
            "SELECT game_key, date_key, aspect_key, n_neg_mentions, n_neg_scored, "
            "n_pos_mentions, n_pos_scored FROM mart.daily_aspect_share "
            "WHERE date_key <= :last AND n_neg_scored + n_pos_scored > 0",
            last=yesterday,
        ),
        "date_key",
        "aspect_key",
        "n_neg_mentions",
        "n_neg_scored",
        "n_pos_mentions",
        "n_pos_scored",
    )
    for appid, game in games.items():
        assert {
            (a.date, a.aspect, a.neg_mentions, a.neg_scored, a.pos_mentions, a.pos_scored)
            for a in game.aspects
        } == {(day_of(key), *rest) for key, *rest in aspects[appid]}
    performance = next(
        a for a in games[GAME_A].aspects if (a.date, a.aspect) == (DAY1.date(), "performance")
    )
    assert performance.neg_mentions / performance.neg_scored == 0.5  # review 3 of 3 and 4
    assert performance.pos_mentions / performance.pos_scored == 1.0
    assert {a.date for a in games[GAME_A].aspects} == {DAY1.date()}  # today's scores left out

    buckets = by_game(
        built.rows(
            "SELECT game_key, date_key, playtime_bucket, n_positive, n_reviews "
            "FROM mart.daily_playtime_sentiment WHERE date_key <= :last",
            last=yesterday,
        ),
        "date_key",
        "playtime_bucket",
        "n_positive",
        "n_reviews",
    )
    for appid, game in games.items():
        assert {(b.date, b.bucket, b.positives, b.totals) for b in game.buckets} == {
            (day_of(key), *rest) for key, *rest in buckets[appid]
        }
    day1 = {
        b.bucket: (b.positives, b.totals) for b in games[GAME_A].buckets if b.date == DAY1.date()
    }
    assert day1 == {"lt_2h": (1, 2), "10h_50h": (1, 1), "unknown": (0, 1)}

    assert [
        (p.patch_id, p.gid, p.day, p.title, p.patch_type, p.bundle_id)
        for p in games[GAME_A].patches
    ] == [
        ("100", "100", DAY1.date(), "Patch 7.1.1", "patch", "100"),
        ("103", "103", DAY2.date(), "Halloween event begins", "event", None),
        ("102", "102", DAY3.date(), "Patch 7.1.2", "patch", "100"),
    ]
    assert games[GAME_A].patches[0].published_at == DAY1 + timedelta(hours=15)
    assert games[GAME_B].patches == ()

    stored = built.rows(
        "SELECT bundle_id, outcome, sample, horizon, method, run_id FROM app.patch_effect"
    )
    latest: dict[tuple[Any, ...], int] = {}
    for r in stored:
        key = (r.bundle_id, r.outcome, r.sample, r.horizon, r.method)
        latest[key] = max(latest.get(key, 0), r.run_id)
    bundled = games[GAME_A].estimates
    assert {
        (e.bundle_id, e.outcome, e.sample, e.horizon, e.method): e.run_id for e in bundled
    } == latest
    assert len(bundled) == len(latest) == 6
    with built.admin.connect() as connection:
        served = latest_estimates(
            connection, outcome=POSITIVE_SHARE, sample=Sample.ALL, methods={"its", "synth"}
        )
    assert sorted(served, key=str) == sorted(
        (e for e in bundled if (e.outcome, e.sample) == (POSITIVE_SHARE, Sample.ALL)), key=str
    )
    assert games[GAME_B].estimates == ()


def test_bundle_holds_no_review_text(built: Built) -> None:
    raw = gzip.decompress(dump_bundle(built.bundle)).decode("utf-8")

    for r in REVIEWS:
        assert text_of(r.recommendation_id) not in raw
        assert r.author_hash not in raw
    assert "frame pacing" not in raw
