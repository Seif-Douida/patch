"""app.patch_effect and Phase 5's two readers (Phase 4 Task 12, design D10), on the SQL Server
container."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import SecretStr
from sqlalchemy import Engine

from patchpulse.causal.config import CausalConfig, write_causal_config
from patchpulse.causal.decompose import decompose
from patchpulse.causal.estimate import Estimate, Method, Sample
from patchpulse.causal.store import (
    decompose_change,
    estimate_patch_effect,
    stored_fingerprints,
    write_estimates,
)
from patchpulse.config import Settings
from patchpulse.db.engine import make_engine
from patchpulse.db.migrate import upgrade_to_head
from patchpulse.db.provision import WRITER_USER, provision_writer
from patchpulse.ingest.games import Game, Role
from patchpulse.ingest.load import upsert_games, upsert_reviews
from patchpulse.ingest.models import ReviewRecord
from patchpulse.pipeline.dbt import run_dbt

pytestmark = pytest.mark.integration

GAME = 553850
NIGHT = datetime(2026, 11, 2, 3, tzinfo=UTC)
TOO_FEW = "too few comparison games without their own patch (3 of 5 needed)"


def estimate(**changes: object) -> Estimate:
    values: dict[str, object] = {
        "bundle_id": "p1",
        "patch_ids": ("p1", "p2"),
        "appid": GAME,
        "outcome": "positive_share",
        "sample": Sample.ALL,
        "horizon": 14,
        "method": Method.ITS,
        "effect": -2.5,
        "lo90": -4.0,
        "hi90": -1.0,
        "p_value": None,
        "pre_fit_rmse": None,
        "donors": (),
        "flags": frozenset(),
        "reason": None,
        "fingerprint": "a" * 64,
        "version": "e2",
    }
    values.update(changes)
    return Estimate(**values)  # type: ignore[arg-type]


def config(tmp_path: Path, method: str) -> Path:
    path = tmp_path / "causal.yaml"
    write_causal_config(CausalConfig(method, "e2", "ab" * 32), path)
    return path


def test_estimates_follow_their_inputs_and_keep_history(
    migrated_engine: Engine, tmp_path: Path
) -> None:
    its = estimate()
    synth = estimate(
        method=Method.SYNTH, effect=-2.0, p_value=0.04, donors=((101, 0.6), (102, 0.4))
    )
    its_config = config(tmp_path, "its")
    with migrated_engine.begin() as connection:
        first = write_estimates(connection, [its, synth], run_id=1, computed_at=NIGHT)
        again = write_estimates(connection, [its, synth], run_id=2, computed_at=NIGHT)
        moved = replace(its, effect=-3.0, fingerprint="b" * 64)
        changed = write_estimates(connection, [moved, synth], run_id=3, computed_at=NIGHT)
        latest = stored_fingerprints(connection)
        before = estimate_patch_effect(connection, "p1", as_of=2, config_path=its_config)
        after = estimate_patch_effect(connection, "p1", config_path=its_config)
        twin = estimate_patch_effect(connection, "p1", method="synth", config_path=its_config)

    assert (first, again, changed) == (2, 0, 1)  # unchanged inputs write nothing
    assert latest == {its.key: "b" * 64, synth.key: "a" * 64}
    assert [(r.effect, r.run_id) for r in before] == [(-2.5, 1)]
    assert [(r.effect, r.run_id) for r in after] == [(-3.0, 3)]
    (stored,) = twin
    assert (stored.method, stored.donors) == ("synth", ((101, 0.6), (102, 0.4)))
    assert stored.p_value == pytest.approx(0.04)  # REAL: single precision
    assert (stored.patch_ids, stored.flags, stored.computed_at) == (
        ("p1", "p2"),
        frozenset(),
        NIGHT,
    )


def test_any_patch_of_a_bundle_finds_it(migrated_engine: Engine, tmp_path: Path) -> None:
    its_config = config(tmp_path, "its")
    with migrated_engine.begin() as connection:
        write_estimates(
            connection,
            [estimate(flags=frozenset({"confounded", "binned"}))],
            run_id=1,
            computed_at=NIGHT,
        )
        found = estimate_patch_effect(connection, "p2", config_path=its_config)
        missing = estimate_patch_effect(connection, "p9", config_path=its_config)

    assert [(r.bundle_id, r.flags) for r in found] == [("p1", frozenset({"binned", "confounded"}))]
    assert missing == []


def test_default_method_and_its_fallback(migrated_engine: Engine, tmp_path: Path) -> None:
    rows = [
        estimate(horizon=7, method=Method.SYNTH, effect=-1.0, fingerprint="c" * 64),
        estimate(horizon=7, effect=-0.5, fingerprint="d" * 64),
        estimate(method=Method.SYNTH, effect=None, lo90=None, hi90=None, reason=TOO_FEW),
        estimate(effect=-2.5),
    ]
    synth_config = config(tmp_path, "synth")
    with migrated_engine.begin() as connection:
        write_estimates(connection, rows, run_id=1, computed_at=NIGHT)
        served = estimate_patch_effect(connection, "p1", config_path=synth_config)
        asked = estimate_patch_effect(connection, "p1", method="synth", config_path=synth_config)

    # The default (synthetic control) where it has an estimate; interrupted time series where it
    # hasn't, with the default's reason kept so the reader can say why.
    assert [(r.horizon, r.method, r.effect, r.reason) for r in served] == [
        (7, "synth", -1.0, None),
        (14, "its", -2.5, TOO_FEW),
    ]
    assert [(r.horizon, r.method, r.effect) for r in asked] == [
        (7, "synth", -1.0),
        (14, "synth", None),
    ]


def review(rid: int, created: datetime, *, minutes: int | None, voted_up: bool) -> ReviewRecord:
    return ReviewRecord(
        recommendation_id=rid,
        appid=GAME,
        author_hash=f"{rid:064d}",
        author_num_games_owned=None,
        author_num_reviews=2,
        author_playtime_forever=minutes or 0,
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


def test_decompose_change_matches_the_library(
    new_database: Callable[[], Settings], tmp_path_factory: pytest.TempPathFactory
) -> None:
    admin_settings = new_database()
    admin = make_engine(admin_settings)
    upgrade_to_head(admin)
    base_day = datetime(2026, 10, 5, 12, tzinfo=UTC)
    window_day = base_day + timedelta(days=10)
    reviews = [
        review(1, base_day, minutes=30, voted_up=True),
        review(2, base_day, minutes=30, voted_up=True),
        review(3, base_day, minutes=700, voted_up=False),
        review(4, base_day + timedelta(days=1), minutes=700, voted_up=True),
        review(5, window_day, minutes=30, voted_up=False),
        review(6, window_day, minutes=30, voted_up=False),
        review(7, window_day, minutes=30, voted_up=True),
        review(8, window_day, minutes=None, voted_up=False),
        review(9, window_day + timedelta(days=3), minutes=700, voted_up=True),  # after the window
    ]
    game = Game(
        appid=GAME,
        name="Helldivers 2",
        genres=["Action"],
        role=Role.BOTH,
        backfill_start=date(2026, 9, 1),
    )
    with admin.begin() as connection:
        provision_writer(connection, password="Writer-Store-1")
        upsert_games(connection, [game], run_id=1)
        upsert_reviews(connection, reviews, run_id=1)
    writer = admin_settings.model_copy(
        update={"db_user": WRITER_USER, "db_password": SecretStr("Writer-Store-1")}
    )
    built = run_dbt(["build"], settings=writer, target_path=tmp_path_factory.mktemp("dbt"))
    assert built.ok, built.failures

    with admin.connect() as connection:
        result = decompose_change(
            connection,
            GAME,
            window=(date(2026, 10, 15), date(2026, 10, 16)),
            baseline_window=(date(2026, 10, 5), date(2026, 10, 6)),
        )

    expected = decompose(
        {"lt_2h": (2, 2), "10h_50h": (1, 2)},
        {"lt_2h": (1, 3), "unknown": (0, 1)},
    )
    assert result == expected
    assert (result.baseline_n, result.window_n) == (4, 4)
