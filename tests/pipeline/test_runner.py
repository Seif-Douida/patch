"""The nightly stage runner (plan Task 8), on the SQL Server container with a fake Steam."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fake_steam import FakeSteam
from sqlalchemy import Connection, Engine, text

from patchpulse.config import Settings
from patchpulse.db.engine import make_engine
from patchpulse.db.migrate import upgrade_to_head
from patchpulse.ingest.coverage import REREAD, start_of_day
from patchpulse.pipeline.dbt import DbtResult
from patchpulse.pipeline.runner import PipelineError, run_nightly

pytestmark = pytest.mark.integration

NOW = datetime.now(UTC).replace(microsecond=0)
GAME_A, GAME_B = 553850, 949230
BACKFILL_START = (NOW - timedelta(days=20)).date()


@dataclass
class RecordingPublisher:
    calls: list[tuple[dict[str, bytes], int]] = field(default_factory=list)

    def publish(self, files: Mapping[str, bytes], *, data_version: int) -> None:
        self.calls.append((dict(files), data_version))


def ok_dbt(args: Any, **_: Any) -> DbtResult:
    return DbtResult(0, {"success": 1})


def failing_dbt(args: Any, **_: Any) -> DbtResult:
    return DbtResult(1, {"error": 1}, ["model.patchpulse.fact_review: boom"])


def stub_export(
    connection: Connection, *, data_version: int, generated_at: datetime
) -> dict[str, bytes]:
    return {"index.json": b"{}\n"}


@dataclass
class Env:
    settings: Settings
    engine: Engine
    games_file: Path
    steam: FakeSteam
    target_root: Path
    publisher: RecordingPublisher = field(default_factory=RecordingPublisher)

    def run(self, **overrides: Any) -> Any:
        arguments: dict[str, Any] = {
            "engine": self.engine,
            "steam": self.steam.http(),
            "games_file": self.games_file,
            "now": NOW,
            "publisher": self.publisher,
            "target_root": self.target_root,
        }
        arguments.update(overrides)
        settings = arguments.pop("settings", self.settings)
        return run_nightly(settings, **arguments)

    def rows(self, sql: str) -> list[Any]:
        with self.engine.connect() as connection:
            return list(connection.execute(text(sql)))


@pytest.fixture
def env(new_database: Callable[[], Settings], tmp_path: Path) -> Env:
    settings = new_database().model_copy(
        update={"author_hash_salt": "test-salt", "backfill_budget_minutes": 30}
    )
    settings = Settings.model_validate(settings.model_dump())  # re-validate the salt as a secret
    engine = make_engine(settings)
    upgrade_to_head(engine)
    games_file = tmp_path / "games.yaml"
    games_file.write_text(
        "games:\n"
        f"  - {{appid: {GAME_A}, name: Helldivers 2, genres: [Action], role: tracked, "
        f"backfill_start: {BACKFILL_START}}}\n"
        f"  - {{appid: {GAME_B}, name: 'Cities: Skylines II', genres: [Simulation], "
        f"role: tracked, backfill_start: {BACKFILL_START}}}\n",
        encoding="utf-8",
    )
    steam = FakeSteam()
    for rid in range(1, 6):
        steam.add_review(GAME_A, rid, NOW - timedelta(days=1, hours=rid))
    for rid in range(6, 9):
        steam.add_review(GAME_A, rid, NOW - timedelta(days=10, hours=rid), voted_up=False)
    steam.add_review(GAME_B, 20, NOW - timedelta(days=1))
    steam.add_review(GAME_B, 21, NOW - timedelta(days=15))
    steam.add_news(GAME_A, "900", "Patch 1.2.3", NOW - timedelta(days=2))
    steam.prices[GAME_A] = {
        "currency": "USD",
        "initial": 3999,
        "final": 3999,
        "discount_percent": 0,
    }
    return Env(settings, engine, games_file, steam, tmp_path / "dbt-target")


def test_nightly_end_to_end_with_mocked_steam(env: Env) -> None:
    result = env.run()

    assert result.status == "succeeded"
    ((files, version),) = env.publisher.calls
    assert version == result.run_id
    assert json.loads(files["index.json"])["data_version"] == result.run_id
    assert len(json.loads(files[f"games/{GAME_A}.json"])["series"]) >= 10
    assert env.rows("SELECT COUNT(*) FROM core.fact_review")[0][0] == 10
    assert env.rows("SELECT COUNT(*) FROM core.fact_patch")[0][0] == 1
    coverage = {
        r.appid: (r.covered_from, r.covered_to) for r in env.rows("SELECT * FROM ops.ingest_state")
    }
    floor = start_of_day(BACKFILL_START).replace(tzinfo=None)
    assert coverage == {
        GAME_A: (floor, NOW.replace(tzinfo=None)),
        GAME_B: (floor, NOW.replace(tzinfo=None)),
    }
    (run,) = env.rows("SELECT status, stages FROM ops.pipeline_run")
    assert run.status == "succeeded"
    stages = json.loads(run.stages)
    assert {"reviews", "news", "prices", "backfill", "dbt", "export", "durations_s"} <= set(stages)
    assert stages["reviews"]["new_versions"] + stages["backfill"]["new_versions"] == 10


def test_watermark_moves_only_after_a_successful_run(env: Env) -> None:
    with pytest.raises(PipelineError, match="dbt"):
        env.run(dbt=failing_dbt, export=stub_export)

    assert env.rows("SELECT COUNT(*) FROM ops.ingest_state")[0][0] == 0
    assert env.rows("SELECT COUNT(*) FROM raw.steam_review")[0][0] == 10  # the data is kept
    (run,) = env.rows("SELECT status, error FROM ops.pipeline_run")
    assert run.status == "failed"
    assert "fact_review" in run.error
    assert env.publisher.calls == []


def test_one_game_failing_does_not_stop_the_others_but_fails_the_run(env: Env) -> None:
    env.steam.failing.add(GAME_B)

    with pytest.raises(PipelineError, match=str(GAME_B)):
        env.run(dbt=ok_dbt, export=stub_export)

    assert [r.appid for r in env.rows("SELECT appid FROM ops.ingest_state")] == [GAME_A]
    assert len(env.publisher.calls) == 1  # the healthy game's data still ships
    (run,) = env.rows("SELECT status FROM ops.pipeline_run")
    assert run.status == "failed"


def test_backfill_respects_the_time_budget(env: Env) -> None:
    no_budget = env.settings.model_copy(update={"backfill_budget_minutes": 0})

    result = env.run(settings=no_budget, dbt=ok_dbt, export=stub_export)

    assert result.status == "succeeded"
    (coverage,) = env.rows(f"SELECT covered_from FROM ops.ingest_state WHERE appid = {GAME_A}")
    assert coverage.covered_from == (NOW - REREAD).replace(tzinfo=None)
    loaded = env.rows("SELECT COUNT(*) FROM raw.steam_review")[0][0]
    assert loaded == 6  # the last 3 days only: 5 for A, 1 for B


def test_a_second_night_resumes_the_backfill_where_the_first_stopped(env: Env) -> None:
    no_budget = env.settings.model_copy(update={"backfill_budget_minutes": 0})
    env.run(settings=no_budget, dbt=ok_dbt, export=stub_export)

    env.run(dbt=ok_dbt, export=stub_export)

    assert env.rows("SELECT COUNT(*) FROM raw.steam_review")[0][0] == 10
    (coverage,) = env.rows(f"SELECT covered_from FROM ops.ingest_state WHERE appid = {GAME_A}")
    assert coverage.covered_from == start_of_day(BACKFILL_START).replace(tzinfo=None)
