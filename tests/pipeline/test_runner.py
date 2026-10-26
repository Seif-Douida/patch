"""The nightly stage runner (plan Task 8), on the SQL Server container with a fake Steam."""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
from agent_fixtures import SAMPLE_QUOTE, make_bundle, sample_report
from fake_steam import FakeSteam
from pydantic import SecretStr
from qdrant_client import QdrantClient
from review_index import create_index_tables, index_row, write_target, write_texts
from sqlalchemy import Connection, Engine, text
from tiny_encoder import write_tiny_embedder
from toy_reviews import toy_reviews

from patchpulse.agent.bundle import Bundle, load_bundle
from patchpulse.agent.graph import ReportRequest
from patchpulse.agent.report import Report
from patchpulse.agent.store import latest_reports, write_report
from patchpulse.config import Settings
from patchpulse.db.engine import make_engine
from patchpulse.db.migrate import upgrade_to_head
from patchpulse.export.site import AspectModel
from patchpulse.ingest.coverage import REREAD, start_of_day
from patchpulse.models.artifact import record_metrics
from patchpulse.models.s0 import save_s0, train_s0
from patchpulse.pipeline.dbt import DbtResult
from patchpulse.pipeline.effects import EffectsOutcome
from patchpulse.pipeline.indexing import IndexOutcome, run_index
from patchpulse.pipeline.reports import ReportsOutcome, run_reports
from patchpulse.pipeline.runner import (
    BASIC_REPORTS,
    NIGHT_RESERVE_S,
    NIGHT_SECONDS,
    PipelineError,
    run_nightly,
)
from patchpulse.retrieval.embedders import CANDIDATES
from patchpulse.retrieval.pin import pin_for, write_pin
from patchpulse.retrieval.qdrant import ensure_collection
from patchpulse.retrieval.reconcile import Reconciliation, RepairStats
from patchpulse.retrieval.sync import SyncStats

pytestmark = pytest.mark.integration

NOW = datetime.now(UTC).replace(microsecond=0)
GAME_A, GAME_B = 553850, 949230
BACKFILL_START = (NOW - timedelta(days=20)).date()


@dataclass
class RecordingPublisher:
    calls: list[tuple[dict[str, bytes], int]] = field(default_factory=list)
    branches: list[tuple[str, dict[str, bytes], int]] = field(default_factory=list)
    reports: list[tuple[int, str, str | None]] = field(default_factory=list)
    alerts: list[tuple[str, str]] = field(default_factory=list)
    resolved: list[tuple[str, str]] = field(default_factory=list)
    report_fails: bool = False
    branch_error: str | None = None

    def publish(self, files: Mapping[str, bytes], *, data_version: int) -> None:
        self.calls.append((dict(files), data_version))

    def publish_branch(self, branch: str, files: Mapping[str, bytes], *, data_version: int) -> None:
        if self.branch_error is not None:
            raise RuntimeError(self.branch_error)
        self.branches.append((branch, dict(files), data_version))

    def report(self, *, run_id: int, status: str, error: str | None) -> None:
        if self.report_fails:
            raise RuntimeError("GitHub is down")
        self.reports.append((run_id, status, error))

    def alert(self, *, title: str, body: str) -> None:
        self.alerts.append((title, body))

    def resolve(self, *, title: str, comment: str) -> None:
        self.resolved.append((title, comment))


def ok_dbt(args: Any, **_: Any) -> DbtResult:
    return DbtResult(0, {"success": 1})


def failing_dbt(args: Any, **_: Any) -> DbtResult:
    return DbtResult(1, {"error": 1}, ["model.patchpulse.fact_review: boom"])


def stub_export(
    connection: Connection,
    *,
    data_version: int,
    generated_at: datetime,
    aspect_model: AspectModel | None = None,
) -> dict[str, bytes]:
    return {"index.json": b"{}\n"}


@dataclass
class RecordingExport:
    models: list[AspectModel | None] = field(default_factory=list)

    def __call__(
        self,
        connection: Connection,
        *,
        data_version: int,
        generated_at: datetime,
        aspect_model: AspectModel | None = None,
    ) -> dict[str, bytes]:
        self.models.append(aspect_model)
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
    # The real stage on the real marts: the one patch is 2 days old, so no window is complete.
    assert stages["effects"] == {
        "estimated": 0,
        "written": 0,
        "unchanged": 0,
        "remaining": 0,
        "failed": 0,
    }


DONOR = 1245620
DONOR_2 = 2050650
YESTERDAY = NOW.date() - timedelta(days=1)


def add_donor(env: Env, appid: int = DONOR, name: str = "Elden Ring") -> None:
    with env.games_file.open("a", encoding="utf-8") as games:
        games.write(
            f"  - {{appid: {appid}, name: {name}, genres: [Action, RPG], role: donor, "
            f"backfill_start: {BACKFILL_START}}}\n"
        )


def review_query(request: Any) -> dict[str, Any] | None:
    """A GetAppReviews request's input_json, or None for any other request."""
    url = urlsplit(str(request.url))
    if url.path != "/IUserReviewsService/GetAppReviews/v1/":
        return None
    query: dict[str, Any] = json.loads(parse_qs(url.query)["input_json"][0])
    return query


def ago(days: int) -> date:
    return YESTERDAY - timedelta(days=days)


def noon(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 12, tzinfo=UTC)


def store_count(env: Env, appid: int, day: date) -> None:
    with env.engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO raw.steam_review_daily "
                "(appid, [day], positive, negative, fetched_run_id, fetched_at) "
                "VALUES (:appid, :day, 0, 0, 0, '2026-01-01')"
            ),
            {"appid": appid, "day": day},
        )


def count_days(env: Env, appid: int) -> list[tuple[date, int, int]]:
    return [
        (r.day, r.positive, r.negative)
        for r in env.rows(
            "SELECT [day], positive, negative FROM raw.steam_review_daily "
            f"WHERE appid = {appid} ORDER BY [day]"
        )
    ]


def test_a_donor_never_reaches_reviews_backfill_prices_or_the_site(env: Env) -> None:
    # Phase 4: a comparison game gets its news (patch dates) and nothing else of the review
    # pipeline; review text, prices and the site are for tracked games only.
    add_donor(env)
    env.steam.add_review(DONOR, 900, NOW - timedelta(days=1))
    env.steam.add_review(DONOR, 901, NOW - timedelta(days=20))
    env.steam.add_news(DONOR, "950", "Patch 1.16", NOW - timedelta(days=3))
    env.steam.prices[DONOR] = {
        "currency": "USD",
        "initial": 5999,
        "final": 5999,
        "discount_percent": 0,
    }

    result = env.run()

    assert result.status == "succeeded"
    urls = [str(request.url) for request in env.steam.requests]
    # Its reviews are only ever counted: one a page, read for the counts and dropped.
    asked = [q for q in map(review_query, env.steam.requests) if q and q["appid"] == DONOR]
    assert asked
    assert {q["num_per_page"] for q in asked} == {1}
    assert not [u for u in urls if "appids=" in u and str(DONOR) in u]
    assert [u for u in urls if "GetNewsForApp" in u and f"appid={DONOR}" in u]
    ((files, _),) = env.publisher.calls
    assert f"games/{DONOR}.json" not in files
    assert DONOR not in {g["appid"] for g in json.loads(files["index.json"])["games"]}
    assert DONOR not in [r.appid for r in env.rows("SELECT appid FROM ops.ingest_state")]
    assert [r.appid for r in env.rows("SELECT appid FROM raw.steam_review")].count(DONOR) == 0


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


def test_steam_rate_limit_during_backfill_stops_the_backfill_not_the_run(env: Env) -> None:
    # Backfill is best effort: a blocked night keeps its fresh data and resumes history tomorrow.
    env.steam.rate_limit_after = 4  # the incremental windows' 4 requests succeed; then HTTP 429

    result = env.run(dbt=ok_dbt, export=stub_export)

    assert result.status == "succeeded"
    assert result.stages["backfill"]["stopped"].startswith("SteamUnavailableError")
    coverage = {r.appid: r.covered_from for r in env.rows("SELECT * FROM ops.ingest_state")}
    assert coverage == {game: (NOW - REREAD).replace(tzinfo=None) for game in (GAME_A, GAME_B)}
    checks = {(r.check_name, r.severity) for r in env.rows("SELECT * FROM ops.data_quality_result")}
    assert ("backfill_stopped", "warn") in checks
    assert not any(severity == "error" for _, severity in checks)


def test_steam_refusing_prices_is_recorded_not_fatal(env: Env) -> None:
    env.steam.prices_down = True

    result = env.run(dbt=ok_dbt, export=stub_export)

    assert result.status == "succeeded"
    assert result.stages["prices"]["error"].startswith("SteamUnavailableError")
    checks = {(r.check_name, r.severity) for r in env.rows("SELECT * FROM ops.data_quality_result")}
    assert ("prices_unavailable", "warn") in checks


def test_publisher_hears_the_outcome_of_every_run(env: Env) -> None:
    # In Azure the outcome becomes a GitHub issue on failure, closed by the next success.
    ok = env.run(dbt=ok_dbt, export=stub_export)
    with pytest.raises(PipelineError):
        env.run(dbt=failing_dbt, export=stub_export)

    assert env.publisher.reports[0] == (ok.run_id, "succeeded", None)
    failed_run, status, error = env.publisher.reports[1]
    assert (failed_run, status) == (ok.run_id + 1, "failed")
    assert error is not None
    assert "dbt build failed" in error


def test_a_reporting_failure_never_hides_the_run_result(env: Env) -> None:
    env.publisher.report_fails = True

    assert env.run(dbt=ok_dbt, export=stub_export).status == "succeeded"
    with pytest.raises(PipelineError, match="dbt"):
        env.run(dbt=failing_dbt, export=stub_export)


def test_backfill_respects_the_time_budget(env: Env) -> None:
    no_budget = env.settings.model_copy(update={"backfill_budget_minutes": 0})

    result = env.run(settings=no_budget, dbt=ok_dbt, export=stub_export)

    assert result.status == "succeeded"
    (coverage,) = env.rows(f"SELECT covered_from FROM ops.ingest_state WHERE appid = {GAME_A}")
    assert coverage.covered_from == (NOW - REREAD).replace(tzinfo=None)
    loaded = env.rows("SELECT COUNT(*) FROM raw.steam_review")[0][0]
    assert loaded == 6  # the last 3 days only: 5 for A, 1 for B


class LateClock:
    """The run starts at 0; every later reading is `later` seconds into the night."""

    def __init__(self, later: float) -> None:
        self.later = later
        self.readings = 0

    def __call__(self) -> float:
        self.readings += 1
        return 0.0 if self.readings == 1 else self.later


NIGHT_END = NIGHT_SECONDS - NIGHT_RESERVE_S


def test_backfill_stops_at_the_night_deadline(env: Env) -> None:
    # 30 minutes of budget, but the night has used its time by the backfill (the clock jumps once
    # the prices, the stage before it, are in): export and publish keep theirs.
    def clock() -> float:
        priced = any("appdetails" in str(request.url) for request in env.steam.requests)
        return NIGHT_END + 1 if priced else 0.0

    result = env.run(dbt=ok_dbt, export=stub_export, clock=clock)

    assert result.status == "succeeded"
    assert result.stages["backfill"]["pages"] == 0
    assert env.rows("SELECT COUNT(*) FROM raw.steam_review")[0][0] == 6  # the last 3 days only


def test_an_early_empty_page_does_not_end_a_games_history(env: Env) -> None:
    # 20 Oct 2026: Steam answered empty pages long before the counts it reported for three games'
    # windows (282,709 reviews for Helldivers 2), and the backfill took them as the end of history.
    env.steam.page_size = 1
    env.steam.cut_short[GAME_A] = 1  # one review of the window's three, then empty pages
    floor = start_of_day(BACKFILL_START).replace(tzinfo=None)

    first = env.run(dbt=ok_dbt, export=stub_export)

    assert GAME_A not in first.stages["backfill"]["complete"]
    (coverage,) = env.rows(f"SELECT covered_from FROM ops.ingest_state WHERE appid = {GAME_A}")
    assert coverage.covered_from > floor  # resumes from its oldest review, not marked complete
    env.steam.cut_short.clear()

    second = env.run(dbt=ok_dbt, export=stub_export)

    assert GAME_A in second.stages["backfill"]["complete"]
    (coverage,) = env.rows(f"SELECT covered_from FROM ops.ingest_state WHERE appid = {GAME_A}")
    assert coverage.covered_from == floor
    assert env.rows("SELECT COUNT(*) FROM raw.steam_review")[0][0] == 10


def test_a_second_night_resumes_the_backfill_where_the_first_stopped(env: Env) -> None:
    no_budget = env.settings.model_copy(update={"backfill_budget_minutes": 0})
    env.run(settings=no_budget, dbt=ok_dbt, export=stub_export)

    env.run(dbt=ok_dbt, export=stub_export)

    assert env.rows("SELECT COUNT(*) FROM raw.steam_review")[0][0] == 10
    (coverage,) = env.rows(f"SELECT covered_from FROM ops.ingest_state WHERE appid = {GAME_A}")
    assert coverage.covered_from == start_of_day(BACKFILL_START).replace(tzinfo=None)


def toy_s0(directory: Path) -> Path:
    texts, labels = toy_reviews(200, seed=41)
    save_s0(train_s0(texts, labels), directory, thresholds=np.full(10, 0.5), version="20261013T0")
    return directory


def test_no_model_skips_scoring(env: Env) -> None:
    result = env.run(dbt=ok_dbt, export=stub_export)

    assert result.status == "succeeded"
    assert result.stages["score"] == {"skipped": "no model (PP_MODEL_DIR unset)"}


def test_broken_model_is_a_warning_and_the_night_succeeds(env: Env, tmp_path: Path) -> None:
    broken = toy_s0(tmp_path / "model")
    (broken / "s0.joblib").write_bytes(b"not the model")
    settings = env.settings.model_copy(update={"model_dir": broken})

    result = env.run(settings=settings, dbt=ok_dbt, export=stub_export)

    assert result.status == "succeeded"
    assert result.stages["score"]["skipped"].startswith("ArtifactError")
    (finding,) = env.rows(
        "SELECT severity, detail FROM ops.data_quality_result WHERE check_name = 'scoring_skipped'"
    )
    assert finding.severity == "warn"
    assert "checksum" in finding.detail


def test_the_night_scores_its_reviews_between_backfill_and_dbt(env: Env, tmp_path: Path) -> None:
    settings = env.settings.model_copy(update={"model_dir": toy_s0(tmp_path / "model")})

    result = env.run(settings=settings, dbt=ok_dbt, export=stub_export)

    assert result.stages["score"]["scored"] == 10
    assert result.stages["score"]["remaining"] == 0
    assert result.stages["score"]["model_version"] == "s0-20261013T0"
    assert env.rows("SELECT COUNT(*) FROM raw.review_aspect_score")[0][0] == 10
    order = list(result.stages["durations_s"])
    assert order.index("backfill") < order.index("score") < order.index("dbt")


def test_the_export_names_the_model_from_pp_model_dir(env: Env, tmp_path: Path) -> None:
    model = toy_s0(tmp_path / "model")
    record_metrics(model, {"test_macro_f1": 0.61})
    with_model, without = RecordingExport(), RecordingExport()

    env.run(
        settings=env.settings.model_copy(update={"model_dir": model}),
        dbt=ok_dbt,
        export=with_model,
    )
    env.run(dbt=ok_dbt, export=without)

    assert with_model.models == [AspectModel("s0", "20261013T0", 0.61)]
    assert without.models == [None]


# The index stage (Phase 3 plan Task 15, design D5).

QDRANT_TEST_URL = os.environ.get("PP_TEST_QDRANT_URL", "http://localhost:6333")
UNREACHABLE, SKIPPED, OUT_OF_SYNC = (
    "Qdrant unreachable",
    "index stage skipped",
    "Qdrant out of sync",
)
TINY = replace(CANDIDATES["e5-small"], name="tiny-index", dims=8, max_tokens=32)


def with_qdrant(settings: Settings, encoder_dir: Path, url: str = "https://qdrant.invalid") -> Any:
    return settings.model_copy(update={"qdrant_url": url, "encoder_dir": encoder_dir})


def outcome(*, difference: int = 0, target: int = 1000) -> IndexOutcome:
    return IndexOutcome(
        collection="reviews_v1",
        read_back=0,
        repaired=RepairStats(days=0, missing=0, extra=0),
        sync=SyncStats(upserted=12, updated=3, deleted=1, remaining=0),
        reconciliation=Reconciliation(
            target=target,
            qdrant=target - difference,
            difference=difference,
            days_off=1 if difference else 0,
        ),
        health_hits=5,
        seconds={"sync": 1.5},
    )


@dataclass
class FakeIndex:
    """The index stage's work, replaced: each call returns (or raises) the next outcome."""

    outcomes: list[IndexOutcome | Exception]
    calls: list[tuple[int, int]] = field(default_factory=list)
    deadlines: list[float] = field(default_factory=list)

    def __call__(
        self,
        engine: Engine,
        settings: Settings,
        *,
        run_id: int,
        health_appid: int,
        clock: Callable[[], float],
        deadline: float,
    ) -> IndexOutcome:
        self.calls.append((run_id, health_appid))
        self.deadlines.append(deadline)
        next_outcome = self.outcomes.pop(0)
        if isinstance(next_outcome, Exception):
            raise next_outcome
        return next_outcome


def test_index_stage_runs_between_dbt_and_export(env: Env, tmp_path: Path) -> None:
    index = FakeIndex([outcome()])

    result = env.run(
        settings=with_qdrant(env.settings, tmp_path), dbt=ok_dbt, export=stub_export, index=index
    )

    assert result.status == "succeeded"
    order = list(result.stages["durations_s"])
    assert order.index("dbt") < order.index("index") < order.index("export")
    assert index.calls == [(result.run_id, GAME_A)]  # the health query searches the first game
    record = result.stages["index"]
    assert (record["upserted"], record["updated"], record["deleted"]) == (12, 3, 1)
    assert record["mismatch"] == 0


@pytest.mark.parametrize(
    ("later", "deadline"),
    [(0.0, 5 * 60), (NIGHT_END - 60, NIGHT_END)],
    ids=["early: its own 5 minutes", "late: the night's end"],
)
def test_index_stage_gets_its_budget_within_the_night(
    env: Env, tmp_path: Path, later: float, deadline: float
) -> None:
    index = FakeIndex([outcome()])

    env.run(
        settings=with_qdrant(env.settings, tmp_path),
        dbt=ok_dbt,
        export=stub_export,
        index=index,
        clock=LateClock(later),
    )

    assert index.deadlines == [deadline]


@pytest.mark.parametrize(
    ("update", "reason"),
    [
        ({"encoder_dir": Path("encoder")}, "no Qdrant (PP_QDRANT_URL unset)"),
        ({"qdrant_url": "https://qdrant.invalid"}, "no encoder (PP_ENCODER_DIR unset)"),
        (
            {
                "qdrant_url": "https://qdrant.invalid",
                "encoder_dir": Path("e"),
                "index_budget_minutes": 0,
            },
            "turned off (PP_INDEX_BUDGET_MINUTES is 0)",
        ),
    ],
)
def test_index_is_skipped_without_qdrant_settings(
    env: Env, update: dict[str, Any], reason: str
) -> None:
    index = FakeIndex([])

    result = env.run(
        settings=env.settings.model_copy(update=update), dbt=ok_dbt, export=stub_export, index=index
    )

    assert result.status == "succeeded"
    assert result.stages["index"] == {"skipped": reason}
    assert index.calls == []
    assert (env.publisher.alerts, env.publisher.resolved) == ([], [])


@pytest.mark.parametrize(
    ("error", "title"),
    [
        (ConnectionError("Qdrant didn't answer at https://qdrant.invalid"), UNREACHABLE),
        (ValueError("collection reviews_v1 doesn't exist"), SKIPPED),
    ],
)
def test_qdrant_down_is_a_warning_and_the_night_publishes(
    env: Env, tmp_path: Path, error: Exception, title: str
) -> None:
    result = env.run(
        settings=with_qdrant(env.settings, tmp_path),
        dbt=ok_dbt,
        export=stub_export,
        index=FakeIndex([error]),
    )

    assert result.status == "succeeded"
    assert len(env.publisher.calls) == 1  # the site still goes out
    kind = type(error).__name__
    assert result.stages["index"]["skipped"].startswith(kind)
    (finding,) = env.rows(
        "SELECT severity, detail FROM ops.data_quality_result WHERE check_name = 'index_skipped'"
    )
    assert finding.severity == "warn"
    assert finding.detail.startswith(kind)
    assert "qdrant.invalid" not in finding.detail  # the cluster's URL is a secret
    ((alerted, body),) = env.publisher.alerts
    assert alerted == title
    assert f"run {result.run_id}" in body
    assert kind in body
    assert "qdrant.invalid" not in body
    assert env.publisher.resolved == []


def test_mismatch_over_half_a_percent_alerts_and_a_clean_night_resolves(
    env: Env, tmp_path: Path
) -> None:
    settings = with_qdrant(env.settings, tmp_path)
    index = FakeIndex([outcome(difference=6), outcome(difference=5)])  # 0.6%, then 0.5%

    first = env.run(settings=settings, dbt=ok_dbt, export=stub_export, index=index)

    assert first.stages["index"]["mismatch"] == pytest.approx(0.006)
    ((title, body),) = env.publisher.alerts
    assert title == OUT_OF_SYNC
    assert f"run {first.run_id}" in body
    (finding,) = env.rows(
        "SELECT severity FROM ops.data_quality_result WHERE check_name = 'qdrant_mismatch'"
    )
    assert finding.severity == "warn"
    assert env.publisher.resolved == []

    second = env.run(settings=settings, dbt=ok_dbt, export=stub_export, index=index)

    assert len(env.publisher.alerts) == 1
    assert {title for title, _ in env.publisher.resolved} == {OUT_OF_SYNC, SKIPPED, UNREACHABLE}
    assert all(f"run {second.run_id}" in comment for _, comment in env.publisher.resolved)


@pytest.mark.qdrant
def test_index_stage_end_to_end_on_both_containers(
    env: Env, qdrant: QdrantClient, new_collection: Callable[[], str], tmp_path: Path
) -> None:
    create_index_tables(env.engine)  # dbt's tables; the stubbed dbt doesn't build them
    created = NOW.replace(tzinfo=None) - timedelta(days=1)
    rows = [
        index_row(rid, created=created - timedelta(days=rid % 2), appid=GAME_A)
        for rid in range(1, 6)
    ]
    rows += [index_row(rid, created=created, appid=GAME_B) for rid in (20, 21)]
    write_target(env.engine, rows)
    write_texts(env.engine, {int(row["review_id"]): "good game" for row in rows})
    encoder = tmp_path / "encoder"
    write_tiny_embedder(encoder, TINY, dim=TINY.dims)
    collection = new_collection()
    ensure_collection(qdrant, collection, TINY.dims)
    pin = tmp_path / "retrieval.yaml"
    write_pin(pin_for(encoder, 1).model_copy(update={"collection": collection}), pin)
    settings = with_qdrant(env.settings, encoder, url=QDRANT_TEST_URL)
    index = functools.partial(run_index, pin_path=pin)

    first = env.run(settings=settings, dbt=ok_dbt, export=stub_export, index=index)

    assert first.status == "succeeded"
    record = first.stages["index"]
    assert (record["collection"], record["upserted"], record["remaining"]) == (collection, 7, 0)
    assert (record["mismatch"], record["health_hits"]) == (0, 5)
    assert qdrant.count(collection, exact=True).count == 7
    stored = env.rows(
        f"SELECT appid, [day], sql_count, qdrant_count FROM ops.qdrant_reconciliation "
        f"WHERE run_id = {first.run_id}"
    )
    assert sorted((r.appid, r.sql_count) for r in stored) == [(GAME_A, 2), (GAME_A, 3), (GAME_B, 2)]
    assert all(r.sql_count == r.qdrant_count for r in stored)
    assert {title for title, _ in env.publisher.resolved} == {OUT_OF_SYNC, SKIPPED, UNREACHABLE}

    second = env.run(settings=settings, dbt=ok_dbt, export=stub_export, index=index)

    assert (second.stages["index"]["upserted"], second.stages["index"]["mismatch"]) == (0, 0)
    assert env.publisher.alerts == []


def test_counts_stage_catches_up_and_stops_cleanly(env: Env) -> None:
    # Review focus 5: Steam stops answering mid-stage; the night still publishes, and the next
    # night fills the gap without duplicates.
    add_donor(env)
    store_count(env, DONOR, ago(6))
    env.steam.add_review(DONOR, 900, noon(ago(5)))
    env.steam.add_review(DONOR, 901, noon(ago(5)), voted_up=False)
    env.steam.add_review(DONOR, 902, noon(ago(0)))
    env.steam.rate_limit_after = 4 + 2  # the reviews stage's 4 requests, then 2 days of counts

    first = env.run(dbt=ok_dbt, export=stub_export)

    assert first.status == "succeeded"
    assert len(env.publisher.calls) == 1
    counts = first.stages["counts"]
    assert (counts["requests"], counts["days"], counts["behind"]) == (3, 2, 1)
    assert counts["stopped"].startswith("SteamUnavailableError")
    checks = {(r.check_name, r.severity) for r in env.rows("SELECT * FROM ops.data_quality_result")}
    assert ("counts_stopped", "warn") in checks

    env.steam.rate_limit_after = None
    second = env.run(dbt=ok_dbt, export=stub_export)
    third = env.run(dbt=ok_dbt, export=stub_export)

    assert second.stages["counts"] == {"requests": 4, "days": 4, "behind": 0, "stopped": None}
    assert third.stages["counts"] == {"requests": 0, "days": 0, "behind": 0, "stopped": None}
    assert count_days(env, DONOR) == [
        (ago(6), 0, 0),
        (ago(5), 1, 1),
        (ago(4), 0, 0),
        (ago(3), 0, 0),
        (ago(2), 0, 0),
        (ago(1), 0, 0),
        (ago(0), 1, 0),
    ]


def test_counts_start_after_the_seed_or_at_the_backfill_start(env: Env) -> None:
    add_donor(env)
    add_donor(env, DONOR_2, "Monster Hunter Wilds")
    with env.engine.begin() as connection:  # what `dbt seed` loads into core
        connection.execute(
            text(
                "CREATE TABLE core.donor_daily_counts "
                "(appid int, [day] date, positive int, negative int)"
            )
        )
        connection.execute(
            text("INSERT INTO core.donor_daily_counts VALUES (:appid, :day, 5, 1)"),
            {"appid": DONOR, "day": ago(2)},
        )

    result = env.run(dbt=ok_dbt, export=stub_export)

    assert [d for d, _, _ in count_days(env, DONOR)] == [ago(1), ago(0)]
    # No seed and nothing stored: from its backfill start, 4 days a night (20 to fetch).
    assert [d for d, _, _ in count_days(env, DONOR_2)] == [
        BACKFILL_START + timedelta(days=k) for k in range(4)
    ]
    assert result.stages["counts"]["behind"] == 1


def test_a_broken_count_fails_that_donor_only(env: Env) -> None:
    add_donor(env)
    add_donor(env, DONOR_2, "Monster Hunter Wilds")
    for appid in (DONOR, DONOR_2):
        store_count(env, appid, ago(2))
    env.steam.day_counts_broken[DONOR] = "no response"

    with pytest.raises(PipelineError, match=str(DONOR)):
        env.run(dbt=ok_dbt, export=stub_export)

    assert [d for d, _, _ in count_days(env, DONOR)] == [ago(2)]
    assert [d for d, _, _ in count_days(env, DONOR_2)] == [ago(2), ago(1), ago(0)]
    assert len(env.publisher.calls) == 1  # the night still ships


def test_counts_stage_stops_at_the_night_deadline(env: Env) -> None:
    add_donor(env)
    store_count(env, DONOR, ago(3))

    def clock() -> float:  # the night's time is used up once the prices (the stage before) are in
        priced = any("appdetails" in str(request.url) for request in env.steam.requests)
        return NIGHT_END + 1 if priced else 0.0

    result = env.run(dbt=ok_dbt, export=stub_export, clock=clock)

    assert result.status == "succeeded"
    assert result.stages["counts"] == {
        "requests": 0,
        "days": 0,
        "behind": 1,
        "stopped": "night deadline",
    }


@dataclass
class FakeEffects:
    """The effects stage's work, replaced: each call returns (or raises) the next outcome."""

    outcomes: list[EffectsOutcome | Exception]
    deadlines: list[float] = field(default_factory=list)

    def __call__(
        self,
        engine: Engine,
        *,
        run_id: int,
        now: datetime,
        clock: Callable[[], float],
        deadline: float,
    ) -> EffectsOutcome:
        self.deadlines.append(deadline)
        next_outcome = self.outcomes.pop(0)
        if isinstance(next_outcome, Exception):
            raise next_outcome
        return next_outcome


def test_effects_failure_is_a_warning_and_the_night_publishes(env: Env) -> None:
    # Review focus 4: the estimates lag a night; the site still goes out.
    effects = FakeEffects([RuntimeError("Invalid object name 'mart.daily_game_sentiment'")])

    result = env.run(dbt=ok_dbt, export=stub_export, effects=effects)

    assert result.status == "succeeded"
    assert len(env.publisher.calls) == 1
    order = list(result.stages["durations_s"])
    assert order.index("dbt") < order.index("effects") < order.index("index")
    assert result.stages["effects"]["skipped"].startswith("RuntimeError: Invalid object name")
    (finding,) = env.rows(
        "SELECT severity, detail FROM ops.data_quality_result WHERE check_name = 'effects_skipped'"
    )
    assert finding.severity == "warn"
    assert finding.detail.startswith("RuntimeError")


@pytest.mark.parametrize(
    ("later", "deadline"),
    [(0.0, 2 * 60), (NIGHT_END - 60, NIGHT_END)],
    ids=["early: its own 2 minutes", "late: the night's end"],
)
def test_effects_stage_gets_its_budget_within_the_night(
    env: Env, later: float, deadline: float
) -> None:
    effects = FakeEffects(
        [EffectsOutcome(estimated=3, written=45, unchanged=1, remaining=0, failed=0)]
    )

    result = env.run(dbt=ok_dbt, export=stub_export, effects=effects, clock=LateClock(later))

    assert effects.deadlines == [deadline]
    assert result.stages["effects"] == {
        "estimated": 3,
        "written": 45,
        "unchanged": 1,
        "remaining": 0,
        "failed": 0,
    }


def test_effects_stage_is_off_at_a_budget_of_0(env: Env) -> None:
    effects = FakeEffects([])

    result = env.run(
        settings=env.settings.model_copy(update={"effects_budget_minutes": 0}),
        dbt=ok_dbt,
        export=stub_export,
        effects=effects,
    )

    assert result.status == "succeeded"
    assert result.stages["effects"] == {"skipped": "turned off (PP_EFFECTS_BUDGET_MINUTES is 0)"}
    assert effects.deadlines == []


# Phase 5 (plan Task 12, design D10): the agent's data bundle and the nightly reports.


def fixture_bundle(connection: Connection, *, run_id: int, today: date) -> Bundle:
    """The agent tests' bundle, as if built from tonight's marts (the stubbed dbt builds none)."""
    return replace(make_bundle(), data_version=run_id, built_on=today)


def failing_bundle(connection: Connection, *, run_id: int, today: date) -> Bundle:
    raise RuntimeError("Invalid object name 'mart.daily_aspect_share'")


def with_gemini(settings: Settings, **changes: Any) -> Settings:
    return settings.model_copy(update={"gemini_api_key": SecretStr("test-gemini-key"), **changes})


def tonight(**counts: int) -> ReportsOutcome:
    values = {"due": 0, "written": 0, "basic": 0, "failed": 0, "remaining": 0}
    values.update(counts)
    return ReportsOutcome(**values)


@dataclass
class FakeReports:
    """The reports stage's work, replaced: each call returns (or raises) the next outcome, after
    storing that night's reports from `stores`, if any."""

    outcomes: list[ReportsOutcome | Exception]
    stores: list[list[Report]] = field(default_factory=list)
    bundles: list[Bundle] = field(default_factory=list)
    deadlines: list[float] = field(default_factory=list)

    def __call__(
        self,
        engine: Engine,
        settings: Settings,
        bundle: Bundle,
        *,
        run_id: int,
        steam: Any,
        clock: Callable[[], float],
        deadline: float,
    ) -> ReportsOutcome:
        self.bundles.append(bundle)
        self.deadlines.append(deadline)
        for report in self.stores.pop(0) if self.stores else []:
            with engine.begin() as connection:
                write_report(connection, report, fingerprint="0" * 64, run_id=run_id)
        next_outcome = self.outcomes.pop(0)
        if isinstance(next_outcome, Exception):
            raise next_outcome
        return next_outcome


def test_bundle_and_reports_run_after_effects(env: Env) -> None:
    effects = FakeEffects(
        [EffectsOutcome(estimated=1, written=8, unchanged=0, remaining=0, failed=0)]
    )
    reports = FakeReports([tonight(due=2, written=2)])

    result = env.run(
        settings=with_gemini(env.settings),
        dbt=ok_dbt,
        export=stub_export,
        effects=effects,
        bundle=fixture_bundle,
        reports=reports,
    )

    assert result.status == "succeeded"
    order = list(result.stages["durations_s"])
    stages = ["effects", "index", "bundle", "reports", "agent_data", "export", "publish"]
    assert [order.index(name) for name in stages] == sorted(order.index(name) for name in stages)
    (bundle,) = reports.bundles
    assert bundle.data_version == result.run_id
    built = result.stages["bundle"]
    assert (built["games"], built["patches"], built["estimates"]) == (2, 4, 8)
    assert result.stages["reports"] == {
        "due": 2,
        "written": 2,
        "basic": 0,
        "failed": 0,
        "remaining": 0,
    }
    # The API's data: tonight's bundle and the reports with the patch-to-bundle map.
    ((branch, files, version),) = env.publisher.branches
    assert (branch, version) == ("agent-data", result.run_id)
    assert built["bytes"] == len(files["bundle.json.gz"])
    assert load_bundle(files["bundle.json.gz"]) == bundle
    assert json.loads(files["reports.json"]) == {
        "data_version": result.run_id,
        "reports": {},
        "patches": {"p0": "p0", "p1": "p1", "p2": "p1"},
    }
    assert len(env.publisher.calls) == 1


def test_a_failing_report_is_a_warning_and_the_night_publishes(env: Env) -> None:
    def agent(request: ReportRequest) -> Report:
        if request.horizon == 14:
            raise RuntimeError("the draft step failed")
        return sample_report(str(request.bundle_id), int(request.horizon or 0))

    result = env.run(
        settings=with_gemini(env.settings),
        dbt=ok_dbt,
        export=stub_export,
        bundle=fixture_bundle,
        reports=functools.partial(run_reports, agent=agent),
    )

    assert result.status == "succeeded"
    assert len(env.publisher.calls) == 1
    assert result.stages["reports"] == {
        "due": 2,
        "written": 1,
        "basic": 0,
        "failed": 1,
        "remaining": 0,
    }
    (finding,) = env.rows(
        "SELECT appid, severity, detail FROM ops.data_quality_result "
        "WHERE check_name = 'report_failed'"
    )
    assert (finding.appid, finding.severity) == (553850, "warn")
    assert finding.detail == "p1-14d: RuntimeError: the draft step failed"
    with env.engine.connect() as connection:
        assert set(latest_reports(connection)) == {("p1", 7)}
    ((_, files, _),) = env.publisher.branches
    shown = json.loads(files["reports.json"])["reports"]
    assert list(shown) == ["p1"]
    assert shown["p1"]["horizon"] == 7
    assert SAMPLE_QUOTE not in files["reports.json"].decode("utf-8")


def test_basic_reports_raise_and_resolve_the_alert(env: Env) -> None:
    # Design D8: a basic report is written without a model. The alert stays open while the site
    # shows one: a night whose budget doesn't reach it leaves the alert open.
    reports = FakeReports(
        [
            tonight(due=3, written=3, basic=2),
            tonight(due=2, written=1, remaining=1),
            tonight(due=1, written=1),
        ],
        stores=[
            [sample_report("p0", 7, basic=True), sample_report("p1", 7, basic=True)],
            [sample_report("p0", 7)],  # p1's report waits for the next night
            [sample_report("p1", 7)],
        ],
    )
    settings = with_gemini(env.settings)

    def night() -> Any:
        return env.run(
            settings=settings,
            dbt=ok_dbt,
            export=stub_export,
            bundle=fixture_bundle,
            reports=reports,
        )

    first = night()

    ((title, body),) = env.publisher.alerts
    assert title == BASIC_REPORTS
    assert f"run {first.run_id}" in body
    assert "2 of the 3 reports" in body
    (finding,) = env.rows(
        "SELECT severity FROM ops.data_quality_result WHERE check_name = 'report_basic'"
    )
    assert finding.severity == "warn"

    night()
    assert env.publisher.resolved == []

    third = night()
    ((resolved, comment),) = env.publisher.resolved
    assert resolved == BASIC_REPORTS
    assert f"run {third.run_id}" in comment
    assert len(env.publisher.alerts) == 1


@pytest.mark.parametrize(
    ("later", "deadline"),
    [(0.0, 3 * 60), (NIGHT_END - 60, NIGHT_END)],
    ids=["early: its own 3 minutes", "late: the night's end"],
)
def test_reports_stage_gets_its_budget_within_the_night(
    env: Env, later: float, deadline: float
) -> None:
    reports = FakeReports([tonight()])

    env.run(
        settings=with_gemini(env.settings),
        dbt=ok_dbt,
        export=stub_export,
        bundle=fixture_bundle,
        reports=reports,
        clock=LateClock(later),
    )

    assert reports.deadlines == [deadline]


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"gemini_api_key": None}, "no Gemini key (PP_GEMINI_API_KEY unset)"),
        ({"reports_budget_minutes": 0}, "turned off (PP_REPORTS_BUDGET_MINUTES is 0)"),
    ],
    ids=["no key", "budget 0"],
)
def test_reports_are_skipped_without_a_key_or_a_budget(
    env: Env, changes: dict[str, Any], reason: str
) -> None:
    reports = FakeReports([])

    result = env.run(
        settings=with_gemini(env.settings, **changes),
        dbt=ok_dbt,
        export=stub_export,
        bundle=fixture_bundle,
        reports=reports,
    )

    assert result.status == "succeeded"
    assert result.stages["reports"] == {"skipped": reason}
    assert reports.bundles == []
    ((branch, _, _),) = env.publisher.branches  # the API still gets tonight's bundle
    assert branch == "agent-data"
    assert (env.publisher.alerts, env.publisher.resolved) == ([], [])


def test_a_failing_bundle_is_a_warning_and_the_night_publishes(env: Env) -> None:
    reports = FakeReports([])

    result = env.run(
        settings=with_gemini(env.settings),
        dbt=ok_dbt,
        export=stub_export,
        bundle=failing_bundle,
        reports=reports,
    )

    assert result.status == "succeeded"
    assert result.stages["bundle"]["skipped"].startswith("RuntimeError: Invalid object name")
    assert result.stages["reports"] == {"skipped": "no bundle tonight"}
    assert reports.bundles == []
    assert env.publisher.branches == []  # the API keeps yesterday's data
    assert len(env.publisher.calls) == 1
    (finding,) = env.rows(
        "SELECT severity FROM ops.data_quality_result WHERE check_name = 'bundle_skipped'"
    )
    assert finding.severity == "warn"


def test_a_failing_reports_stage_is_a_warning_without_secrets(env: Env) -> None:
    settings = with_gemini(env.settings, qdrant_url="https://qdrant.invalid:6333")
    reports = FakeReports(
        [RuntimeError("search failed at https://qdrant.invalid:6333 with key test-gemini-key")]
    )

    result = env.run(
        settings=settings, dbt=ok_dbt, export=stub_export, bundle=fixture_bundle, reports=reports
    )

    assert result.status == "succeeded"
    skipped = result.stages["reports"]["skipped"]
    assert skipped.startswith("RuntimeError: search failed")
    assert "qdrant.invalid" not in skipped
    assert "test-gemini-key" not in skipped
    (finding,) = env.rows(
        "SELECT severity, detail FROM ops.data_quality_result WHERE check_name = 'reports_skipped'"
    )
    assert finding.severity == "warn"
    assert "qdrant.invalid" not in finding.detail
    assert len(env.publisher.branches) == 1  # the bundle and the stored reports still go out
    assert (env.publisher.alerts, env.publisher.resolved) == ([], [])


SECRET_DETAIL = "failed at https://qdrant.invalid:6333 with key test-gemini-key"


def leaky_bundle(connection: Connection, *, run_id: int, today: date) -> Bundle:
    raise RuntimeError(f"bundle read {SECRET_DETAIL}")


def test_bundle_and_agent_data_failures_keep_secrets_out(env: Env) -> None:
    # The review's minor 8: a stage's error can carry the cluster's URL or a key.
    settings = with_gemini(env.settings, qdrant_url="https://qdrant.invalid:6333")
    first = env.run(settings=settings, dbt=ok_dbt, export=stub_export, bundle=leaky_bundle,
                    reports=FakeReports([]))  # fmt: skip
    env.publisher.branch_error = f"push {SECRET_DETAIL}"
    second = env.run(settings=settings, dbt=ok_dbt, export=stub_export, bundle=fixture_bundle,
                     reports=FakeReports([]))  # fmt: skip

    details = [first.stages["bundle"]["skipped"], second.stages["agent_data"]["skipped"]]
    details += [r.detail for r in env.rows(
        "SELECT detail FROM ops.data_quality_result "
        "WHERE check_name IN ('bundle_skipped', 'agent_data_skipped')"
    )]  # fmt: skip
    assert len(details) == 4
    assert all("qdrant.invalid" not in d and "test-gemini-key" not in d for d in details)
