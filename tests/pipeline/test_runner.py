"""The nightly stage runner (plan Task 8), on the SQL Server container with a fake Steam."""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fake_steam import FakeSteam
from qdrant_client import QdrantClient
from review_index import create_index_tables, index_row, write_target, write_texts
from sqlalchemy import Connection, Engine, text
from tiny_encoder import write_tiny_embedder
from toy_reviews import toy_reviews

from patchpulse.config import Settings
from patchpulse.db.engine import make_engine
from patchpulse.db.migrate import upgrade_to_head
from patchpulse.export.site import AspectModel
from patchpulse.ingest.coverage import REREAD, start_of_day
from patchpulse.models.artifact import record_metrics
from patchpulse.models.s0 import save_s0, train_s0
from patchpulse.pipeline.dbt import DbtResult
from patchpulse.pipeline.indexing import IndexOutcome, run_index
from patchpulse.pipeline.runner import PipelineError, run_nightly
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
    reports: list[tuple[int, str, str | None]] = field(default_factory=list)
    alerts: list[tuple[str, str]] = field(default_factory=list)
    resolved: list[tuple[str, str]] = field(default_factory=list)
    report_fails: bool = False

    def publish(self, files: Mapping[str, bytes], *, data_version: int) -> None:
        self.calls.append((dict(files), data_version))

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

    def __call__(
        self,
        engine: Engine,
        settings: Settings,
        *,
        run_id: int,
        health_appid: int,
        clock: Callable[[], float],
    ) -> IndexOutcome:
        self.calls.append((run_id, health_appid))
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
