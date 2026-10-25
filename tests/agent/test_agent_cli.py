"""`patchpulse-agent backfill-reports` (plan Task 14, design D10): a report for every past patch
bundle, written once, on the SQL Server container with scripted models."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from agent_fixtures import ESTIMATES, make_bundle, sample_report
from fake_clock import FakeClock
from sqlalchemy import Connection, Engine, text

from patchpulse.agent import cli
from patchpulse.agent.bundle import Bundle
from patchpulse.agent.cli import BackfillOutcome, backfill_reports
from patchpulse.agent.graph import AgentDeps
from patchpulse.agent.llm_cache import ReplayCache
from patchpulse.agent.prompts import PROMPT_NAMES, load_prompt
from patchpulse.agent.router import ModelRouter
from patchpulse.agent.store import latest_reports, write_report
from patchpulse.labeling.limits import QuotaExhaustedError
from patchpulse.labeling.llm import LLMResponse

pytestmark = pytest.mark.integration

CHAIN = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
TODAY = date(2026, 10, 11)
ANSWERS: dict[str, dict[str, Any]] = {
    "gather": {"calls": []},
    "draft": {"findings": [], "claims": []},
    "summary": {"summary": "Nothing stood out."},
}


class Scripted:
    """A model with a day's budget of answers; past it, the daily quota is used up."""

    def __init__(self, budget: int = 1_000) -> None:
        self.budget = budget
        self.calls = 0
        self.by_prompt = {load_prompt(name).text: name for name in PROMPT_NAMES}

    def generate(self, system: str, user: str, *, json_schema: Any, **_: Any) -> LLMResponse:
        self.calls += 1
        if self.calls > self.budget:
            raise QuotaExhaustedError("the day's requests are used up")
        return LLMResponse(json.dumps(ANSWERS[self.by_prompt[system]]), 900, 120, "STOP")


def two_bundles(connection: Connection, *, run_id: int, today: date) -> Bundle:
    """The agent tests' bundle with estimates for p0 too: four reports to write."""
    bundle = make_bundle()
    game, *others = bundle.games
    older = tuple(replace(e, bundle_id="p0", patch_ids=("p0",)) for e in ESTIMATES)
    return replace(
        bundle,
        data_version=run_id,
        built_on=today,
        games=(replace(game, estimates=ESTIMATES + older), *others),
    )


def backfill(engine: Engine, model: Scripted, lines: list[str]) -> BackfillOutcome:
    clock = FakeClock(datetime(2026, 10, 11, 10, 0, tzinfo=UTC))

    def deps(bundle: Bundle) -> AgentDeps:
        cache = ReplayCache(Path("unused.sqlite"), "live")
        router = ModelRouter(CHAIN, {CHAIN[0]: model, CHAIN[1]: model}, cache, clock=clock)
        return AgentDeps(bundle=bundle, router=router)

    return backfill_reports(engine, deps, today=TODAY, build=two_bundles, out=lines.append, every=2)


def stored(engine: Engine) -> dict[tuple[str, int], Any]:
    with engine.connect() as connection:
        return dict(latest_reports(connection))


def test_backfill_writes_every_missing_report_once(migrated_engine: Engine) -> None:
    with migrated_engine.begin() as connection:  # the nightly job wrote this one already
        write_report(connection, sample_report("p1", 7), fingerprint="0" * 64, run_id=40)
    model = Scripted()
    lines: list[str] = []

    outcome = backfill(migrated_engine, model, lines)

    assert outcome == BackfillOutcome(missing=3, written=3, skipped=0, stopped=None)
    reports = stored(migrated_engine)
    assert set(reports) == {("p1", 7), ("p1", 14), ("p0", 7), ("p0", 14)}
    assert reports[("p1", 7)].run_id == 40  # never redone
    assert not any(r.basic for r in reports.values())
    assert model.calls == 9  # gather, draft and summary for each
    assert lines[:-1] == ["2 of 3 reports done"]
    assert lines[-1] == "3 of 3 missing reports written."
    (run,) = migrated_engine.connect().execute(
        text("SELECT kind, status, stages FROM ops.pipeline_run")
    )
    assert (run.kind, run.status) == ("reports_backfill", "succeeded")
    assert json.loads(run.stages)["written"] == 3

    again = backfill(migrated_engine, model, [])

    assert again == BackfillOutcome(missing=0, written=0, skipped=0, stopped=None)
    assert model.calls == 9


def test_backfill_stops_at_the_quota_and_resumes(migrated_engine: Engine) -> None:
    lines: list[str] = []

    # One report's three calls, then the next report's gather: its draft finds the quota spent.
    first = backfill(migrated_engine, Scripted(budget=4), lines)

    assert first == BackfillOutcome(
        missing=4, written=1, skipped=0, stopped="the day's model quota is used up"
    )
    assert "1 of 4 missing reports written" in lines[-1]
    assert "run the command again tomorrow" in lines[-1]
    reports = stored(migrated_engine)
    assert list(reports) == [("p1", 14)]  # the newest; the cut-off report isn't stored as basic
    assert not reports[("p1", 14)].basic

    second = backfill(migrated_engine, Scripted(), [])

    assert second == BackfillOutcome(missing=3, written=3, skipped=0, stopped=None)
    assert len(stored(migrated_engine)) == 4


def test_backfill_refuses_without_yes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_engine(*args: object, **kwargs: object) -> Callable[..., object]:
        raise AssertionError("refusing must not touch the database")

    monkeypatch.setattr(cli, "make_entra_engine", no_engine)

    assert cli.main(["backfill-reports"]) == 2
    error = capsys.readouterr().err
    assert "wakes Azure SQL" in error
    assert "--yes" in error
