"""`patchpulse-agent backfill-reports` (plan Task 14, design D10): a report for every past patch
bundle, written once, on the SQL Server container with scripted models."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from agent_fixtures import ESTIMATES, make_bundle, sample_report
from fake_clock import FakeClock
from sqlalchemy import Connection, Engine, text

from patchpulse.agent import cli
from patchpulse.agent.bundle import Bundle
from patchpulse.agent.cli import OUTAGE, QUOTA_SPENT, BackfillOutcome, backfill_reports
from patchpulse.agent.graph import AgentDeps
from patchpulse.agent.llm_cache import ReplayCache
from patchpulse.agent.prompts import PROMPT_NAMES, load_prompt
from patchpulse.agent.router import ModelRouter
from patchpulse.agent.store import (
    latest_reports,
    report_fingerprint,
    report_inputs,
    write_report,
)
from patchpulse.labeling.limits import QuotaExhaustedError
from patchpulse.labeling.llm import LLMResponse, TeacherUnavailableError

pytestmark = pytest.mark.integration

CHAIN = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
TODAY = date(2026, 10, 11)
ANSWERS: dict[str, dict[str, Any]] = {
    "gather": {"calls": []},
    "draft": {"findings": [], "claims": []},
    "summary": {"summary": "Nothing stood out."},
}


class Scripted:
    """A model with a day's budget of answers; past it, the daily quota is used up. Its first
    `fail_first` calls meet Gemini's "high demand" outage."""

    def __init__(self, budget: int = 1_000, *, fail_first: int = 0, garbled: bool = False) -> None:
        self.budget = budget
        self.fail_first = fail_first
        self.garbled = garbled  # answers, but never in JSON
        self.calls = 0
        self.by_prompt = {load_prompt(name).text: name for name in PROMPT_NAMES}

    def generate(self, system: str, user: str, *, json_schema: Any, **_: Any) -> LLMResponse:
        self.calls += 1
        if self.calls <= self.fail_first:
            raise TeacherUnavailableError(503, "This model is currently experiencing high demand.")
        if self.calls > self.budget:
            raise QuotaExhaustedError("the day's requests are used up")
        if self.garbled:
            return LLMResponse("Sure! Here is the plan.", 900, 120, "STOP")
        return LLMResponse(json.dumps(ANSWERS[self.by_prompt[system]]), 900, 120, "STOP")


def four_reports(run_id: int = 1, today: date = TODAY) -> Bundle:
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


def two_bundles(connection: Connection, *, run_id: int, today: date) -> Bundle:
    return four_reports(run_id, today)


def fingerprint(bundle_id: str, horizon: int) -> str:
    """A stored report's fingerprint on the test bundle's inputs, as the nightly computes it."""
    return report_fingerprint(report_inputs(four_reports(), bundle_id, horizon))


START = datetime(2026, 10, 11, 10, 0, tzinfo=UTC)


def backfill(
    engine: Engine, model: Scripted, lines: list[str], clock: FakeClock | None = None
) -> BackfillOutcome:
    clock = clock or FakeClock(START)

    def deps(bundle: Bundle) -> AgentDeps:
        cache = ReplayCache(Path("unused.sqlite"), "live")
        router = ModelRouter(CHAIN, {CHAIN[0]: model, CHAIN[1]: model}, cache, clock=clock)
        return AgentDeps(bundle=bundle, router=router)

    return backfill_reports(
        engine,
        deps,
        today=TODAY,
        build=two_bundles,
        out=lines.append,
        every=2,
        sleep=clock.sleep,
    )


def stored(engine: Engine) -> dict[tuple[str, int], Any]:
    with engine.connect() as connection:
        return dict(latest_reports(connection))


def test_backfill_writes_every_due_report_once(migrated_engine: Engine) -> None:
    nightly = sample_report("p1", 7)
    with migrated_engine.begin() as connection:  # the nightly job wrote this one, on these inputs
        write_report(connection, nightly, fingerprint=fingerprint("p1", 7), run_id=40)
    model = Scripted()
    lines: list[str] = []

    outcome = backfill(migrated_engine, model, lines)

    assert outcome == BackfillOutcome(due=3, written=3, skipped=0, stopped=None)
    reports = stored(migrated_engine)
    assert set(reports) == {("p1", 7), ("p1", 14), ("p0", 7), ("p0", 14)}
    assert reports[("p1", 7)].run_id == 40  # never redone
    assert not any(r.basic for r in reports.values())
    assert model.calls == 9  # gather, draft and summary for each
    assert lines[:-1] == ["2 of 3 reports done"]
    assert lines[-1] == "3 of 3 due reports written."
    (run,) = migrated_engine.connect().execute(
        text("SELECT kind, status, stages FROM ops.pipeline_run")
    )
    assert (run.kind, run.status) == ("reports_backfill", "succeeded")
    assert json.loads(run.stages)["written"] == 3

    again = backfill(migrated_engine, model, [])

    assert again == BackfillOutcome(due=0, written=0, skipped=0, stopped=None)
    assert model.calls == 9


def test_backfill_rewrites_reports_whose_inputs_changed(migrated_engine: Engine) -> None:
    # The nightly rewrites these a few a night; after a reclassification there are hundreds.
    # Run 0: a night before the backfill's run (the latest report is the highest run's).
    with migrated_engine.begin() as connection:
        write_report(connection, sample_report("p1", 7), fingerprint="0" * 64, run_id=0)
        basic = sample_report("p0", 14, basic=True)
        write_report(connection, basic, fingerprint=fingerprint("p0", 14), run_id=0)
    lines: list[str] = []

    outcome = backfill(migrated_engine, Scripted(), lines)

    assert outcome == BackfillOutcome(due=4, written=4, skipped=0, stopped=None)
    reports = stored(migrated_engine)
    assert reports[("p1", 7)].run_id > 0
    assert reports[("p1", 7)].fingerprint == fingerprint("p1", 7)
    assert not reports[("p0", 14)].basic


def test_backfill_stops_at_the_quota_and_resumes(migrated_engine: Engine) -> None:
    lines: list[str] = []

    # One report's three calls, then the next report's gather: its draft finds the quota spent.
    first = backfill(migrated_engine, Scripted(budget=4), lines)

    assert first == BackfillOutcome(
        due=4, written=1, skipped=0, stopped="the day's model quota is used up"
    )
    assert "1 of 4 due reports written" in lines[-1]
    assert "run the command again after midnight Pacific (07:00 UTC)" in lines[-1]
    assert "tomorrow" not in lines[-1]
    reports = stored(migrated_engine)
    assert list(reports) == [("p1", 14)]  # the newest; the cut-off report isn't stored as basic
    assert not reports[("p1", 14)].basic

    second = backfill(migrated_engine, Scripted(), [])

    assert second == BackfillOutcome(due=3, written=3, skipped=0, stopped=None)
    assert len(stored(migrated_engine)) == 4


def test_backfill_waits_out_an_outage_and_carries_on(migrated_engine: Engine) -> None:
    # Gemini's 503 "high demand" stopped the 3 Nov backfill twice; it waits and retries now.
    clock = FakeClock(START)
    lines: list[str] = []

    # Three outage retries: the third trips both models' breakers (resting 10 minutes).
    outcome = backfill(migrated_engine, Scripted(fail_first=6), lines, clock=clock)

    assert outcome == BackfillOutcome(due=4, written=4, skipped=0, stopped=None)
    assert not any(r.basic for r in stored(migrated_engine).values())
    assert sum(clock.slept) >= 600
    assert not any("tomorrow" in line for line in lines)


def test_a_report_no_model_can_answer_is_skipped_not_waited_for(
    migrated_engine: Engine,
) -> None:
    # Answers no step can use aren't an outage: the report is skipped and the run goes on (the
    # only wait is for the breakers the bad answers tripped).
    clock = FakeClock(START)

    outcome = backfill(migrated_engine, Scripted(garbled=True), [], clock=clock)

    assert outcome == BackfillOutcome(due=4, written=0, skipped=4, stopped=None)
    assert sum(clock.slept) <= 600


def test_backfill_gives_up_after_an_hour_of_outage(migrated_engine: Engine) -> None:
    clock = FakeClock(START)
    lines: list[str] = []

    outcome = backfill(migrated_engine, Scripted(fail_first=10**6), lines, clock=clock)

    assert outcome == BackfillOutcome(due=4, written=0, skipped=0, stopped=OUTAGE)
    assert stored(migrated_engine) == {}
    assert timedelta(minutes=60) <= clock.now - START < timedelta(minutes=72)
    assert "run the command again later" in lines[-1]
    assert "tomorrow" not in lines[-1]


def test_backfill_stops_for_the_quota_during_an_outage(migrated_engine: Engine) -> None:
    lines: list[str] = []

    outcome = backfill(migrated_engine, Scripted(budget=6, fail_first=6), lines)

    assert outcome == BackfillOutcome(due=4, written=0, skipped=0, stopped=QUOTA_SPENT)
    assert stored(migrated_engine) == {}  # no basic report stored
    assert "after midnight Pacific" in lines[-1]


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
