"""The nightly reports stage's work (Phase 5 plan Task 12, design D10), on the SQL Server
container with a fake agent."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
from agent_fixtures import SAMPLE_QUOTE, make_bundle, sample_report
from sqlalchemy import Engine

from patchpulse.agent.graph import ReportRequest
from patchpulse.agent.report import Report
from patchpulse.agent.store import latest_reports
from patchpulse.config import Settings
from patchpulse.pipeline.reports import ReportsOutcome, reports_file, run_reports

pytestmark = pytest.mark.integration


class Clock:
    """Each reading is the next of `readings`."""

    def __init__(self, *readings: float) -> None:
        self.readings = list(readings)

    def __call__(self) -> float:
        return self.readings.pop(0)


def agent(*, basic: bool = False) -> Callable[[ReportRequest], Report]:
    def write(request: ReportRequest) -> Report:
        return sample_report(str(request.bundle_id), int(request.horizon or 0), basic=basic)

    return write


NO_STEAM: Any = None  # the fake agents never read patch notes


def run(
    engine: Engine,
    settings: Settings,
    clock: Clock,
    agent: Callable[[ReportRequest], Report] | None = None,
) -> ReportsOutcome:
    return run_reports(
        engine,
        settings,
        make_bundle(),
        run_id=61,
        steam=NO_STEAM,
        clock=clock,
        deadline=50.0,
        agent=agent,
    )


def test_reports_stop_at_the_deadline_newest_first(
    migrated_engine: Engine, sql_settings: Settings
) -> None:
    outcome = run(migrated_engine, sql_settings, Clock(0.0, 100.0), agent=agent())

    assert outcome == ReportsOutcome(due=2, written=1, basic=0, failed=0, remaining=1)
    with migrated_engine.connect() as connection:
        assert set(latest_reports(connection)) == {("p1", 14)}  # the latest window first


def test_basic_reports_are_counted_and_due_again(
    migrated_engine: Engine, sql_settings: Settings
) -> None:
    first = run(migrated_engine, sql_settings, Clock(0.0, 0.0), agent=agent(basic=True))
    again = run(migrated_engine, sql_settings, Clock(0.0, 0.0), agent=agent(basic=True))

    assert first == ReportsOutcome(due=2, written=2, basic=2, failed=0, remaining=0)
    # Still due; the same evidence and still basic, so nothing new is stored.
    assert again == ReportsOutcome(due=2, written=0, basic=2, failed=0, remaining=0)


def test_nothing_due_needs_no_model(migrated_engine: Engine, sql_settings: Settings) -> None:
    run(migrated_engine, sql_settings, Clock(0.0, 0.0), agent=agent())
    no_key = sql_settings.model_copy(update={"gemini_api_key": None})

    # The live agent (which needs the key) is built only when a report is due.
    outcome = run(migrated_engine, no_key, Clock())

    assert outcome == ReportsOutcome(due=0, written=0, basic=0, failed=0, remaining=0)


def test_reports_file_holds_public_copies_and_the_patch_map(
    migrated_engine: Engine, sql_settings: Settings
) -> None:
    run(migrated_engine, sql_settings, Clock(0.0, 0.0), agent=agent())
    with migrated_engine.connect() as connection:
        stored = latest_reports(connection)
    bundle = replace(make_bundle(), data_version=61)

    content = reports_file(bundle, stored, data_version=61)

    document = json.loads(content)
    assert document["data_version"] == 61
    assert document["patches"] == {"p0": "p0", "p1": "p1", "p2": "p1"}
    assert document["reports"]["p1"]["horizon"] == 14
    text = content.decode("utf-8")
    assert SAMPLE_QUOTE not in text
    assert "7003" not in text
    assert "The patch caused the drop." not in text
