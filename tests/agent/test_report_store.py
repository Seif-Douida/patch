"""app.report: the stored reports and when one is due (plan Task 11, design D10, spec §10.8), on
the SQL Server container.

A report is redone only when its evidence changes (the core tool results for its window, its
estimate rows, the rules' settings) or the prompt version does, never to get new wording; a basic
report is due again the next night.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date

import pytest
from agent_fixtures import ESTIMATES, PATCH_DAY, SAMPLE_QUOTE, make_bundle, sample_report
from sqlalchemy import Engine, text

from patchpulse.agent.bundle import Bundle, GameData
from patchpulse.agent.store import (
    StoredReport,
    latest_reports,
    report_fingerprint,
    report_inputs,
    reports_due,
    shown_reports,
    write_report,
)
from patchpulse.causal.store import StoredEstimate

pytestmark = pytest.mark.integration


def write(
    engine: Engine, bundle: Bundle, key: tuple[str, int], run_id: int, *, basic: bool = False
) -> bool:
    fingerprint = report_fingerprint(report_inputs(bundle, *key))
    with engine.begin() as connection:
        return write_report(connection, sample_report(*key, basic=basic), fingerprint=fingerprint,
                            run_id=run_id)  # fmt: skip


def stored(engine: Engine) -> dict[tuple[str, int], StoredReport]:
    with engine.connect() as connection:
        return dict(latest_reports(connection))


def rows(engine: Engine) -> int:
    with engine.connect() as connection:
        return int(connection.execute(text("SELECT COUNT(*) FROM app.report")).scalar_one())


def game(bundle: Bundle) -> GameData:
    return bundle.games[0]


def with_game(bundle: Bundle, changed: GameData) -> Bundle:
    return replace(bundle, games=(changed, *bundle.games[1:]))


def with_estimates(bundle: Bundle, estimates: tuple[StoredEstimate, ...]) -> Bundle:
    return with_game(bundle, replace(game(bundle), estimates=estimates))


def with_count(bundle: Bundle, day: date, totals: int) -> Bundle:
    days = tuple(replace(d, totals=totals) if d.date == day else d for d in game(bundle).days)
    return with_game(bundle, replace(game(bundle), days=days))


def test_reports_are_redone_only_when_their_evidence_changes(migrated_engine: Engine) -> None:
    bundle = make_bundle()
    assert reports_due(bundle, {}) == [("p1", 14), ("p1", 7)]

    assert write(migrated_engine, bundle, ("p1", 7), run_id=52)
    assert not write(migrated_engine, bundle, ("p1", 7), run_id=53)  # same inputs: no new row
    assert rows(migrated_engine) == 1
    assert reports_due(bundle, stored(migrated_engine)) == [("p1", 14)]

    # A later night's data after the window changes nothing; a day inside it does.
    later = replace(with_count(bundle, date(2026, 10, 9), 999), data_version=53)
    assert reports_due(later, stored(migrated_engine)) == [("p1", 14)]
    inside = with_count(bundle, PATCH_DAY, 999)
    assert reports_due(inside, stored(migrated_engine)) == [("p1", 14), ("p1", 7)]

    # A changed 7-day estimate: due, and its report is a new row.
    estimates = tuple(
        replace(e, effect=-8.0, fingerprint="c" * 64)
        if (e.horizon, e.method, e.outcome, e.sample) == (7, "synth", "positive_share", "all")
        else e
        for e in ESTIMATES
    )
    changed = with_estimates(bundle, estimates)
    assert reports_due(changed, stored(migrated_engine)) == [("p1", 14), ("p1", 7)]
    assert write(migrated_engine, changed, ("p1", 7), run_id=54)
    assert rows(migrated_engine) == 2
    assert stored(migrated_engine)[("p1", 7)].run_id == 54

    # A new prompt version: due.
    assert reports_due(changed, stored(migrated_engine)) == [("p1", 14)]
    assert reports_due(changed, stored(migrated_engine), prompt_version="0" * 12) == [
        ("p1", 14),
        ("p1", 7),
    ]


def test_a_basic_report_is_due_again(migrated_engine: Engine) -> None:
    bundle = make_bundle()
    assert write(migrated_engine, bundle, ("p1", 7), run_id=52, basic=True)
    assert ("p1", 7) in reports_due(bundle, stored(migrated_engine))

    assert write(migrated_engine, bundle, ("p1", 7), run_id=53)  # the next night's, with a model
    assert ("p1", 7) not in reports_due(bundle, stored(migrated_engine))
    # A basic one never replaces a model-written report on the same evidence.
    assert not write(migrated_engine, bundle, ("p1", 7), run_id=54, basic=True)
    assert stored(migrated_engine)[("p1", 7)].basic is False


def test_the_14_day_report_follows_the_7_day_one(migrated_engine: Engine) -> None:
    full = make_bundle()
    seven = with_estimates(full, tuple(e for e in ESTIMATES if e.horizon == 7))
    assert reports_due(seven, {}) == [("p1", 7)]
    write(migrated_engine, seven, ("p1", 7), run_id=52)
    assert reports_due(seven, stored(migrated_engine)) == []

    # The 14-day rows arrive a week later; the 7-day report's evidence is unchanged.
    assert reports_due(full, stored(migrated_engine)) == [("p1", 14)]
    write(migrated_engine, full, ("p1", 14), run_id=59)
    assert reports_due(full, stored(migrated_engine)) == []
    assert set(stored(migrated_engine)) == {("p1", 7), ("p1", 14)}  # both stay stored


def test_due_reports_are_newest_first(migrated_engine: Engine) -> None:
    bundle = make_bundle()
    older = tuple(replace(e, bundle_id="p0", patch_ids=("p0",)) for e in ESTIMATES)
    both = with_estimates(bundle, ESTIMATES + older)
    assert reports_due(both, {}) == [("p1", 14), ("p1", 7), ("p0", 14), ("p0", 7)]


def test_a_stored_report_keeps_both_copies(migrated_engine: Engine) -> None:
    bundle = make_bundle()
    write(migrated_engine, bundle, ("p1", 7), run_id=52)
    row = stored(migrated_engine)[("p1", 7)]

    assert row.report == sample_report()
    assert row.fingerprint == report_fingerprint(report_inputs(bundle, "p1", 7))
    public = json.dumps(row.public)
    assert SAMPLE_QUOTE not in public
    assert "7001" not in public
    assert "7003" not in public
    assert "The patch caused the drop." not in public
    with migrated_engine.connect() as connection:
        raw: str = connection.execute(text("SELECT report_json FROM app.report")).scalar_one()
    assert SAMPLE_QUOTE in raw  # the stored copy keeps the quote span


def test_a_question_report_is_never_stored(migrated_engine: Engine) -> None:
    asked = sample_report().model_copy(
        update={"scope": "question", "bundle_id": None, "horizon": None}
    )
    with migrated_engine.begin() as connection, pytest.raises(ValueError, match="question"):
        write_report(connection, asked, fingerprint="0" * 64, run_id=52)


def test_the_site_shows_the_14_day_report_once_it_exists(migrated_engine: Engine) -> None:
    bundle = make_bundle()
    write(migrated_engine, bundle, ("p1", 7), run_id=52)
    shown = shown_reports(stored(migrated_engine))
    assert {b: r.horizon for b, r in shown.items()} == {"p1": 7}

    write(migrated_engine, bundle, ("p1", 14), run_id=59)
    shown = shown_reports(stored(migrated_engine))
    assert {b: r.horizon for b, r in shown.items()} == {"p1": 14}
