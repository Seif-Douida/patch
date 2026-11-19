"""Phase 7 Task 5: the weekly drift stage (design D1, D2, D11)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from monitoring_fixtures import SENTINEL_TEXT, add_rows, review_row, score_row
from sqlalchemy import Engine, event

from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.monitoring.alerts import AlertRule, load_rules
from patchpulse.monitoring.drift import DRIFT_RULES, WeekSample
from patchpulse.monitoring.reference import Reference, write_reference
from patchpulse.monitoring.store import read_metrics
from patchpulse.pipeline.monitor import (
    drill_resample,
    load_week,
    run_monitor,
    week_due,
    week_window,
)

TODAY = date(2026, 11, 23)  # a Monday
CHAMPION = "20261012T223218Z"
# The champion's scores as the scoring stage stores them (family-version). The first night (19 Nov)
# looked them up by the bare version: 0 reviews every week, the drift check never ran.
SCORED_AS = f"s0-{CHAMPION}"


def test_week_due_after_six_days() -> None:
    assert week_due(None, TODAY)
    assert not week_due(TODAY - timedelta(days=5), TODAY)
    assert week_due(TODAY - timedelta(days=6), TODAY)


def test_the_week_is_the_seven_full_days_before_today() -> None:
    assert week_window(TODAY) == (date(2026, 11, 16), date(2026, 11, 22))


def sample(languages: list[str]) -> WeekSample:
    n = len(languages)
    return WeekSample(
        review_ids=list(range(n)),
        languages=languages,
        lengths=[100] * n,
        labels=np.zeros((n, len(ASPECTS)), dtype=bool),
        probs=np.zeros((n, len(ASPECTS))),
    )


def test_the_drill_doubles_the_non_english_share() -> None:
    week = sample(["english"] * 900 + ["french"] * 100)
    drilled = drill_resample(week)

    assert len(drilled.review_ids) == 1000
    share = sum(lang != "english" for lang in drilled.languages) / 1000
    assert share == pytest.approx(0.2, abs=0.01)
    assert drilled.labels.shape == week.labels.shape


# --- integration: SQL Server -------------------------------------------------------------------


def seed(
    engine: Engine,
    *,
    n: int = 300,
    language: str = "english",
    created: datetime = datetime(2026, 11, 18, 12, 0),
    first: int = 1,
) -> None:
    ids = range(first, first + n)
    reviews = [review_row(i, created, language=language) for i in ids]
    scores = [score_row(i, created, model_version=SCORED_AS, aspects=["content"]) for i in ids]
    # Another model's scores on the same reviews must never be read.
    scores += [score_row(i, created, model_version="other", aspects=["performance"]) for i in ids]
    old = datetime(2026, 11, 1, 12, 0)  # outside the week
    add_rows(engine, [*reviews, review_row(first + 100_000, old)], scores)


@pytest.mark.integration
def test_the_week_uses_only_the_current_models_scores(migrated_engine: Engine) -> None:
    seed(migrated_engine)
    with migrated_engine.connect() as connection:
        week = load_week(connection, week_window(TODAY), scored_as=SCORED_AS)

    assert len(week.review_ids) == 300
    assert week.labels[:, ASPECTS.index("content")].all()
    assert not week.labels[:, ASPECTS.index("performance")].any()
    assert week.lengths[0] == len(SENTINEL_TEXT)


@pytest.mark.integration
def test_the_week_never_selects_review_text(migrated_engine: Engine) -> None:
    seed(migrated_engine, n=5)
    statements: list[str] = []

    def record(_c: Any, _cur: Any, statement: str, *_args: Any) -> None:
        statements.append(statement)

    event.listen(migrated_engine, "before_cursor_execute", record)
    try:
        with migrated_engine.connect() as connection:
            load_week(connection, week_window(TODAY), scored_as=SCORED_AS)
    finally:
        event.remove(migrated_engine, "before_cursor_execute", record)

    sql = " ".join(statements).replace(" ", "")
    assert "review_text" in sql
    assert sql.count("review_text") == sql.count("LEN(r.review_text)")


class FakeVectors:
    def vectors(self, ids: Sequence[int]) -> np.ndarray:
        rng = np.random.default_rng(len(ids))
        return rng.normal(size=(len(ids), 8)).astype(np.float32)


class Issues:
    def __init__(self) -> None:
        self.alerts: list[tuple[str, str, str]] = []
        self.resolved: list[str] = []

    def alert(self, *, title: str, body: str, label: str = "alert") -> None:
        self.alerts.append((title, label, body))

    def resolve(self, *, title: str, comment: str, label: str = "alert") -> None:
        self.resolved.append(title)


def reference(model_version: str = CHAMPION) -> Reference:
    return Reference(
        model_version=model_version,
        languages_top=["english", "french"],
        language_shares={"english": 0.9, "french": 0.1},
        length_shares={"bin4": 1.0},
        aspect_shares={code: 0.1 for code in ASPECTS},
        uncertain_share=0.0,
        thresholds={code: 0.5 for code in ASPECTS},
        teacher_agreement_macro_f1=0.7,
        sample_ids=list(range(1, 101)),
        n=1000,
    )


def uncalibrated() -> dict[str, AlertRule]:
    """The committed rules with every drift threshold unset (as before the calibration)."""
    rules = load_rules()
    return rules | {k: rules[k].model_copy(update={"threshold": None}) for k in DRIFT_RULES}


def thresholds(**values: float) -> dict[str, AlertRule]:
    rules = uncalibrated()
    return rules | {k: rules[k].model_copy(update={"threshold": v}) for k, v in values.items()}


def run(
    engine: Engine,
    tmp_path: Path,
    *,
    rules: dict[str, AlertRule],
    drill: str | None = None,
    model_version: str = CHAMPION,
    ref: Reference | None = None,
    today: date = TODAY,
    issues: Issues | None = None,
) -> tuple[Any, Issues]:
    path = tmp_path / "reference.json"
    write_reference(ref or reference(), path)
    issues = issues or Issues()
    outcome = run_monitor(
        engine,
        run_id=9,
        today=today,
        family="s0",
        model_version=model_version,
        vectors=FakeVectors(),
        publisher=issues,
        rules=rules,
        drill=drill,
        deadline=float("inf"),
        clock=lambda: 0.0,
        reference_path=path,
    )
    return outcome, issues


@pytest.mark.integration
def test_a_breach_opens_the_drift_issue_and_reports_new_drift(
    migrated_engine: Engine, tmp_path: Path
) -> None:
    seed(migrated_engine, language="french")
    outcome, issues = run(migrated_engine, tmp_path, rules=thresholds(drift_language=0.5))

    assert outcome.ran
    assert outcome.new_drift
    assert outcome.raised == ["drift: language mix"]
    assert issues.alerts[0][:2] == ("drift: language mix", "drift")
    with migrated_engine.connect() as connection:
        rows = read_metrics(connection, family="drift", scope="week")
    assert {r.window_end for r in rows} == {date(2026, 11, 22)}


@pytest.mark.integration
def test_a_lasting_drift_is_not_new(migrated_engine: Engine, tmp_path: Path) -> None:
    seed(migrated_engine, language="french")
    seed(migrated_engine, language="french", created=datetime(2026, 11, 25, 12, 0), first=5000)
    rules = thresholds(drift_language=0.5)
    run(migrated_engine, tmp_path, rules=rules)
    later, issues = run(migrated_engine, tmp_path, rules=rules, today=TODAY + timedelta(days=7))

    assert later.ran
    assert not later.new_drift
    assert issues.alerts[0][0] == "drift: language mix"


@pytest.mark.integration
def test_without_thresholds_it_reports_without_alerting(
    migrated_engine: Engine, tmp_path: Path
) -> None:
    seed(migrated_engine, language="french")
    outcome, issues = run(migrated_engine, tmp_path, rules=uncalibrated())

    assert outcome.ran
    assert not outcome.new_drift
    assert issues.alerts == []
    with migrated_engine.connect() as connection:
        assert read_metrics(connection, family="drift", scope="week")


@pytest.mark.integration
def test_a_drill_runs_whatever_the_timing_and_marks_everything(
    migrated_engine: Engine, tmp_path: Path
) -> None:
    seed(migrated_engine)
    rules = thresholds(drift_language=0.05)
    run(migrated_engine, tmp_path, rules=rules)  # this week's real check: no breach (English)
    outcome, issues = run(migrated_engine, tmp_path, rules=rules, drill="language_shift")

    assert outcome.ran
    assert outcome.new_drift
    assert issues.alerts[0][0] == "[drill] drift: language mix"
    with migrated_engine.connect() as connection:
        drill_rows = read_metrics(connection, family="drift", scope="drill")
    assert drill_rows


@pytest.mark.integration
def test_the_week_is_skipped_until_six_days_have_passed(
    migrated_engine: Engine, tmp_path: Path
) -> None:
    seed(migrated_engine)
    run(migrated_engine, tmp_path, rules=uncalibrated())
    outcome, _ = run(
        migrated_engine, tmp_path, rules=uncalibrated(), today=TODAY + timedelta(days=2)
    )

    assert not outcome.ran
    assert outcome.skipped == "not due"


@pytest.mark.integration
def test_a_week_with_too_few_reviews_is_skipped(migrated_engine: Engine, tmp_path: Path) -> None:
    seed(migrated_engine, n=20)
    outcome, issues = run(migrated_engine, tmp_path, rules=thresholds(drift_language=0.01))

    assert not outcome.ran
    assert outcome.skipped == "20 reviews this week, fewer than 200"
    assert issues.alerts == []


@pytest.mark.integration
def test_a_reference_for_another_model_skips_with_a_warning(
    migrated_engine: Engine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    seed(migrated_engine)
    outcome, issues = run(migrated_engine, tmp_path, rules=uncalibrated(), ref=reference("older"))

    assert not outcome.ran
    assert outcome.skipped is not None
    assert "older" in outcome.skipped
    assert issues.alerts == []
    assert "reference" in caplog.text


@pytest.mark.integration
def test_the_monitor_stores_the_weekly_agreement_and_raises_its_drop(
    migrated_engine: Engine, tmp_path: Path
) -> None:
    # Task 6 (pre-flight ruling): the weekly agreement runs with the drift check.
    from patchpulse.monitoring.store import StoredTeacherLabel, write_teacher_labels

    seed(migrated_engine)
    created = datetime(2026, 11, 18, 12, 0)
    labels = [
        StoredTeacherLabel(
            i, created, "gemma-4-31b-it", "p", "spotcheck", frozenset({"performance"})
        )
        for i in range(1, 301)
    ]
    with migrated_engine.begin() as connection:
        write_teacher_labels(connection, labels, run_id=1)
        connection.exec_driver_sql(
            "UPDATE raw.teacher_label SET created_at = '2026-11-19 03:00:00'"
        )

    outcome, issues = run(migrated_engine, tmp_path, rules=uncalibrated())

    with migrated_engine.connect() as connection:
        rows = read_metrics(connection, family="agreement", scope="week")
    macro = next(r for r in rows if r.metric == "macro_f1")
    assert macro.n == 300
    assert macro.value == pytest.approx(0.0)  # the champion says content, the teacher performance
    assert "teacher-student agreement dropped" in outcome.raised
    assert ("teacher-student agreement dropped", "alert") in [
        (t, lab) for t, lab, _ in issues.alerts
    ]
