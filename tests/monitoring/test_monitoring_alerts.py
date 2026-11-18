"""Phase 7 Task 2: one rule file for every alert, and the evaluator that raises and resolves."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from patchpulse.monitoring import nightly_watch
from patchpulse.monitoring.alerts import (
    ALERTS_FILE,
    DRILL_PREFIX,
    Outcome,
    apply_outcomes,
    load_rules,
)
from patchpulse.pipeline import runner
from patchpulse.publish import github

COST_GUARD = Path(".github/workflows/cost-guard.yml")
SENTINEL = "SENTINEL-REVIEW-the servers crash every evening since the patch"


class FakeIssues:
    """One issue per (label, title), as GitHubPublisher keeps them."""

    def __init__(self) -> None:
        self.open: dict[tuple[str, str], list[str]] = {}
        self.opened: list[tuple[str, str]] = []
        self.closed: list[tuple[str, str]] = []

    def alert(self, *, title: str, body: str, label: str = "alert") -> None:
        key = (label, title)
        if key not in self.open:
            self.opened.append(key)
            self.open[key] = []
        self.open[key].append(body)

    def resolve(self, *, title: str, comment: str, label: str = "alert") -> None:
        key = (label, title)
        if key in self.open:
            del self.open[key]
            self.closed.append(key)


def test_rules_file_lists_every_existing_alert() -> None:
    titles = {rule.title for rule in load_rules().values()}
    cost_guard = re.search(r'title="(cost-guard[^"]*)"', COST_GUARD.read_text(encoding="utf-8"))

    assert cost_guard is not None
    assert {
        github.FAILURE_TITLE,
        nightly_watch.FAILURE_TITLE,
        runner.OUT_OF_SYNC,
        runner.INDEX_SKIPPED,
        runner.QDRANT_UNREACHABLE,
        runner.BASIC_REPORTS,
        cost_guard.group(1),
    } <= titles
    assert load_rules()["qdrant_out_of_sync"].threshold == runner.MISMATCH_LIMIT


def test_the_rules_file_has_the_phase_7_rules() -> None:
    rules = load_rules()
    drift = {r for r in rules if r.startswith("drift_")}
    assert drift == {
        "drift_language",
        "drift_length",
        "drift_embedding",
        "drift_aspects",
        "drift_confidence",
    }
    assert all(rules[r].label == "drift" for r in drift)
    assert rules["agreement_dropped"].threshold == 0.05
    assert rules["teacher_unavailable"].threshold == 7
    assert rules["verifier_dropped"].threshold == 0.80
    assert rules["langfuse_units"].threshold == 40000
    assert rules["api_down"].threshold == 2
    assert rules["data_stale"].threshold == 60
    assert rules["sql_allowance"].threshold == 0.2


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ("{id: api_down, title: another, label: alert, where: uptime}", "duplicate alert id"),
        ("{id: other, title: pp-api down, label: alert, where: uptime}", "duplicate alert title"),
        ("{id: other, title: other, label: page, where: uptime}", "label"),
        ("{id: other, title: other, label: alert, where: moon}", "where"),
    ],
)
def test_duplicate_ids_or_titles_and_unknown_fields_are_refused(
    tmp_path: Path, extra: str, match: str
) -> None:
    rule = extra[:-1] + ", threshold: 1, meaning: x}"
    path = tmp_path / "alerts.yaml"
    path.write_text(ALERTS_FILE.read_text(encoding="utf-8") + f"- {rule}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_rules(path)


def test_a_breach_opens_its_issue_with_its_label() -> None:
    issues = FakeIssues()
    raised = apply_outcomes(
        issues, load_rules(), [Outcome("drift_language", True, "language_psi 0.41 > 0.22")]
    )

    assert raised == ["drift: language mix"]
    assert issues.opened == [("drift", "drift: language mix")]


def test_a_lasting_breach_comments_and_never_opens_a_second_issue() -> None:
    issues = FakeIssues()
    rules = load_rules()
    apply_outcomes(issues, rules, [Outcome("verifier_dropped", True, "week 1: 0.74")])
    apply_outcomes(issues, rules, [Outcome("verifier_dropped", True, "week 2: 0.71")])

    assert issues.opened == [("alert", "verifier pass rate dropped")]
    assert issues.open[("alert", "verifier pass rate dropped")] == ["week 1: 0.74", "week 2: 0.71"]


def test_recovery_closes_the_issue_with_its_numbers() -> None:
    issues = FakeIssues()
    rules = load_rules()
    apply_outcomes(issues, rules, [Outcome("data_stale", True, "data is 71 hours old")])
    apply_outcomes(issues, rules, [Outcome("data_stale", False, "data is 20 hours old")])

    assert issues.closed == [("alert", "data stale")]
    assert issues.open == {}


def test_an_unmeasured_rule_leaves_its_issue_alone() -> None:
    issues = FakeIssues()
    rules = load_rules()
    apply_outcomes(issues, rules, [Outcome("drift_embedding", True, "auc 0.81 > 0.70")])
    apply_outcomes(issues, rules, [Outcome("drift_embedding", None, "not measured")])

    assert issues.open[("drift", "drift: embeddings")] == ["auc 0.81 > 0.70"]
    assert issues.closed == []


def test_a_drill_never_touches_a_real_issue() -> None:
    issues = FakeIssues()
    rules = load_rules()
    apply_outcomes(issues, rules, [Outcome("drift_language", True, "real: 0.30")])

    raised = apply_outcomes(
        issues,
        rules,
        [Outcome("drift_language", True, "drill: 0.90"), Outcome("drift_length", False, "ok")],
        drill=True,
    )

    assert raised == [DRILL_PREFIX + "drift: language mix"]
    assert issues.open[("drift", "drift: language mix")] == ["real: 0.30"]
    assert issues.open[("drift", "[drill] drift: language mix")] == ["drill: 0.90"]
    assert issues.closed == []


def test_issue_bodies_hold_no_text() -> None:
    issues = FakeIssues()
    with pytest.raises(ValueError, match="text"):
        apply_outcomes(issues, load_rules(), [Outcome("drift_length", True, SENTINEL)])
    with pytest.raises(ValueError, match="text"):
        apply_outcomes(issues, load_rules(), [Outcome("drift_length", True, "x" * 201)])
    assert issues.opened == []


def test_an_unknown_rule_is_refused() -> None:
    with pytest.raises(KeyError):
        apply_outcomes(FakeIssues(), load_rules(), [Outcome("no_such_rule", True, "1")])
