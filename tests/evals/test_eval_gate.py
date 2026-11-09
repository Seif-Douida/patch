"""The gate (Phase 6 plan Task 11, design D11): a PR's smoke metrics against `main`'s newest
baseline, with the tolerances in evals/thresholds.yaml, and the PR comment."""

from __future__ import annotations

from pathlib import Path

import pytest
from eval_fixtures import smoke_metrics

from patchpulse.evals.gate import Thresholds, compare, read_thresholds, render_comment
from patchpulse.evals.smoke import SmokeMetrics

REPO = Path(__file__).resolve().parents[2]


def test_the_repo_thresholds_are_the_designs() -> None:
    assert read_thresholds(REPO / "evals" / "thresholds.yaml") == Thresholds(
        root_cause_correct=-1,
        cause_alarms=1,
        verifier_regressions=0,
        injection_after=0,
        ndcg_at_10=-0.02,
        macro_f1=-0.02,
        unreplayed_cases=0,
    )


def test_equal_metrics_pass() -> None:
    result = compare(smoke_metrics(), smoke_metrics(), Thresholds())

    assert result.passed
    assert result.breaches == ()
    assert all(row.ok for row in result.rows)


@pytest.mark.parametrize(
    ("metric", "changes"),
    [
        ("root_cause_correct", {"root_cause_correct": 18}),
        ("cause_alarms", {"cause_alarms": 4}),
        ("injection_after", {"injection_after": 1}),
        ("ndcg_at_10", {"ndcg_at_10": 0.8499}),
        ("macro_f1", {"macro_f1": 0.5939}),
        (
            "verifier_regressions",
            {"verifier": {"hd2-a": {"c1": "shipped", "c2": "shipped", "c3": "reworded"}}},
        ),
        ("unreplayed_cases", {"unreplayed": {"hd2-a": ["model:draft"]}}),
    ],
)
def test_each_threshold_breached_by_one_fails(metric: str, changes: dict[str, object]) -> None:
    result = compare(smoke_metrics(**changes), smoke_metrics(), Thresholds())

    assert not result.passed
    assert result.breaches == (metric,)


def test_a_change_at_the_tolerance_passes() -> None:
    pr = smoke_metrics(root_cause_correct=19, cause_alarms=3, ndcg_at_10=0.85, macro_f1=0.594)

    assert compare(pr, smoke_metrics(), Thresholds()).passed


def test_a_case_that_misses_fails_the_gate_and_is_named() -> None:
    pr = smoke_metrics(unreplayed={"hd2-a": ["model:draft", "model:summary"], "inj-03": ["tool:x"]})

    result = compare(pr, smoke_metrics(), Thresholds())
    comment = render_comment(result)

    assert "unreplayed_cases" in result.breaches
    assert result.unreplayed == {"hd2-a": ["model:draft", "model:summary"], "inj-03": ["tool:x"]}
    assert "hd2-a (model:draft, model:summary)" in comment
    assert "inj-03 (tool:x)" in comment
    assert "patchpulse-evals run --arm main --mode fill --cases hd2-a" in comment
    assert "patchpulse-evals run --arm injection --mode fill --cases inj-03" in comment
    assert "patchpulse-evals bundle --version 2" in comment


def test_a_verifier_regression_is_a_claim_main_withheld_that_the_pr_ships() -> None:
    shipped = {"hd2-a": {"c1": "shipped", "c2": "shipped", "c3": "shipped"}}
    withheld = {"hd2-a": {"c1": "withheld:V7", "c2": "withheld:V2", "c3": "withheld:V4"}}

    loosened = compare(smoke_metrics(verifier=shipped), smoke_metrics(), Thresholds())
    tightened = compare(smoke_metrics(verifier=withheld), smoke_metrics(), Thresholds())

    assert loosened.regressions == ("hd2-a/c2", "hd2-a/c3")
    assert loosened.breaches == ("verifier_regressions",)
    assert tightened.regressions == ()
    assert tightened.passed  # stricter is never a regression
    assert "hd2-a/c2" in render_comment(loosened)


def test_without_a_baseline_the_gate_reports_only() -> None:
    result = compare(smoke_metrics(root_cause_correct=0), None, Thresholds())

    assert result.passed
    assert not result.baseline
    assert all(row.ok is None for row in result.rows if row.metric != "unreplayed_cases")
    assert "no baseline yet" in render_comment(result)


def test_unreplayed_cases_fail_even_without_a_baseline() -> None:
    """Every case must replay: an absolute rule, baseline or not (Final review I4)."""
    result = compare(smoke_metrics(unreplayed={"x": ["model:scope"]}), None, Thresholds())

    assert result.breaches == ("unreplayed_cases",)
    assert render_comment(result).splitlines()[0].startswith("## Eval gate: failed")


def test_a_claim_main_withheld_that_the_pr_rewords_is_a_regression() -> None:
    """A reworded claim ships, hedged, in the public copy (Final review I8)."""
    reworded = {"hd2-a": {"c1": "shipped", "c2": "reworded", "c3": "reworded"}}

    result = compare(smoke_metrics(verifier=reworded), smoke_metrics(), Thresholds())

    assert result.regressions == ("hd2-a/c2",)


def test_a_re_recorded_case_is_not_compared_claim_by_claim() -> None:
    """Claim ids are per draft: a case recorded again has other claims (Final review I3)."""
    pr = smoke_metrics(
        verifier={"hd2-a": {"c1": "shipped", "c2": "shipped", "c3": "shipped"}},
        recorded={"hd2-a": "c" * 64, "bg3-b": "b" * 64},
    )

    result = compare(pr, smoke_metrics(), Thresholds())

    assert result.passed
    assert result.regressions == ()
    assert result.rerecorded == ("hd2-a",)
    assert "Recorded again since main's baseline (verifier not compared): hd2-a." in (
        render_comment(result)
    )


def test_a_baseline_with_other_fields_still_compares() -> None:
    """A PR that adds or drops a metric can still read main's baseline (Final review I5)."""
    document = smoke_metrics().to_json()
    document["a_later_metric"] = 3
    del document["macro_f1"]

    base = SmokeMetrics.from_json(document)
    result = compare(smoke_metrics(), base, Thresholds())

    assert base.macro_f1 is None
    assert result.passed
    macro = next(row for row in result.rows if row.metric == "macro_f1")
    assert macro.ok is None  # nothing to compare with: reported, not gated


def test_the_comment_holds_numbers_and_ids_only() -> None:
    result = compare(smoke_metrics(root_cause_correct=18), smoke_metrics(), Thresholds())

    comment = render_comment(result)

    assert "| Golden incidents right | 20 | 18 | -1 | FAIL |" in comment
    assert "| NDCG@10 (pool) | 0.870 | 0.870 | -0.02 | pass |" in comment
    assert "Judge (replayed, reported only): citation support 0.900" in comment
    assert "bundle v1" in comment
    assert "failed" in comment.splitlines()[0]
