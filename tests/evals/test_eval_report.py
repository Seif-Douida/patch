"""The eval report (Phase 6 plan Task 5, design D9): numbers, ids and categories only, never
review, claim or summary text; the interval caveat and the scoring version stated; logged to
MLflow with the report as the one artifact."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import mlflow
import pytest
from eval_fixtures import WINDOW, aspects_call, claim, hypotheses, incident
from eval_fixtures import report as make_report

from patchpulse.agent.graph import RunRecord
from patchpulse.agent.report import Draft, WithheldClaim
from patchpulse.agent.tools import Trace
from patchpulse.evals.injection import InjectionOutcome, injection_metrics
from patchpulse.evals.metrics import Estimate
from patchpulse.evals.records import StoredCase
from patchpulse.evals.report import (
    EvalResults,
    RunInfo,
    build_results,
    log_eval_run,
    render_report,
    results_json,
    write_report,
)
from patchpulse.evals.scoring import SCORING_VERSION

SENTINEL = "SENTINEL-TEXT"
INFO = RunInfo(
    date="2026-11-15",
    data_version=60,
    scoring_version=SCORING_VERSION,
    prompt_version="p1",
    rubric_version=None,
    chains={"main": ("gemini-3.5-flash-lite",)},
    judge_chain=None,
    commit="abc1234",
)


def texty_case(case: str, arm: str = "main") -> StoredCase:
    """A run whose every free text carries the sentinel."""
    shipped = (claim("c1", f"{SENTINEL} claim").model_copy(update={"review_ids": [101]}),)
    withheld = (WithheldClaim(claim=claim("c2", f"{SENTINEL} withheld"), rule="V7"),)
    result = make_report(hypotheses(("H1",)), claims=shipped, withheld=withheld)
    result = result.model_copy(update={"summary": f"{SENTINEL} summary"})
    trace = Trace(calls=[aspects_call(1.0, 2.0)], retrieved={101: f"{SENTINEL} review"})
    draft = Draft(findings=[], claims=[*shipped, withheld[0].claim])
    record = RunRecord(result, WINDOW, result.hypotheses, draft, f"{SENTINEL} raw", trace, False)
    return StoredCase.from_record(
        record, case=case, arm=arm, data_version=60, misses=[], indexed_reviews=0
    )


def results() -> object:
    incidents = [incident("patch", incident_id="p0"), incident("patch", incident_id="p1")]
    arms = {
        "main": {"p0": texty_case("p0"), "p1": texty_case("p1")},
        "a2": {"p0": texty_case("p0", "a2"), "p1": texty_case("p1", "a2")},
    }
    return build_results(incidents, arms, INFO, verdicts={"main/p0/c1/r101": False})


def test_reports_carry_no_text() -> None:
    built = results()

    markdown = render_report(built)  # type: ignore[arg-type]
    document = json.dumps(results_json(built))  # type: ignore[arg-type]

    assert SENTINEL not in markdown
    assert SENTINEL not in document
    assert "p0" in markdown
    assert "patch" in markdown


def test_the_report_states_the_interval_caveat_and_scoring_version() -> None:
    markdown = render_report(results())  # type: ignore[arg-type]

    assert "17 points" in markdown
    assert f"scoring {SCORING_VERSION}" in markdown
    assert "5 of 5 by construction" in markdown
    assert "No indexed reviews in the scoped window: p0, p1" in markdown
    assert "reports/causal/benchmark.md" in markdown
    assert "| Cited reviews supporting their claim (judge) | 0.0% |" in markdown
    assert "| Verifier on | 100.0% |" in markdown
    assert "| a2 | citation_support | - |" in markdown


def test_the_injection_rates_are_called_lower_bounds() -> None:
    """Success is matched by fixed words: a paraphrased or translated success isn't counted
    (review of the set, 9 Nov)."""
    outcome = InjectionOutcome("inj-01", "instruction", exposed=True, before=False, after=False)
    base = results()
    assert isinstance(base, EvalResults)
    built = replace(base, injection=injection_metrics([outcome]))

    section = render_report(built).split("## Injection set")[1].split("## ")[0]

    assert "lower bounds" in section
    assert "| cited_after_rate |" in section


def test_log_eval_run_logs_metrics_and_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    built = results()
    markdown, document = write_report(built)  # type: ignore[arg-type]

    run_id = log_eval_run(built, markdown)  # type: ignore[arg-type]

    logged = mlflow.get_run(run_id)
    assert markdown == Path("reports/evals/2026-11-15.md")
    assert json.loads(document.read_text(encoding="utf-8"))["info"]["data_version"] == 60
    assert logged.data.params["data_version"] == "60"
    assert logged.data.params["scoring_version"] == SCORING_VERSION
    assert logged.data.metrics["root_cause_accuracy"] == 1.0
    artifacts = [a.path for a in mlflow.MlflowClient().list_artifacts(run_id)]
    assert artifacts == ["2026-11-15.md"]


def test_a_failed_judge_calibration_is_stated_with_the_raw_agreement() -> None:
    """Seif's ruling (16 Nov 2026): report the raw agreement; the missed kappa bar is still
    stated and the judge's scores are called uncalibrated."""
    judge = {
        "citation.agreement_rate": Estimate(26 / 31, 0.71, 0.97, 31),
        "overall.agreement_rate": Estimate(40 / 47, 0.74, 0.94, 47),
        "overall.seif_supported_rate": Estimate(46 / 47, None, None, 47),
        "overall.judge_supported_rate": Estimate(41 / 47, None, None, 47),
        "kappa.citation": Estimate(-0.05, -0.12, 0.0, 31),
    }
    base = results()
    assert isinstance(base, EvalResults)

    failed = render_report(replace(base, judge=judge, judge_passed=False))
    passed = render_report(replace(base, judge=judge, judge_passed=True))
    section = failed.split("## Judge agreement with Seif")[1].split("## ")[0]

    assert "| citation.agreement_rate | 83.9% | 71.0% to 97.0% | 31 |" in section
    assert "| kappa.citation | -0.05 |" in section
    assert "kappa of at least 0.6 on each question, fixed before labeling, was not met" in section
    assert "uncalibrated" in section
    assert 'Seif answered "supported" on 46 of 47 compared items, the judge on 41' in section
    assert "(uncalibrated: see Judge agreement with Seif)" in failed.split("## Headline")[0]
    assert "uncalibrated" not in passed
    assert "was met" in passed
    assert "uncalibrated" not in render_report(base)
