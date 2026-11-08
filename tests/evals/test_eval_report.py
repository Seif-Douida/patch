"""The eval report (Phase 6 plan Task 5, design D9): numbers, ids and categories only, never
review, claim or summary text; the interval caveat and the scoring version stated; logged to
MLflow with the report as the one artifact."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest
from eval_fixtures import WINDOW, aspects_call, claim, hypotheses, incident
from eval_fixtures import report as make_report

from patchpulse.agent.graph import RunRecord
from patchpulse.agent.report import Draft, WithheldClaim
from patchpulse.agent.tools import Trace
from patchpulse.evals.records import StoredCase
from patchpulse.evals.report import (
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
    shipped = (claim("c1", f"{SENTINEL} claim"),)
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
    return build_results(incidents, arms, INFO, support={"p0": {"c1": False}})


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
