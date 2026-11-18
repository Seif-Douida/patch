"""Phase 7 Task 12: retrain.yml starts pp-retrain and opens the promotion pull request as the
GitHub App (design D5-D6)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from image_groups import run_with_only

from patchpulse.pipeline.runner import RETRAIN_EVENT

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "retrain.yml"
TEXT = WORKFLOW.read_text(encoding="utf-8")
SPEC: dict[Any, Any] = yaml.safe_load(TEXT)
TRIGGERS: dict[str, Any] = SPEC["on"] if "on" in SPEC else SPEC[True]  # PyYAML: `on` is True
(JOB,) = SPEC["jobs"].values()
STEPS: list[dict[str, Any]] = JOB["steps"]


def step(name: str) -> dict[str, Any]:
    (found,) = [s for s in STEPS if s.get("name", "").startswith(name)]
    return found


def test_only_the_dispatch_and_manual_triggers_start_it() -> None:
    assert set(TRIGGERS) == {"repository_dispatch", "workflow_dispatch"}
    assert TRIGGERS["repository_dispatch"]["types"] == [RETRAIN_EVENT]
    inputs = TRIGGERS["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"force", "reason"}
    assert inputs["force"]["type"] == "boolean"
    assert inputs["force"]["default"] is False
    assert SPEC["concurrency"] == {"group": "retrain", "cancel-in-progress": False}
    assert JOB["environment"] == "production"


def test_it_refuses_while_pp_api_is_stopped() -> None:
    # As deploy.yml does: a stopped pp-api means cost-guard tripped (ADR-017).
    check = step("Refuse while pp-api is stopped")["run"]
    assert "properties.runningStatus" in check
    assert '"Stopped"' in check
    assert "exit 1" in check


def test_a_second_request_within_twelve_hours_is_refused() -> None:
    guard = step("Refuse a second run within 12 hours")
    assert "az containerapp job execution list" in guard["run"]
    assert "patchpulse.monitoring.retrain_result guard" in guard["run"]
    # Only by hand, with `force`, does a recent execution not stop it.
    assert (
        guard["env"]["FORCE"] == "${{ github.event_name == 'workflow_dispatch' && inputs.force }}"
    )
    assert step("Start pp-retrain")["if"] == "steps.guard.outputs.skip != 'true'"


def test_the_job_is_awaited_for_its_whole_timeout() -> None:
    start = step("Start pp-retrain")["run"]
    assert "az containerapp job start" in start
    assert "-n pp-retrain" in start
    attempts, interval = re.search(r"seq 1 (\d+)\).*?sleep (\d+)", start, re.DOTALL).groups()  # type: ignore[union-attr]
    assert int(attempts) * int(interval) >= 50 * 60  # the job's own 45 min, and its start


def test_a_failed_job_fails_the_workflow_and_opens_no_pr() -> None:
    start = step("Start pp-retrain")["run"]
    assert re.search(r"Failed\|Stopped\|Degraded\)[^;]*\n[^;]*exit 1", start)
    for later in STEPS[STEPS.index(step("Start pp-retrain")) + 1 :]:
        condition = str(later.get("if", ""))
        assert not re.search(r"always\(\)|failure\(\)|cancelled\(\)", condition), later["name"]
    assert step("Open the promotion pull request")["if"] == "steps.result.outputs.promote == 'true'"


def test_the_promotion_pr_uses_the_app_token_and_changes_two_files() -> None:
    app = step("Get a token from the GitHub App")
    assert app["uses"] == "actions/create-github-app-token@v3"
    assert app["with"]["app-id"] == "${{ vars.PP_GITHUB_APP_ID }}"
    assert app["with"]["private-key"] == "${{ secrets.PP_GITHUB_APP_PRIVATE_KEY }}"
    assert app["if"] == "steps.result.outputs.promote == 'true'"
    pr = step("Open the promotion pull request")
    assert pr["env"]["GH_TOKEN"] == "${{ steps.app.outputs.token }}"
    assert "git add models/champion.yaml models/reference.json\n" in pr["run"]
    assert 'git switch -c "retrain/$VERSION"' in pr["run"]
    assert 'gh pr create --base main --head "retrain/$VERSION"' in pr["run"]
    assert '--title "Promote aspect-classifier v$REGISTERED ($REASON)"' in pr["run"]
    # GITHUB_TOKEN's pull requests start no workflows, so the required checks would never run.
    assert "secrets.GITHUB_TOKEN" not in pr["run"] + str(pr["env"])
    assert JOB["permissions"]["contents"] == "read"


def test_without_a_promotion_the_table_goes_to_the_drift_issue() -> None:
    report = step("Report on the drift issue")
    assert report["if"] == "steps.result.outputs.promote == 'false'"
    assert "--label drift" in report["run"]
    assert "gh issue comment" in report["run"]
    assert JOB["permissions"]["issues"] == "write"


def test_untrusted_values_reach_the_scripts_through_env_only() -> None:
    # The dispatch payload and the inputs are data: never interpolated into a script.
    for s in STEPS:
        assert "${{" not in str(s.get("run", "")), s.get("name")


def test_the_result_reader_imports_with_only_the_workflows_groups() -> None:
    reader = step("Read the result from MLflow")["run"]
    groups = set(re.findall(r"--group (\w+)", reader))
    assert groups

    result = run_with_only(
        groups, "import patchpulse.monitoring.retrain_result\nprint('imported')\n"
    )

    assert result.returncode == 0, result.stderr[-1500:]
    assert result.stdout.strip() == "imported"
