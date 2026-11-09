"""eval-gate.yml (Phase 6 plan Task 12, design D11): every PR replays the pinned eval bundle, with
no model, Azure or cluster; main's runs keep the baseline; the comment is posted before the check
fails."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def load(name: str) -> dict[Any, Any]:
    workflow: dict[Any, Any] = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    return workflow


def gate() -> dict[Any, Any]:
    return load("eval-gate.yml")


def steps() -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = gate()["jobs"]["eval-gate"]["steps"]
    return found


def step(name: str) -> dict[str, Any]:
    (found,) = [s for s in steps() if s.get("name") == name]
    return found


def index(name: str) -> int:
    return next(i for i, s in enumerate(steps()) if s.get("name") == name)


def test_the_gate_runs_on_every_pull_request() -> None:
    triggers = gate()[True]  # PyYAML reads the `on:` key as True.

    assert set(triggers) == {"pull_request", "push", "workflow_dispatch"}
    assert not (triggers["pull_request"] or {}).get("paths")  # a required check never waits
    assert triggers["push"] == {"branches": ["main"]}
    relevant = step("Relevant changes")["run"]
    for path in ("src/", "config/", "evals/", "models/", "pyproject", "uv", "eval-gate"):
        assert path in relevant, path


def test_the_job_is_named_eval_gate_and_bounded() -> None:
    workflow = gate()
    job = workflow["jobs"]["eval-gate"]

    assert job["name"] == "eval-gate"
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["timeout-minutes"] == 15
    assert workflow["concurrency"]["group"] == "eval-gate-${{ github.ref }}"
    assert workflow["concurrency"]["cancel-in-progress"] is True


def test_least_permissions() -> None:
    workflow = gate()

    assert workflow["permissions"] == {
        "contents": "read",
        "pull-requests": "write",
        "actions": "read",
    }
    assert "permissions" not in workflow["jobs"]["eval-gate"]
    assert "environment" not in workflow["jobs"]["eval-gate"]
    assert step("Check out")["with"]["persist-credentials"] is False


def test_no_model_azure_or_cluster_secrets() -> None:
    text = (WORKFLOWS / "eval-gate.yml").read_text(encoding="utf-8")

    assert set(re.findall(r"secrets\.(\w+)", text)) == {"EVAL_BUNDLE_KEY"}
    assert set(re.findall(r"vars\.(\w+)", text)) == {"MLFLOW_TRACKING_URI"}
    for forbidden in ("azure/login", "environment:", "GEMINI", "PP_QDRANT", "production"):
        assert forbidden not in text, forbidden
    assert step("Smoke")["env"]["PP_EVAL_BUNDLE_KEY"] == "${{ secrets.EVAL_BUNDLE_KEY }}"
    download = step("Download the champion model")
    assert download["env"] == {"MLFLOW_TRACKING_URI": "${{ vars.MLFLOW_TRACKING_URI }}"}


def test_pinned_like_ci() -> None:
    ci = load("ci.yml")
    ci_steps = ci["jobs"]["integration"]["steps"]
    ci_checkout = next(s for s in ci_steps if s.get("uses", "").startswith("actions/checkout"))
    ci_uv = next(s for s in ci_steps if s.get("uses", "").startswith("astral-sh/setup-uv"))

    assert step("Check out")["uses"] == ci_checkout["uses"]
    uv = next(s for s in steps() if s.get("uses", "").startswith("astral-sh/setup-uv"))
    assert uv["uses"] == ci_uv["uses"]
    assert uv["with"]["version"] == ci_uv["with"]["version"]
    assert gate()["env"]["QDRANT_IMAGE"] == ci["env"]["QDRANT_IMAGE"]
    assert "uv sync --locked" in [s.get("run") for s in steps()]


def test_it_skips_green_without_a_change_a_pin_or_a_key() -> None:
    later = steps()[index("Bundle key") + 1 :]

    assert "run=false" in step("Relevant changes")["run"]
    assert "run=false" in step("Pinned bundle")["run"]
    assert "run=false" in step("Bundle key")["run"]
    assert all("steps.key.outputs.run == 'true'" in s["if"] for s in later), [
        s.get("name") or s.get("uses") for s in later if "steps.key" not in s.get("if", "")
    ]


def test_main_keeps_the_baseline_and_a_pr_compares_with_it() -> None:
    upload = step("Keep main's baseline")
    baseline = step("Main's baseline")

    assert upload["uses"].startswith("actions/upload-artifact@")
    assert upload["with"]["name"] == "eval-baseline"
    assert upload["with"]["retention-days"] == 90
    assert "refs/heads/main" in upload["if"]
    assert "gh run list" in baseline["run"]
    assert "gh run download" in baseline["run"]
    assert "pull_request" in baseline["if"]
    assert "workflow_dispatch" in baseline["run"]  # a manual run on main refreshes it


def test_a_missing_baseline_fails_once_main_pins_a_bundle() -> None:
    """An expired or missing baseline must not turn the gate off (Final review I4)."""
    baseline = step("Main's baseline")

    assert baseline["env"]["BASE_REF"] == "${{ github.base_ref }}"
    assert 'git show "origin/$BASE_REF:evals/thresholds.yaml"' in baseline["run"]
    assert "::error::" in baseline["run"]
    assert "gh workflow run eval-gate.yml --ref main" in baseline["run"]
    assert "exit 1" in baseline["run"]


def test_the_comment_is_posted_before_the_check_fails() -> None:
    gate_step = step("Gate")
    fail = step("Fail on a breach")

    assert gate_step["continue-on-error"] is True
    assert index("Gate") < index("Comment on the PR") < index("Fail on a breach")
    assert "--edit-last" in step("Comment on the PR")["run"]
    assert "steps.gate.outcome == 'failure'" in fail["if"]
    assert "exit 1" in fail["run"]
