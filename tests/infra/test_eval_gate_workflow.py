"""eval-gate.yml (Phase 6 plan Task 12, design D11): every PR replays the pinned eval bundle, with
no model, Azure or cluster; main's runs keep the baseline; the comment is posted before the check
fails."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from patchpulse.evals.bundle import THRESHOLDS_PATH
from patchpulse.evals.frozen import FROZEN_FILE, INJECTION_FILE, RUBRIC_FILE
from patchpulse.evals.incidents import INCIDENTS_DIR
from patchpulse.evals.retrieval_suite import QUERIES_PATH
from patchpulse.labeling.teachers import MODELS_FILE
from patchpulse.retrieval.judging import JUDGMENTS_PATH

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


def test_every_file_the_smoke_reads_counts_as_a_relevant_change() -> None:
    """A PR that changes only the smoke's inputs still runs the gate (Final review M2)."""
    relevant = step("Relevant changes")["run"]
    (pattern,) = re.findall(r"pattern='([^']+)'", relevant)
    read_by_the_smoke = [
        QUERIES_PATH,
        JUDGMENTS_PATH,
        THRESHOLDS_PATH,
        FROZEN_FILE,
        INJECTION_FILE,
        RUBRIC_FILE,
        INCIDENTS_DIR / "a-case.yaml",
        MODELS_FILE,
        Path("src/patchpulse/evals/smoke.py"),
        Path("pyproject.toml"),
        Path("uv.lock"),
        Path(".github/workflows/eval-gate.yml"),
    ]

    missed = [p.as_posix() for p in read_by_the_smoke if not re.search(pattern, p.as_posix())]

    assert missed == []


def test_a_failed_diff_fails_the_check_instead_of_skipping_it() -> None:
    """A git error must never read as "nothing relevant changed" (Final review M6)."""
    relevant = step("Relevant changes")["run"]

    assert not re.search(r"git diff[^\n]*\|", relevant)  # a pipe hides git's exit status
    assert re.search(r"if ! changed=\$\(git diff --name-only [^\n]*\); then", relevant)
    assert "::error::" in relevant
    assert "exit 1" in relevant


def test_the_pins_are_read_by_the_evals_cli() -> None:
    """Not by sed: a hand-edited pin is read or refused, never skipped (Final review M6)."""
    text = (WORKFLOWS / "eval-gate.yml").read_text(encoding="utf-8")
    sync = next(i for i, s in enumerate(steps()) if s.get("run") == "uv sync --locked")
    uv = next(i for i, s in enumerate(steps()) if s.get("uses", "").startswith("astral-sh/"))

    assert "sed " not in text
    assert "uv run --locked patchpulse-evals pin)" in step("Pinned bundle")["run"]
    assert (
        "patchpulse-evals pin --thresholds build/evals/base-thresholds.yaml"
        in (step("Main's baseline")["run"])
    )
    assert index("Relevant changes") < uv < sync < index("Pinned bundle")
    for setup in steps()[uv : sync + 1]:
        assert setup["if"] == "steps.relevant.outputs.run == 'true'", setup


def test_the_gate_judges_by_mains_tolerances() -> None:
    """The PR's own thresholds.yaml could loosen them (Final review M1)."""
    baseline = step("Main's baseline")["run"]
    gate_step = step("Gate")["run"]

    assert (
        'git show "origin/$BASE_REF:evals/thresholds.yaml" > build/evals/base-thresholds.yaml'
    ) in baseline
    assert 'git rev-parse --verify "origin/$BASE_REF"' in baseline  # a missing base fails
    assert "--thresholds build/evals/base-thresholds.yaml" in gate_step


def test_a_missing_key_fails_unless_the_pr_comes_from_a_fork() -> None:
    """Only a fork's PR can't read the secret. On main or a branch PR a missing key is a broken
    setup and must fail: on 16 Nov 2026 PR #38 and its merge both skipped green without it."""
    key = step("Bundle key")
    run = key["run"]

    assert key["env"]["FROM_FORK"] == (
        "${{ github.event_name == 'pull_request' && "
        "github.event.pull_request.head.repo.full_name != github.repository }}"
    )
    assert run.index('"$HAS_KEY" = "true"') < run.index('"$FROM_FORK" = "true"')
    assert run.index("run=false") < run.index("::error::") < run.index("exit 1")
    assert "repository secret" in run
