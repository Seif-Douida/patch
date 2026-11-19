"""Phase 7 Task 14: uptime.yml (design D9) - every 6 hours, /health and the published data's age,
issues through GITHUB_TOKEN, no Azure login and no SQL."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from image_groups import run_with_only

from patchpulse.monitoring.uptime import INDEX_PATH

ROOT = Path(__file__).resolve().parents[2]
TEXT = (ROOT / ".github" / "workflows" / "uptime.yml").read_text(encoding="utf-8")
SPEC: dict[Any, Any] = yaml.safe_load(TEXT)
TRIGGERS: dict[str, Any] = SPEC["on"] if "on" in SPEC else SPEC[True]  # PyYAML: `on` is True
(JOB,) = SPEC["jobs"].values()
STEPS: list[dict[str, Any]] = JOB["steps"]
SCRIPT = ROOT / ".github" / "scripts" / "apply-issue-plan.sh"


def step(name: str) -> dict[str, Any]:
    (found,) = [s for s in STEPS if s.get("name", "").startswith(name)]
    return found


def test_uptime_never_logs_in_to_azure_and_waits_sixty_seconds() -> None:
    assert TRIGGERS["schedule"] == [{"cron": "17 */6 * * *"}]
    assert set(TRIGGERS) == {"schedule", "workflow_dispatch"}
    assert SPEC["permissions"] == {"contents": "read", "issues": "write"}
    assert "azure/login" not in TEXT
    assert "az " not in TEXT
    health = step("Call /health")
    assert health["env"]["API_URL"] == "${{ vars.PP_API_URL }}"
    assert "--max-time 60" in health["run"]
    assert re.search(r"for try in 1 2; do", health["run"])
    index = step("Read the published data's index")
    assert index["env"]["SITE_URL"] == "${{ vars.PP_SITE_URL }}"
    assert f'"$SITE_URL{INDEX_PATH}"' in index["run"]


def test_the_addresses_are_trimmed_before_use() -> None:
    # 19 Nov: /health failed in under a second and pp-api saw no request; re-pasting the same
    # address fixed it (most likely a stray space or line break, which curl refuses). A trailing
    # slash would double the one in front of the path.
    for name, variable in (
        ("Call /health", "API_URL"),
        ("Read the published data's index", "SITE_URL"),
    ):
        run = step(name)["run"]
        trim = f"{variable}=$(printf '%s' \"${variable}\" | tr -d '[:space:]')"
        slash = f'{variable}="${{{variable}%/}}"'
        assert trim in run
        assert slash in run
        assert run.index(trim) < run.index(slash) < run.index(f'[ -z "${variable}" ]')


def test_the_issues_go_through_the_shared_plan_script() -> None:
    apply = step("Open or close the issues")
    assert apply["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert ".github/scripts/apply-issue-plan.sh uptime-plan" in apply["run"]
    script = SCRIPT.read_text(encoding="utf-8")
    assert "gh issue create" in script
    assert "gh issue close" in script
    # An issue already open is left as it is: cost-guard runs every 15 minutes.
    assert "already open" in script


def test_the_checks_import_with_only_the_pipeline_group() -> None:
    # The workflows run them with `uv run --no-dev --group pipeline` (PyYAML for the rules).
    assert "uv run --no-dev --group pipeline python -m patchpulse.monitoring.uptime" in TEXT
    result = run_with_only(
        {"pipeline"},
        "import patchpulse.monitoring.uptime, patchpulse.monitoring.sql_allowance\n"
        "print('imported')\n",
    )

    assert result.returncode == 0, result.stderr[-1500:]
    assert result.stdout.strip() == "imported"
