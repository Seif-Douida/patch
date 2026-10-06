"""The backstop for a night that failed without reporting it (ADR-018).

On 2026-10-06 a broken app key failed the night and also kept it from opening its own failure
issue, so the failure only showed in the Azure portal. cost-guard already reads pp-nightly's
executions every few hours; this module tells it whether the last finished one failed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from patchpulse.monitoring.nightly_watch import FAILURE_TITLE, last_failure, main
from patchpulse.publish.github import FAILURE_TITLE as PUBLISHER_FAILURE_TITLE

NIGHT = datetime(2026, 10, 6, 3, 0, tzinfo=UTC)
RUN = timedelta(minutes=32)


def execution(
    name: str, start: datetime, status: str, end: datetime | None = None
) -> dict[str, Any]:
    """One item of `az containerapp job execution list -o json`. Azure gives a failed execution
    no endTime."""
    properties: dict[str, Any] = {"status": status, "startTime": start.isoformat()}
    if end is not None:
        properties["endTime"] = end.isoformat()
    return {"name": name, "properties": properties}


def test_the_issue_is_the_one_the_next_successful_night_closes() -> None:
    assert FAILURE_TITLE == PUBLISHER_FAILURE_TITLE


@pytest.mark.parametrize("newest_first", [True, False])
def test_a_failed_last_night_is_reported(newest_first: bool) -> None:
    yesterday = NIGHT - timedelta(days=1)
    executions = [
        execution("pp-nightly-ok", yesterday, "Succeeded", yesterday + RUN),
        execution("pp-nightly-bad", NIGHT, "Failed"),
    ]
    if newest_first:  # the order az prints
        executions.reverse()

    failure = last_failure(executions)

    assert failure is not None
    assert failure["name"] == "pp-nightly-bad"


def test_a_success_after_a_failure_clears_it() -> None:
    executions = [
        execution("pp-nightly-ok", NIGHT, "Succeeded", NIGHT + RUN),
        execution("pp-nightly-bad", NIGHT - timedelta(days=1), "Failed"),
    ]

    assert last_failure(executions) is None


def test_a_run_in_progress_does_not_hide_the_last_failure() -> None:
    # A re-run started after a failure: until it succeeds, the failure stands.
    executions = [
        execution("pp-nightly-again", NIGHT + timedelta(hours=12), "Running"),
        execution("pp-nightly-bad", NIGHT, "Failed"),
    ]

    failure = last_failure(executions)

    assert failure is not None
    assert failure["name"] == "pp-nightly-bad"


@pytest.mark.parametrize("status", ["Stopped", "Degraded", "Unknown"])
def test_anything_but_success_counts_as_a_failure(status: str) -> None:
    assert last_failure([execution("pp-nightly-x", NIGHT, status)]) is not None


def test_no_executions_means_nothing_to_report() -> None:
    # pp-nightly deleted by a cost-guard trip, or not created yet.
    assert last_failure([]) is None


def test_main_writes_the_output_and_the_issue_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pp-nightly-executions.json"
    path.write_text(json.dumps({"value": [execution("pp-nightly-g0jkw7s", NIGHT, "Failed")]}))
    github_output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    body = tmp_path / "nightly-issue.md"

    assert main([str(path), "--body", str(body)]) == 0

    assert "failed=true" in github_output.read_text()
    text = body.read_text()
    assert "pp-nightly-g0jkw7s" in text
    assert "Failed" in text
    assert "2026-10-06 03:00 UTC" in text
    assert "closes this issue" in text


def test_main_reports_nothing_after_a_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pp-nightly-executions.json"
    path.write_text(json.dumps([execution("pp-nightly-ok", NIGHT, "Succeeded", NIGHT + RUN)]))
    github_output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    body = tmp_path / "nightly-issue.md"

    assert main([str(path), "--body", str(body)]) == 0

    assert "failed=false" in github_output.read_text()
    assert not body.exists()


def test_main_rejects_executions_it_cannot_read(tmp_path: Path) -> None:
    # An Azure error page read as "no executions" would hide a failure.
    path = tmp_path / "pp-nightly-executions.json"
    path.write_text(json.dumps({"error": {"code": "Throttled"}}))

    with pytest.raises(SystemExit):
        main([str(path), "--body", str(tmp_path / "nightly-issue.md")])
