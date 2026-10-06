"""Backstop for a night that failed without reporting it (ADR-018).

pp-nightly opens its own "pipeline failed: nightly" issue, but it can only do that while it can
reach GitHub: with a broken app key (2026-10-06) the failure showed only in the Azure portal.
cost-guard.yml already saves pp-nightly's executions every few hours, so it runs this module too.
If the last finished execution failed, it writes the issue body and `failed=true`; the workflow
then opens the issue unless one is already open. The next successful night closes it, as it closes
the ones the job opens itself.

Like cost_guard, this module only reads a file, so it is fully unit-tested.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from patchpulse.monitoring.cost_guard import read_executions

# The publisher's title (patchpulse.publish.github, which needs the pipeline's dependencies);
# a test keeps the two equal.
FAILURE_TITLE = "pipeline failed: nightly"
# Still going, so not a result yet.
_IN_PROGRESS = {"Running", "Processing"}


def last_failure(executions: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The latest finished execution, if it didn't succeed. Azure gives a failed one no endTime,
    so executions are ordered by start time and "finished" means no longer running."""
    finished = [item for item in executions if item["properties"]["status"] not in _IN_PROGRESS]
    if not finished:
        return None
    latest = max(finished, key=lambda item: datetime.fromisoformat(item["properties"]["startTime"]))
    return None if latest["properties"]["status"] == "Succeeded" else latest


def issue_body(failure: Mapping[str, Any]) -> str:
    properties = failure["properties"]
    started = datetime.fromisoformat(properties["startTime"]).astimezone(UTC)
    return (
        f"`pp-nightly` execution `{failure['name']}` (started {started:%Y-%m-%d %H:%M} UTC) ended "
        f"with status **{properties['status']}**, and no `{FAILURE_TITLE}` issue was open.\n\n"
        "The run reports its own failures, so it probably couldn't reach GitHub: for example a "
        "broken `PP_GITHUB_APP_PRIVATE_KEY`, or a timeout before it could report.\n\n"
        "Logs: Log Analytics, table `ContainerAppConsoleLogs_CL`, `ContainerGroupName_s` starting "
        f"with `{failure['name']}`.\n\n"
        "The next successful night closes this issue. Opened by cost-guard.\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report pp-nightly's last failure (ADR-018).")
    parser.add_argument(
        "executions", type=Path, help="`az containerapp job execution list` output for pp-nightly"
    )
    parser.add_argument("--body", type=Path, default=Path("nightly-issue.md"))
    args = parser.parse_args(argv)
    try:
        failure = last_failure(read_executions(args.executions))
    except (KeyError, TypeError, ValueError) as error:
        # Reading an Azure error as "no executions" would hide a failure.
        parser.error(f"unreadable executions in {args.executions}: {error!r}")
    if failure is None:
        print("pp-nightly: the last finished execution succeeded (or there is none).")
    else:
        args.body.write_text(issue_body(failure), encoding="utf-8")
        print(f"pp-nightly: {failure['name']} ended {failure['properties']['status']}.")
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as output:
            output.write(f"failed={str(failure is not None).lower()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
