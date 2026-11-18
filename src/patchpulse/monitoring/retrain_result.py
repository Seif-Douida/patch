"""What retrain.yml reads back from pp-retrain (Phase 7 design D5-D6), on a GitHub runner.

    guard EXECUTIONS_JSON   `recent=<name>` if an execution of pp-retrain started in the last
                            12 hours (empty otherwise): retrain.yml refuses a second one.
    result --execution NAME --reason REASON --out DIR
                            Find the `retraining` run tagged with that execution (a public read
                            of DagsHub's MLflow, no token) and write DIR/champion.yaml (the pin a
                            promotion PR commits), DIR/reference.json (the challenger's drift
                            reference), DIR/body.md (the PR body) and DIR/table.md; print
                            `promote=`, `version=` and `registered=` lines for $GITHUB_OUTPUT.

Numbers only: the result, the reference and the body hold ids hashes, versions and scores.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from patchpulse.labeling.taxonomy import TAXONOMY_VERSION
from patchpulse.models.champion import ChampionPin, write_pin
from patchpulse.models.promotion import RetrainResult, parse_result

RETRAIN_EXPERIMENT = "retraining"  # models.retrain.RETRAIN_EXPERIMENT, without its trainer
GUARD_HOURS = 12
REASONS = ("drift", "monthly", "manual", "drill")
RESULT_SUFFIX = "-result.json"
REFERENCE_SUFFIX = "-reference.json"


def recent_execution(
    executions: Sequence[Mapping[str, Any]], *, now: datetime, hours: int = GUARD_HOURS
) -> str | None:
    """The newest execution that started within `hours` of `now`, if any."""
    since = now - timedelta(hours=hours)
    started = [
        (datetime.fromisoformat(item["properties"]["startTime"]), str(item["name"]))
        for item in executions
        if item.get("properties", {}).get("startTime")
    ]
    recent = [(when, name) for when, name in started if when >= since]
    return max(recent)[1] if recent else None


def _run_for(client: Any, execution: str) -> Any:
    experiment = client.get_experiment_by_name(RETRAIN_EXPERIMENT)
    runs = (
        []
        if experiment is None
        else client.search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.execution = '{execution}'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
    )
    if not runs:
        raise LookupError(f"no {RETRAIN_EXPERIMENT} run is tagged with execution {execution}")
    return runs[0]


def _artifact(client: Any, run_id: str, suffix: str, target: Path) -> Path:
    (name,) = [item.path for item in client.list_artifacts(run_id) if item.path.endswith(suffix)]
    return Path(client.download_artifacts(run_id, name, str(target)))


def _client(tracking_uri: str) -> Any:
    import mlflow  # the `retrain` group: only the result command needs it

    return mlflow.MlflowClient(tracking_uri=tracking_uri)


def read_result(tracking_uri: str, *, execution: str) -> RetrainResult:
    with tempfile.TemporaryDirectory() as scratch:
        return _download(tracking_uri, execution=execution, target=Path(scratch))[0]


def _download(tracking_uri: str, *, execution: str, target: Path) -> tuple[RetrainResult, Path]:
    """The execution's run (found by its tag): its result and its reference file."""
    client = _client(tracking_uri)
    run_id = _run_for(client, execution).info.run_id
    result_path = _artifact(client, run_id, RESULT_SUFFIX, target)
    reference = _artifact(client, run_id, REFERENCE_SUFFIX, target)
    return parse_result(result_path.read_text(encoding="utf-8")), reference


def pin_for_result(result: RetrainResult) -> ChampionPin:
    if result.registered_version is None:
        raise ValueError(f"challenger {result.version} was never registered: nothing to pin")
    return ChampionPin(
        version=result.registered_version,
        family="s0",  # pp-retrain retrains S0 only (design D5)
        model_version=result.version,
        model_json_sha256=result.model_json_sha256,
        taxonomy_version=TAXONOMY_VERSION,
    )


def _value(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def pr_body(result: RetrainResult, *, reason: str) -> str:
    if reason not in REASONS:
        raise ValueError(f"unknown retrain reason {reason!r}; expected one of {REASONS}")
    lines = [
        f"Promote aspect-classifier v{result.registered_version} ({reason})",
        "",
        f"pp-retrain trained S0 {result.version} and it passed every check of the promotion "
        "rule (config/retrain.yaml):",
        "",
        "| Check | Champion | Challenger | Limit | Passed |",
        "|---|---|---|---|---|",
    ]
    for check in result.decision.checks:
        passed = "yes" if check.passed else "no"
        lines.append(
            f"| {check.name} | {_value(check.champion)} | {_value(check.challenger)} "
            f"| {_value(check.limit)} | {passed} |"
        )
    lines += [
        "",
        "The full comparison, gold test included (reported only):",
        "",
        result.table.rstrip("\n"),
        "",
        f"MLflow run {result.mlflow_run_id}; model.json SHA-256 {result.model_json_sha256}.",
        "",
        "This PR changes models/champion.yaml and models/reference.json. Its eval-gate run scores "
        "the new champion on gold test against main, a second check.",
        "Merging it is the approval. Then the deploy bakes the model into the jobs image, the next "
        "nights rescore the reviews (a few nights for the whole corpus), and the index stage moves",
        "the changed aspects into Qdrant: a Qdrant out of sync alert may stay open meanwhile.",
    ]
    return "\n".join(lines) + "\n"


def _write_outputs(result: RetrainResult, reference: Path, *, reason: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(reference, out / "reference.json")
    if result.decision.promote:
        write_pin(pin_for_result(result), out / "champion.yaml")
        (out / "body.md").write_text(pr_body(result, reason=reason), encoding="utf-8")
    (out / "table.md").write_text(result.table, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m patchpulse.monitoring.retrain_result")
    commands = parser.add_subparsers(dest="command", required=True)
    guard = commands.add_parser("guard")
    guard.add_argument("executions", type=Path)
    read = commands.add_parser("result")
    read.add_argument("--execution", required=True)
    read.add_argument("--reason", required=True, choices=REASONS)
    read.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "guard":
        executions = json.loads(args.executions.read_text(encoding="utf-8"))
        print(f"recent={recent_execution(executions, now=datetime.now(UTC)) or ''}")
        return 0

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        print("retrain_result: MLFLOW_TRACKING_URI is not set", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as scratch:
        try:
            result, reference = _download(
                tracking_uri, execution=args.execution, target=Path(scratch)
            )
        except LookupError as error:
            print(f"retrain_result: {error}", file=sys.stderr)
            return 1
        _write_outputs(result, reference, reason=args.reason, out=args.out)
    print(f"promote={'true' if result.decision.promote else 'false'}")
    print(f"version={result.version}")
    print(f"registered={result.registered_version or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
