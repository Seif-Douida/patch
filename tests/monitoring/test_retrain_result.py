"""Phase 7 Task 12: what retrain.yml reads back from pp-retrain's MLflow run (design D5-D6)."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import mlflow
import pytest
import yaml

from patchpulse.labeling.taxonomy import TAXONOMY_VERSION
from patchpulse.models.champion import ChampionPin
from patchpulse.models.promotion import Check, Decision, RetrainResult, result_json
from patchpulse.monitoring.retrain_result import (
    main,
    pin_for_result,
    pr_body,
    read_result,
    recent_execution,
)

NOW = datetime(2026, 12, 1, 9, 0, tzinfo=UTC)
SHA = "ab" * 32


def execution(name: str, started: datetime) -> dict[str, object]:
    return {"name": name, "properties": {"startTime": started.isoformat(), "status": "Succeeded"}}


def result(
    *, version: str = "20261201T030000Z", registered: int | None = 4, gain: float = 0.034
) -> RetrainResult:
    checks = [
        Check("recent_macro_f1", 0.61, 0.61 + gain, 0.62, True, f"gain {gain:+.3f}, n 812"),
        Check("recent_by_language", 0.58, 0.60, 0.56, True, "weakest: french (+0.020, n 140)"),
        Check("gold_dev_macro_f1", 0.65, 0.66, 0.63, True, "out of fold"),
        Check("reviews_per_second", None, 2400.0, 70.0, True, "one thread"),
    ]
    table = f"| recent_macro_f1 | 0.610 | {0.61 + gain:.3f} | 0.620 | yes | gain {gain:+.3f} |\n"
    return RetrainResult(version, Decision(True, checks), "run123", registered, SHA, table)


def test_a_second_request_within_twelve_hours_is_refused() -> None:
    recent = [execution("pp-retrain-old", NOW - timedelta(hours=30)),
              execution("pp-retrain-a1", NOW - timedelta(hours=11))]  # fmt: skip

    assert recent_execution(recent, now=NOW) == "pp-retrain-a1"
    assert recent_execution(recent[:1], now=NOW) is None
    assert (
        recent_execution([execution("pp-retrain-b2", NOW - timedelta(hours=13))], now=NOW) is None
    )
    assert recent_execution([{"name": "pending", "properties": {}}], now=NOW) is None


def test_the_pin_matches_the_registered_version_and_its_hash() -> None:
    pin = pin_for_result(result())

    assert pin == ChampionPin(
        version=4,
        family="s0",
        model_version="20261201T030000Z",
        model_json_sha256=SHA,
        taxonomy_version=TAXONOMY_VERSION,
    )
    with pytest.raises(ValueError, match="registered"):
        pin_for_result(result(registered=None))


def test_the_pr_body_holds_no_text() -> None:
    # Numbers only: two results that differ in every number give the same body once the numbers
    # are masked, so nothing but the template and the numbers can reach it.
    first = pr_body(result(), reason="drift")
    second = pr_body(result(version="20270104T030000Z", registered=9, gain=0.051), reason="drift")

    def mask(body: str) -> str:
        return re.sub(r"[0-9][0-9.]*", "#", body)

    assert first != second
    assert mask(first) == mask(second)
    assert "Promote aspect-classifier v4 (drift)" in first
    assert "eval-gate" in first
    assert "Qdrant out of sync" in first
    assert "recent_macro_f1" in first
    assert all(len(line) <= 200 for line in first.splitlines())


def test_an_unknown_reason_is_refused() -> None:
    with pytest.raises(ValueError, match="reason"):
        pr_body(result(), reason="because the issue said so")


@pytest.fixture
def logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A retraining run as pp-retrain logs it, in a local MLflow store; returns its URI."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("retraining")
    reports = tmp_path / "reports" / "retrain"
    reports.mkdir(parents=True)
    retrained = result()
    (reports / f"{retrained.version}-result.json").write_text(result_json(retrained))
    (reports / f"{retrained.version}-reference.json").write_text('{"model_version": "x"}\n')
    with mlflow.start_run(tags={"execution": "pp-retrain-x7k2p", "decision": "promote"}):
        mlflow.log_artifact(str(reports / f"{retrained.version}-result.json"))
        mlflow.log_artifact(str(reports / f"{retrained.version}-reference.json"))
    with mlflow.start_run(tags={"execution": "pp-retrain-other", "decision": "keep"}):
        pass
    return uri


def test_the_result_is_read_by_the_executions_tag(logged: str) -> None:
    assert read_result(logged, execution="pp-retrain-x7k2p") == result()
    with pytest.raises(LookupError, match="pp-retrain-none"):
        read_result(logged, execution="pp-retrain-none")


def test_main_writes_the_pin_the_reference_and_the_body(
    logged: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", logged)
    out = tmp_path / "out"

    code = main(
        ["result", "--execution", "pp-retrain-x7k2p", "--reason", "monthly", "--out", str(out)]
    )

    assert code == 0
    pin = ChampionPin.model_validate(yaml.safe_load((out / "champion.yaml").read_text()))
    assert pin.version == 4
    assert json.loads((out / "reference.json").read_text()) == {"model_version": "x"}
    assert "(monthly)" in (out / "body.md").read_text()
    lines = capsys.readouterr().out.splitlines()
    assert {"promote=true", "version=20261201T030000Z", "registered=4"} <= set(lines)


def test_the_guard_command_prints_the_recent_execution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    executions = tmp_path / "executions.json"
    started = datetime.now(UTC) - timedelta(hours=2)
    executions.write_text(json.dumps([execution("pp-retrain-a1", started)]))

    assert main(["guard", str(executions)]) == 0
    assert capsys.readouterr().out.strip() == "recent=pp-retrain-a1"
