"""Tests for the $0 guard's parsing and trip logic (ADR-017)."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from patchpulse.monitoring.cost_guard import (
    GuardDecision,
    evaluate,
    main,
    metric_total,
    month_to_date_cost,
    render_summary,
)


def cost_response(column: str, rows: list[list[Any]]) -> dict[str, Any]:
    columns = [{"name": column, "type": "Number"}, {"name": "Currency", "type": "String"}]
    return {"properties": {"columns": columns, "rows": rows}}


def metrics_response(*totals: float | None) -> dict[str, Any]:
    data: list[dict[str, Any]] = []
    for hour, total in enumerate(totals):
        point: dict[str, Any] = {"timeStamp": f"2026-09-01T{hour:02d}:00:00Z"}
        if total is not None:
            point["total"] = total
        data.append(point)
    return {"value": [{"name": {"value": "Replicas"}, "timeseries": [{"data": data}]}]}


def decide(**overrides: Any) -> GuardDecision:
    arguments: dict[str, Any] = {
        "cost": Decimal(0),
        "replica_minutes": Decimal(0),
        "requests": Decimal(0),
        "vcpu_per_replica": Decimal("0.5"),
        "cost_threshold": Decimal(0),
    }
    arguments.update(overrides)
    return evaluate(**arguments)


def test_month_to_date_cost_sums_unrounded_rows() -> None:
    response = cost_response("Cost", [[0.004, "GBP"], [0.004, "GBP"]])
    assert month_to_date_cost(response) == Decimal("0.008")


def test_month_to_date_cost_is_zero_without_rows() -> None:
    assert month_to_date_cost(cost_response("Cost", [])) == Decimal(0)


@pytest.mark.parametrize("column", ["totalCost", "Cost", "PreTaxCost"])
def test_month_to_date_cost_accepts_known_column_names(column: str) -> None:
    assert month_to_date_cost(cost_response(column, [[1.5, "EUR"]])) == Decimal("1.5")


def test_month_to_date_cost_rejects_unknown_shape() -> None:
    with pytest.raises(ValueError, match="no cost column"):
        month_to_date_cost(cost_response("Amount", [[1, "EUR"]]))


def test_metric_total_ignores_missing_points() -> None:
    assert metric_total(metrics_response(1, None, 2.5)) == Decimal("3.5")


def test_metric_total_of_empty_response_is_zero() -> None:
    assert metric_total({"value": []}) == Decimal(0)


def test_quiet_month_does_not_trip() -> None:
    decision = decide(replica_minutes=Decimal(300), requests=Decimal(5_000))
    assert not decision.tripped
    assert decision.vcpu_seconds == Decimal(9_000)  # 300 min x 60 s x 0.5 vCPU


def test_any_cost_above_threshold_trips() -> None:
    decision = decide(cost=Decimal("0.01"))
    assert decision.tripped
    assert "month-to-date cost" in decision.reasons[0]


def test_vcpu_limit_trips_just_above_80_percent_of_grant() -> None:
    # 144,000 vCPU-s = 80% of 180,000; at 0.5 vCPU that is 4,800 replica-minutes.
    assert not decide(replica_minutes=Decimal(4_800)).tripped
    assert decide(replica_minutes=Decimal(4_801)).tripped


def test_request_limit_trips_just_above_80_percent_of_grant() -> None:
    assert not decide(requests=Decimal(1_600_000)).tripped
    assert decide(requests=Decimal(1_600_001)).tripped


def test_unavailable_cost_does_not_trip_on_its_own() -> None:
    decision = decide(cost=None)
    assert not decision.tripped
    assert decision.cost is None


def test_summary_reports_reasons_without_claiming_an_action() -> None:
    # The workflow reports what it actually did (disabled, dry run, failed); the summary must not.
    summary = render_summary(decide(cost=Decimal("0.01")))
    assert "month-to-date cost" in summary
    assert "ingress is disabled" not in summary
    assert "COST_GUARD_THRESHOLD" in summary  # a cost trip re-trips until it is raised


def test_main_treats_unreadable_cost_response_as_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "cost.json").write_text(json.dumps({"error": {"code": "429"}}))
    (tmp_path / "replicas.json").write_text(json.dumps(metrics_response(4_801)))
    (tmp_path / "requests.json").write_text(json.dumps({"value": []}))
    github_output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))

    exit_code = main(
        [
            "--cost", str(tmp_path / "cost.json"),
            "--replicas", str(tmp_path / "replicas.json"),
            "--requests", str(tmp_path / "requests.json"),
            "--summary", str(tmp_path / "summary.md"),
        ]
    )  # fmt: skip

    assert exit_code == 0
    assert "cost_available=false" in github_output.read_text()
    assert "tripped=true" in github_output.read_text()  # the usage check still ran


def test_main_rejects_a_missing_metrics_file(tmp_path: Path) -> None:
    missing = str(tmp_path / "missing.json")
    with pytest.raises(SystemExit):
        main(["--replicas", missing, "--requests", missing, "--summary", str(tmp_path / "s.md")])


def test_main_writes_outputs_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "cost.json").write_text(json.dumps(cost_response("Cost", [[0.02, "GBP"]])))
    (tmp_path / "replicas.json").write_text(json.dumps(metrics_response(10)))
    (tmp_path / "requests.json").write_text(json.dumps(metrics_response(50)))
    github_output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    summary = tmp_path / "summary.md"

    exit_code = main(
        [
            "--cost", str(tmp_path / "cost.json"),
            "--replicas", str(tmp_path / "replicas.json"),
            "--requests", str(tmp_path / "requests.json"),
            "--summary", str(summary),
        ]
    )  # fmt: skip

    assert exit_code == 0
    assert "tripped=true" in github_output.read_text()
    assert "cost_available=true" in github_output.read_text()
    assert "month-to-date cost 0.02" in summary.read_text()


def test_main_without_cost_file_reports_cost_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "empty.json").write_text(json.dumps({"value": []}))
    github_output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    summary = tmp_path / "summary.md"

    main(
        [
            "--replicas", str(tmp_path / "empty.json"),
            "--requests", str(tmp_path / "empty.json"),
            "--summary", str(summary),
        ]
    )  # fmt: skip

    assert "tripped=false" in github_output.read_text()
    assert "cost_available=false" in github_output.read_text()
    assert "unavailable (query failed)" in summary.read_text()
