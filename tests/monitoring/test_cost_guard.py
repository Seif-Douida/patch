"""Tests for the $0 guard's parsing and trip logic (ADR-017)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from patchpulse.monitoring.cost_guard import (
    GuardDecision,
    count_replica_minutes,
    evaluate,
    job_vcpu_seconds,
    main,
    month_to_date_cost,
    render_summary,
)

START = datetime(2026, 9, 1, tzinfo=UTC)
NOW = datetime(2026, 9, 29, 12, tzinfo=UTC)


def execution(
    start: datetime, end: datetime | None, *, cpu: float | None = 1.0, status: str = "Succeeded"
) -> dict[str, Any]:
    """One item of `az containerapp job execution list -o json`."""
    container: dict[str, Any] = {"name": "nightly", "image": "ghcr.io/x/patchpulse-jobs:abc"}
    if cpu is not None:
        container["resources"] = {"cpu": cpu, "memory": f"{cpu * 2}Gi"}
    properties: dict[str, Any] = {
        "status": status,
        "startTime": start.isoformat(),
        "template": {"containers": [container]},
    }
    if end is not None:
        properties["endTime"] = end.isoformat()
    return {"name": "pp-nightly-abc12", "properties": properties}


def cost_response(column: str, rows: list[list[Any]]) -> dict[str, Any]:
    columns = [{"name": column, "type": "Number"}, {"name": "Currency", "type": "String"}]
    return {"properties": {"columns": columns, "rows": rows}}


def azure_time(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def replicas_response(*averages: float | None) -> dict[str, Any]:
    """An `az monitor metrics list` response for Replicas at PT1M, one point per minute from START.

    Azure samples Replicas about twice a minute, so each point's `total` is twice its `average`.
    """
    data: list[dict[str, Any]] = []
    for minute, average in enumerate(averages):
        point: dict[str, Any] = {"timeStamp": azure_time(START + timedelta(minutes=minute))}
        if average is not None:
            point["average"] = average
            point["total"] = average * 2
        data.append(point)
    end = START + timedelta(minutes=len(averages))
    return {
        "interval": "PT1M",
        "timespan": f"{azure_time(START)}/{azure_time(end)}",
        "value": [{"name": {"value": "Replicas"}, "timeseries": [{"data": data}]}],
    }


def decide(**overrides: Any) -> GuardDecision:
    arguments: dict[str, Any] = {
        "cost": Decimal(0),
        "replica_minutes": Decimal(0),
        "recent_replica_minutes": Decimal(0),
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


def test_replica_minutes_sum_per_minute_averages_not_sample_totals() -> None:
    # Calibration 2026-09-28: summing Totals double-counted, because of the two samples a minute.
    assert count_replica_minutes(replicas_response(1, 1, 0.5, 0, None)) == Decimal("2.5")


def test_replica_minutes_of_empty_response_is_zero() -> None:
    assert count_replica_minutes({"value": []}) == Decimal(0)
    assert count_replica_minutes({"value": []}, last=timedelta(minutes=60)) == Decimal(0)


def test_replica_minutes_in_the_last_window_count_only_its_minutes() -> None:
    response = replicas_response(*[1.0] * 30, *[0.5] * 60)
    assert count_replica_minutes(response) == Decimal(60)
    assert count_replica_minutes(response, last=timedelta(minutes=60)) == Decimal(30)


def test_job_seconds_count_only_this_month() -> None:
    executions = [
        execution(datetime(2026, 8, 30, 3, tzinfo=UTC), datetime(2026, 8, 30, 3, 40, tzinfo=UTC)),
        # Started 10 minutes before the month, ended 20 minutes into it: 20 minutes count.
        execution(START - timedelta(minutes=10), START + timedelta(minutes=20)),
        execution(datetime(2026, 9, 2, 3, tzinfo=UTC), datetime(2026, 9, 2, 3, 30, tzinfo=UTC)),
    ]

    seconds = job_vcpu_seconds(executions, month_start=START, now=NOW)

    assert seconds == Decimal((20 + 30) * 60)  # x 1 vCPU


def test_running_execution_counts_until_now() -> None:
    running = execution(NOW - timedelta(minutes=12), None, status="Running")

    assert job_vcpu_seconds([running], month_start=START, now=NOW) == Decimal(12 * 60)


def test_job_vcpu_comes_from_the_execution_or_the_one_vcpu_cap() -> None:
    migrate = execution(NOW - timedelta(minutes=10), NOW, cpu=0.5)
    unknown = execution(NOW - timedelta(minutes=10), NOW, cpu=None)

    assert job_vcpu_seconds([migrate], month_start=START, now=NOW) == Decimal(300)
    # Without a size, assume the most a job may use (tests/infra guard: 1 vCPU).
    assert job_vcpu_seconds([unknown], month_start=START, now=NOW) == Decimal(600)


def test_jobs_and_api_share_the_monthly_limit() -> None:
    # 4,000 replica-minutes of pp-api = 120,000 vCPU-s; the jobs' 30,000 push it over 144,000.
    assert not decide(replica_minutes=Decimal(4_000)).tripped
    decision = decide(replica_minutes=Decimal(4_000), job_vcpu_seconds=Decimal(30_000))
    assert decision.tripped
    assert "jobs" in decision.reasons[0]


def test_quiet_month_does_not_trip() -> None:
    decision = decide(replica_minutes=Decimal(300), recent_replica_minutes=Decimal(6))
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


def test_replica_running_most_of_the_last_hour_trips() -> None:
    # A request flood or a stuck replica keeps pp-api up; normal use wakes it for ~6 minutes.
    assert not decide(recent_replica_minutes=Decimal("44.5")).tripped
    decision = decide(recent_replica_minutes=Decimal(45))
    assert decision.tripped
    assert "last 60 minutes" in decision.reasons[0]


def test_unavailable_cost_does_not_trip_on_its_own() -> None:
    decision = decide(cost=None)
    assert not decision.tripped
    assert decision.cost is None


def test_summary_reports_reasons_without_claiming_an_action() -> None:
    # The workflow reports what it actually did (stopped, dry run, failed); the summary must not.
    summary = render_summary(decide(cost=Decimal("0.01")))
    assert "month-to-date cost" in summary
    assert "is stopped" not in summary
    assert "COST_GUARD_THRESHOLD" in summary  # a cost trip re-trips until it is raised


def test_main_treats_unreadable_cost_response_as_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "cost.json").write_text(json.dumps({"error": {"code": "429"}}))
    (tmp_path / "replicas.json").write_text(json.dumps(replicas_response(*[1.0] * 60)))
    github_output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))

    exit_code = main(
        [
            "--cost", str(tmp_path / "cost.json"),
            "--replicas", str(tmp_path / "replicas.json"),
            "--summary", str(tmp_path / "summary.md"),
        ]
    )  # fmt: skip

    assert exit_code == 0
    assert "cost_available=false" in github_output.read_text()
    assert "tripped=true" in github_output.read_text()  # the usage check still ran


def test_main_rejects_a_missing_metrics_file(tmp_path: Path) -> None:
    missing = str(tmp_path / "missing.json")
    with pytest.raises(SystemExit):
        main(["--replicas", missing, "--summary", str(tmp_path / "s.md")])


def test_main_writes_outputs_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "cost.json").write_text(json.dumps(cost_response("Cost", [[0.02, "GBP"]])))
    (tmp_path / "replicas.json").write_text(json.dumps(replicas_response(*[1.0] * 10)))
    github_output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    summary = tmp_path / "summary.md"

    exit_code = main(
        [
            "--cost", str(tmp_path / "cost.json"),
            "--replicas", str(tmp_path / "replicas.json"),
            "--summary", str(summary),
        ]
    )  # fmt: skip

    assert exit_code == 0
    assert "tripped=true" in github_output.read_text()
    assert "cost_available=true" in github_output.read_text()
    assert "month-to-date cost 0.02" in summary.read_text()


def test_main_trips_when_a_replica_ran_most_of_the_last_hour(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 50 replica-minutes is 1,500 vCPU-s: far below the monthly limit, but a stuck replica.
    (tmp_path / "replicas.json").write_text(json.dumps(replicas_response(*[1.0] * 50)))
    github_output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    summary = tmp_path / "summary.md"

    main(["--replicas", str(tmp_path / "replicas.json"), "--summary", str(summary)])

    assert "tripped=true" in github_output.read_text()
    assert "replica-minutes in the last 60 minutes: 50" in summary.read_text()


def test_main_adds_the_jobs_executions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "replicas.json").write_text(json.dumps(replicas_response(*[1.0] * 10)))
    nightly = [execution(NOW - timedelta(minutes=40), NOW - timedelta(minutes=5))]
    (tmp_path / "pp-nightly.json").write_text(json.dumps(nightly))
    (tmp_path / "pp-migrate.json").write_text(json.dumps([]))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "github_output"))
    summary = tmp_path / "summary.md"

    main(
        [
            "--replicas", str(tmp_path / "replicas.json"),
            "--job-executions", str(tmp_path / "pp-nightly.json"),
            "--job-executions", str(tmp_path / "pp-migrate.json"),
            "--now", NOW.isoformat(),
            "--summary", str(summary),
        ]
    )  # fmt: skip

    assert "Jobs' vCPU-seconds this month: 2100" in summary.read_text()  # 35 min x 1 vCPU


@pytest.mark.parametrize("wrap", [False, True])
def test_main_reads_executions_as_a_list_or_a_value_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wrap: bool
) -> None:
    (tmp_path / "replicas.json").write_text(json.dumps({"value": []}))
    items = [execution(NOW - timedelta(minutes=10), NOW)]
    (tmp_path / "jobs.json").write_text(json.dumps({"value": items} if wrap else items))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "github_output"))
    summary = tmp_path / "summary.md"

    main(
        [
            "--replicas", str(tmp_path / "replicas.json"),
            "--job-executions", str(tmp_path / "jobs.json"),
            "--now", NOW.isoformat(),
            "--summary", str(summary),
        ]
    )  # fmt: skip

    assert "Jobs' vCPU-seconds this month: 600" in summary.read_text()


def test_main_rejects_executions_it_cannot_read(tmp_path: Path) -> None:
    # Reading an unexpected shape as zero usage would under-count the month.
    (tmp_path / "replicas.json").write_text(json.dumps({"value": []}))
    (tmp_path / "jobs.json").write_text(json.dumps({"error": {"code": "Throttled"}}))
    with pytest.raises(SystemExit):
        main(
            [
                "--replicas", str(tmp_path / "replicas.json"),
                "--job-executions", str(tmp_path / "jobs.json"),
                "--summary", str(tmp_path / "s.md"),
            ]
        )  # fmt: skip


def test_main_rejects_a_missing_executions_file(tmp_path: Path) -> None:
    (tmp_path / "empty.json").write_text(json.dumps({"value": []}))
    with pytest.raises(SystemExit):
        main(
            [
                "--replicas", str(tmp_path / "empty.json"),
                "--job-executions", str(tmp_path / "missing.json"),
                "--summary", str(tmp_path / "s.md"),
            ]
        )  # fmt: skip


def test_main_without_cost_file_reports_cost_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "empty.json").write_text(json.dumps({"value": []}))
    github_output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    summary = tmp_path / "summary.md"

    main(["--replicas", str(tmp_path / "empty.json"), "--summary", str(summary)])

    assert "tripped=false" in github_output.read_text()
    assert "cost_available=false" in github_output.read_text()
    assert "unavailable (query failed)" in summary.read_text()
