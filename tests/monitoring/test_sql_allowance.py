"""Phase 7 Task 14: the free SQL offer's monthly allowance (design D7's `sql_allowance`), read by
cost-guard from the database's metric, never from the database."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from patchpulse.monitoring.alerts import load_rules
from patchpulse.monitoring.sql_allowance import SQL_ALLOWANCE, main, sql_outcome, used_this_month

RULE = load_rules()["sql_allowance"]


def at(day: int, hour: int = 0) -> datetime:
    return datetime(2026, 11, day, hour, tzinfo=UTC)  # November: 30 days


def test_twenty_percent_left_breaches_whatever_the_day() -> None:
    assert SQL_ALLOWANCE == 100_000
    assert sql_outcome(used=81_000, now=at(28), rule=RULE).breached
    assert sql_outcome(used=81_000, now=at(1, 12), rule=RULE).breached
    assert sql_outcome(used=79_000, now=at(30, 12), rule=RULE).breached is False


def test_the_pace_rule_waits_for_the_third_and_projects_the_month() -> None:
    # Day 2: 20,000 in a day and a half would be 400,000 for the month, but two days say little.
    assert sql_outcome(used=20_000, now=at(2, 12), rule=RULE).breached is False
    # Day 3, noon: 2.5 of 30 days gone. 10,000 projects to 120,000; 8,000 to 96,000.
    fast = sql_outcome(used=10_000, now=at(3, 12), rule=RULE)
    assert fast.breached
    assert "120,000" in fast.body
    assert sql_outcome(used=8_000, now=at(3, 12), rule=RULE).breached is False


def test_the_months_use_is_the_metrics_largest_value(tmp_path: Path) -> None:
    metrics = {
        "value": [
            {
                "name": {"value": "free_amount_consumed"},
                "timeseries": [
                    {"data": [{"maximum": 1200.0}, {"maximum": None}, {"maximum": 4100.5}]}
                ],
            }
        ]
    }

    assert used_this_month(metrics) == 4100.5
    assert used_this_month({"value": []}) is None

    path = tmp_path / "sql.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    out = tmp_path / "plan"
    assert main([str(path), "--out", str(out)]) == 0
    (row,) = [line.split("\t") for line in (out / "issues.tsv").read_text().splitlines()]
    assert row[:3] == ["resolve", "alert", "SQL allowance low"]
