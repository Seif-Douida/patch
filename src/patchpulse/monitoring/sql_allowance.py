"""The free SQL offer's monthly allowance (Phase 7 design D7, rule `sql_allowance`).

The free offer gives 100,000 vCore-seconds a month; when they run out the database pauses until
the 1st (ADR-017: it never bills). cost-guard.yml reads the database's `free_amount_consumed`
metric from Azure Monitor, which never wakes or connects to the database, and runs this module.
The rule is breached when less than 20% is left, or, from the 3rd of the month (two days say
little), when the month's pace would use more than the allowance. Numbers only.

    python -m patchpulse.monitoring.sql_allowance METRICS_JSON --out DIR
"""

from __future__ import annotations

import argparse
import calendar
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from patchpulse.monitoring.alerts import AlertRule, IssuePlan, Outcome, apply_outcomes, load_rules

SQL_ALLOWANCE = 100_000  # vCore-seconds a month
PACE_FROM_DAY = 3


def used_this_month(metrics: Mapping[str, Any]) -> float | None:
    """The largest `maximum` of the month's series (the counter only grows within a month)."""
    values = [
        float(point["maximum"])
        for metric in metrics.get("value", [])
        for series in metric.get("timeseries", [])
        for point in series.get("data", [])
        if point.get("maximum") is not None
    ]
    return max(values) if values else None


def _month_share(now: datetime) -> float:
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    days = calendar.monthrange(now.year, now.month)[1]
    return (now - start).total_seconds() / (days * 86_400)


def sql_outcome(*, used: float, now: datetime, rule: AlertRule) -> Outcome:
    threshold = rule.threshold if rule.threshold is not None else 0.2
    left = 1 - used / SQL_ALLOWANCE
    body = f"used {used:,.0f} of {SQL_ALLOWANCE:,} vCore-s ({left:.0%} left), day {now.day}"
    if left < threshold:
        return Outcome(rule.id, True, f"{body}; under {threshold:.0%} left")
    if now.day < PACE_FROM_DAY:
        return Outcome(rule.id, False, body)
    projected = used / _month_share(now)
    body = f"{body}; pace {projected:,.0f} for the month"
    return Outcome(rule.id, projected > SQL_ALLOWANCE, body)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m patchpulse.monitoring.sql_allowance")
    parser.add_argument("metrics", type=Path, help="`az monitor metrics list` output")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    rules = load_rules()
    rule = rules["sql_allowance"]
    used = used_this_month(json.loads(args.metrics.read_text(encoding="utf-8")))
    if used is None:
        outcome = Outcome(rule.id, None, "free_amount_consumed: not measured")
    else:
        outcome = sql_outcome(used=used, now=datetime.now(UTC), rule=rule)
    plan = IssuePlan()
    apply_outcomes(plan, rules, [outcome])
    plan.write(args.out)
    print(f"{outcome.rule_id}: {outcome.body}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
