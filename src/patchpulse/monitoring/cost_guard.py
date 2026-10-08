"""Zero-cost guard (ADR-017): decide whether PatchPulse is about to spend money.

`.github/workflows/cost-guard.yml` saves two Azure responses as JSON files, then runs this module:

* a Cost Management query for rg-patchpulse's month-to-date actual cost (lags up to 72 hours),
* the pp-api `Replicas` metric from the 1st of the month, Average per minute (near real time).
  Azure samples it about twice a minute, so the per-minute Average is the replica count and the
  sum of the Averages is replica-minutes; summing Totals would double-count.
* the executions of the Container Apps Jobs (pp-nightly, pp-migrate), which share the grant
  (ADR-009). Each execution runs one replica; its start and end times give its run time.

The guard trips on cost, on 80% of the monthly vCPU grant (pp-api and the jobs together), or on a
pp-api replica that ran for most of the last hour (a request flood or a stuck replica; Azure's
`Requests` metric has no data for pp-api, so it can't be counted directly). Every workload has
2 GiB per vCPU, so the memory grant (360,000 GiB-s) runs out exactly when the vCPU one does.
The workflow then stops pp-api, deletes pp-nightly and opens a GitHub issue. This module only
reads files and does arithmetic, so it is fully unit-tested.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

# Azure Container Apps Consumption free grant, per subscription per calendar month (spec §4).
FREE_VCPU_SECONDS = Decimal(180_000)
# Trip at 80% of a grant: headroom for metric delay and for anything else sharing the grant.
GRANT_TRIP_FRACTION = Decimal("0.8")
# A request wakes pp-api for about 6 minutes (the 300 s cool-down), so a replica that ran for
# 45 of the last 60 minutes means sustained traffic or a stuck replica.
BURN_WINDOW = timedelta(minutes=60)
BURN_TRIP_REPLICA_MINUTES = Decimal(45)
# Names Cost Management uses for the summed cost column, depending on API version and offer.
_COST_COLUMNS = ("totalCost", "Cost", "PreTaxCost")
# The most a job replica may use (tests/infra/test_zero_cost_guards.py); assumed when an
# execution doesn't say.
MAX_JOB_VCPU = Decimal(1)
# The longest any job may run (its replicaTimeout, capped by tests/infra/test_zero_cost_guards.py).
# Azure stops an execution there, so none can use more, even one listed without an endTime: a failed
# execution has none, and counting it "until now" tripped the guard on 2026-10-08.
MAX_JOB_SECONDS = Decimal(2700)


def month_to_date_cost(query_response: Mapping[str, Any]) -> Decimal:
    """Sum the cost column of a Cost Management query response. No rows means no cost yet."""
    properties = query_response["properties"]
    names = [column["name"] for column in properties["columns"]]
    index = next((names.index(name) for name in _COST_COLUMNS if name in names), None)
    if index is None:
        raise ValueError(f"no cost column in Cost Management response: {names}")
    return sum((Decimal(str(row[index])) for row in properties["rows"]), Decimal(0))


def count_replica_minutes(
    metrics_response: Mapping[str, Any], *, last: timedelta | None = None
) -> Decimal:
    """Sum the per-minute `average` points of a PT1M `Replicas` response: replica-minutes.

    With `last`, count only the minutes inside the final `last` of the response's timespan.
    """
    since: datetime | None = None
    if last is not None and "timespan" in metrics_response:
        end = datetime.fromisoformat(metrics_response["timespan"].split("/")[1])
        since = end - last
    minutes = Decimal(0)
    for metric in metrics_response.get("value", []):
        for series in metric.get("timeseries", []):
            for point in series.get("data", []):
                if point.get("average") is None:
                    continue
                if since is not None and datetime.fromisoformat(point["timeStamp"]) < since:
                    continue
                minutes += Decimal(str(point["average"]))
    return minutes


def job_vcpu_seconds(
    executions: Iterable[Mapping[str, Any]], *, month_start: datetime, now: datetime
) -> Decimal:
    """vCPU-seconds of job executions inside [month_start, now].

    One without an endTime (running, or failed: Azure records no end for those) counts until now,
    but no execution counts beyond its start + MAX_JOB_SECONDS, where Azure stops it.
    """
    total = Decimal(0)
    for item in executions:
        properties = item["properties"]
        started = datetime.fromisoformat(properties["startTime"])
        start = max(started, month_start)
        end_time = properties.get("endTime")
        latest_possible = started + timedelta(seconds=float(MAX_JOB_SECONDS))
        end = min(datetime.fromisoformat(end_time) if end_time else now, now, latest_possible)
        if end <= start:
            continue
        containers = properties.get("template", {}).get("containers", [])
        vcpu = sum(
            (
                Decimal(str(container["resources"]["cpu"]))
                if "cpu" in container.get("resources", {})
                else MAX_JOB_VCPU
                for container in containers
            ),
            Decimal(0),
        )
        total += Decimal(str((end - start).total_seconds())) * (vcpu or MAX_JOB_VCPU)
    return total


@dataclass(frozen=True)
class GuardDecision:
    cost: Decimal | None  # None when the Cost Management query failed
    vcpu_seconds: Decimal  # pp-api's
    recent_replica_minutes: Decimal
    reasons: tuple[str, ...]
    job_vcpu_seconds: Decimal = Decimal(0)

    @property
    def tripped(self) -> bool:
        return bool(self.reasons)


def evaluate(
    *,
    cost: Decimal | None,
    replica_minutes: Decimal,
    recent_replica_minutes: Decimal,
    vcpu_per_replica: Decimal,
    cost_threshold: Decimal,
    job_vcpu_seconds: Decimal = Decimal(0),
) -> GuardDecision:
    """Compare month-to-date usage and the last hour's usage with the $0 limits."""
    vcpu_seconds = replica_minutes * 60 * vcpu_per_replica
    vcpu_limit = FREE_VCPU_SECONDS * GRANT_TRIP_FRACTION
    reasons: list[str] = []
    if cost is not None and cost > cost_threshold:
        reasons.append(f"month-to-date cost {cost} is above the {cost_threshold} threshold")
    if vcpu_seconds + job_vcpu_seconds > vcpu_limit:
        reasons.append(
            f"pp-api and the jobs used about {vcpu_seconds + job_vcpu_seconds:.0f} vCPU-s "
            f"(pp-api {vcpu_seconds:.0f}, jobs {job_vcpu_seconds:.0f}), over {vcpu_limit:.0f} "
            "(80% of the free grant)"
        )
    if recent_replica_minutes >= BURN_TRIP_REPLICA_MINUTES:
        reasons.append(
            f"pp-api ran a replica for {recent_replica_minutes:.0f} of the last 60 minutes "
            f"(limit {BURN_TRIP_REPLICA_MINUTES}): sustained traffic or a stuck replica"
        )
    return GuardDecision(
        cost, vcpu_seconds, recent_replica_minutes, tuple(reasons), job_vcpu_seconds
    )


def render_summary(decision: GuardDecision) -> str:
    """Markdown for the job summary and the alert issue."""
    cost = "unavailable (query failed)" if decision.cost is None else str(decision.cost)
    lines = [
        "## Cost guard",
        "",
        f"- Month-to-date cost: {cost}",
        f"- pp-api vCPU-seconds this month (from replica-minutes): {decision.vcpu_seconds:.0f}",
        f"- Jobs' vCPU-seconds this month: {decision.job_vcpu_seconds:.0f}",
        f"- pp-api replica-minutes in the last 60 minutes: {decision.recent_replica_minutes:.0f}",
        "",
    ]
    if decision.tripped:
        lines += ["**Tripped.** Reasons:", ""]
        lines += [f"- {reason}" for reason in decision.reasons]
        lines += [
            "",
            "Recovery: fix the cause, then re-run `infra.yml` (what_if_only unticked), which "
            "restores pp-api ingress, starts the app again and recreates pp-nightly. "
            "After a vCPU trip, wait for the next month: pp-nightly's execution history is "
            "deleted with the job, so a recreated job's usage this month would be under-counted. "
            "A cost trip repeats every run until the month ends, because month-to-date cost "
            "stays above the threshold; raising the `COST_GUARD_THRESHOLD` variable is a "
            "deliberate decision to accept that cost.",
        ]
    else:
        lines.append("All limits OK.")
    return "\n".join(lines) + "\n"


def _load(path: Path | None) -> Mapping[str, Any] | None:
    if path is None or not path.exists():
        return None
    loaded: Mapping[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def read_executions(path: Path) -> list[Mapping[str, Any]]:
    """A job's executions: the CLI prints a list; the REST API wraps it as {"value": [...]}."""
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(loaded, dict):
        loaded = loaded["value"]
    if not isinstance(loaded, list):
        raise TypeError(f"expected a list of executions, got {type(loaded).__name__}")
    return loaded


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PatchPulse $0 guard (ADR-017).")
    parser.add_argument("--cost", type=Path, help="Cost Management query response; omit if failed")
    parser.add_argument("--replicas", type=Path, required=True, help="Replicas metric, 1m Average")
    parser.add_argument("--vcpu-per-replica", type=Decimal, default=Decimal("0.5"))
    parser.add_argument("--cost-threshold", type=Decimal, default=Decimal(0))
    parser.add_argument(
        "--job-executions",
        type=Path,
        action="append",
        default=[],
        help="`az containerapp job execution list` output; once per job",
    )
    parser.add_argument("--now", type=datetime.fromisoformat, help="for tests; default: now")
    parser.add_argument("--summary", type=Path, default=Path("cost-guard-summary.md"))
    args = parser.parse_args(argv)
    # The workflow always writes these files, so a missing one is a bug, not zero usage.
    for path in (args.replicas, *args.job_executions):
        if not path.exists():
            parser.error(f"usage file not found: {path}")
    now: datetime = args.now or datetime.now(UTC)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    jobs = Decimal(0)
    for path in args.job_executions:
        try:
            executions = read_executions(path)
            jobs += job_vcpu_seconds(executions, month_start=month_start, now=now)
        except (KeyError, TypeError, ValueError) as error:
            # Unlike cost, job usage has no other source: guessing zero would under-count.
            parser.error(f"unreadable job executions in {path}: {error!r}")

    cost: Decimal | None = None
    cost_response = _load(args.cost)
    if cost_response is not None:
        try:
            cost = month_to_date_cost(cost_response)
        except (KeyError, TypeError, ValueError) as error:
            # An unreadable cost response must not stop the usage checks below.
            print(f"Cost Management response unreadable, treating cost as unavailable: {error!r}")
    replicas = _load(args.replicas) or {}
    decision = evaluate(
        cost=cost,
        replica_minutes=count_replica_minutes(replicas),
        recent_replica_minutes=count_replica_minutes(replicas, last=BURN_WINDOW),
        vcpu_per_replica=args.vcpu_per_replica,
        cost_threshold=args.cost_threshold,
        job_vcpu_seconds=jobs,
    )
    summary = render_summary(decision)
    args.summary.write_text(summary, encoding="utf-8")
    print(summary)
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as output:
            output.write(f"tripped={str(decision.tripped).lower()}\n")
            output.write(f"cost_available={str(decision.cost is not None).lower()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
