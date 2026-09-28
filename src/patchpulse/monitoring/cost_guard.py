"""Zero-cost guard (ADR-017): decide whether PatchPulse is about to spend money.

`.github/workflows/cost-guard.yml` saves two Azure responses as JSON files, then runs this module:

* a Cost Management query for rg-patchpulse's month-to-date actual cost (lags up to 72 hours),
* the pp-api `Replicas` metric from the 1st of the month, Average per minute (near real time).
  Azure samples it about twice a minute, so the per-minute Average is the replica count and the
  sum of the Averages is replica-minutes; summing Totals would double-count.

The guard trips on cost, on 80% of the monthly vCPU grant, or on a replica that ran for most of
the last hour (a request flood or a stuck replica; Azure's `Requests` metric has no data for
pp-api, so it can't be counted directly). The workflow then stops pp-api and opens a GitHub
issue. This module only reads files and does arithmetic, so it is fully unit-tested.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
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


@dataclass(frozen=True)
class GuardDecision:
    cost: Decimal | None  # None when the Cost Management query failed
    vcpu_seconds: Decimal
    recent_replica_minutes: Decimal
    reasons: tuple[str, ...]

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
) -> GuardDecision:
    """Compare month-to-date usage and the last hour's usage with the $0 limits."""
    vcpu_seconds = replica_minutes * 60 * vcpu_per_replica
    vcpu_limit = FREE_VCPU_SECONDS * GRANT_TRIP_FRACTION
    reasons: list[str] = []
    if cost is not None and cost > cost_threshold:
        reasons.append(f"month-to-date cost {cost} is above the {cost_threshold} threshold")
    if vcpu_seconds > vcpu_limit:
        reasons.append(
            f"pp-api used about {vcpu_seconds:.0f} vCPU-s, over {vcpu_limit:.0f} "
            "(80% of the free grant)"
        )
    if recent_replica_minutes >= BURN_TRIP_REPLICA_MINUTES:
        reasons.append(
            f"pp-api ran a replica for {recent_replica_minutes:.0f} of the last 60 minutes "
            f"(limit {BURN_TRIP_REPLICA_MINUTES}): sustained traffic or a stuck replica"
        )
    return GuardDecision(cost, vcpu_seconds, recent_replica_minutes, tuple(reasons))


def render_summary(decision: GuardDecision) -> str:
    """Markdown for the job summary and the alert issue."""
    cost = "unavailable (query failed)" if decision.cost is None else str(decision.cost)
    lines = [
        "## Cost guard",
        "",
        f"- Month-to-date cost: {cost}",
        f"- pp-api vCPU-seconds this month (from replica-minutes): {decision.vcpu_seconds:.0f}",
        f"- pp-api replica-minutes in the last 60 minutes: {decision.recent_replica_minutes:.0f}",
        "",
    ]
    if decision.tripped:
        lines += ["**Tripped.** Reasons:", ""]
        lines += [f"- {reason}" for reason in decision.reasons]
        lines += [
            "",
            "Recovery: fix the cause, then re-run `infra.yml` (what_if_only unticked), which "
            "restores pp-api ingress and starts the app again. "
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PatchPulse $0 guard (ADR-017).")
    parser.add_argument("--cost", type=Path, help="Cost Management query response; omit if failed")
    parser.add_argument("--replicas", type=Path, required=True, help="Replicas metric, 1m Average")
    parser.add_argument("--vcpu-per-replica", type=Decimal, default=Decimal("0.5"))
    parser.add_argument("--cost-threshold", type=Decimal, default=Decimal(0))
    parser.add_argument("--summary", type=Path, default=Path("cost-guard-summary.md"))
    args = parser.parse_args(argv)
    # The metrics step always writes the file, so a missing one is a bug, not zero usage.
    if not args.replicas.exists():
        parser.error(f"metrics file not found: {args.replicas}")

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
