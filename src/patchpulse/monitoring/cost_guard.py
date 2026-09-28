"""Zero-cost guard (ADR-017): decide whether PatchPulse is about to spend money.

`.github/workflows/cost-guard.yml` saves three Azure responses as JSON files, then runs this module:

* a Cost Management query for rg-patchpulse's month-to-date actual cost (lags up to 72 hours),
* the pp-api `Replicas` metric, Total per hour, which is replica-minutes (near real time),
* the pp-api `Requests` metric, Total per hour.

The guard trips when any of them crosses its limit; the workflow then disables pp-api ingress and
opens a GitHub issue. This module only reads files and does arithmetic, so it is fully unit-tested.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

# Azure Container Apps Consumption free grant, per subscription per calendar month (spec §4).
FREE_VCPU_SECONDS = Decimal(180_000)
FREE_REQUESTS = Decimal(2_000_000)
# Trip at 80% of a grant: headroom for metric delay and for anything else sharing the grant.
GRANT_TRIP_FRACTION = Decimal("0.8")
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


def metric_total(metrics_response: Mapping[str, Any]) -> Decimal:
    """Sum every `total` data point in an `az monitor metrics list` response."""
    total = Decimal(0)
    for metric in metrics_response.get("value", []):
        for series in metric.get("timeseries", []):
            for point in series.get("data", []):
                if point.get("total") is not None:
                    total += Decimal(str(point["total"]))
    return total


@dataclass(frozen=True)
class GuardDecision:
    cost: Decimal | None  # None when the Cost Management query failed
    vcpu_seconds: Decimal
    requests: Decimal
    reasons: tuple[str, ...]

    @property
    def tripped(self) -> bool:
        return bool(self.reasons)


def evaluate(
    *,
    cost: Decimal | None,
    replica_minutes: Decimal,
    requests: Decimal,
    vcpu_per_replica: Decimal,
    cost_threshold: Decimal,
) -> GuardDecision:
    """Compare month-to-date usage with the $0 limits."""
    vcpu_seconds = replica_minutes * 60 * vcpu_per_replica
    vcpu_limit = FREE_VCPU_SECONDS * GRANT_TRIP_FRACTION
    request_limit = FREE_REQUESTS * GRANT_TRIP_FRACTION
    reasons: list[str] = []
    if cost is not None and cost > cost_threshold:
        reasons.append(f"month-to-date cost {cost} is above the {cost_threshold} threshold")
    if vcpu_seconds > vcpu_limit:
        reasons.append(
            f"pp-api used about {vcpu_seconds:.0f} vCPU-s, over {vcpu_limit:.0f} "
            "(80% of the free grant)"
        )
    if requests > request_limit:
        reasons.append(
            f"pp-api served {requests:.0f} requests, over {request_limit:.0f} "
            "(80% of the free grant)"
        )
    return GuardDecision(cost, vcpu_seconds, requests, tuple(reasons))


def render_summary(decision: GuardDecision) -> str:
    """Markdown for the job summary and the alert issue."""
    cost = "unavailable (query failed)" if decision.cost is None else str(decision.cost)
    lines = [
        "## Cost guard",
        "",
        f"- Month-to-date cost: {cost}",
        f"- pp-api vCPU-seconds this month (from replica-minutes): {decision.vcpu_seconds:.0f}",
        f"- pp-api requests this month: {decision.requests:.0f}",
        "",
    ]
    if decision.tripped:
        lines += ["**Tripped.** Reasons:", ""]
        lines += [f"- {reason}" for reason in decision.reasons]
        lines += [
            "",
            "Recovery: fix the cause, then re-run `infra.yml` to restore pp-api ingress. "
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
    parser.add_argument("--replicas", type=Path, required=True, help="Replicas metric response")
    parser.add_argument("--requests", type=Path, required=True, help="Requests metric response")
    parser.add_argument("--vcpu-per-replica", type=Decimal, default=Decimal("0.5"))
    parser.add_argument("--cost-threshold", type=Decimal, default=Decimal(0))
    parser.add_argument("--summary", type=Path, default=Path("cost-guard-summary.md"))
    args = parser.parse_args(argv)
    # The metrics step always writes both files, so a missing one is a bug, not zero usage.
    for metrics_file in (args.replicas, args.requests):
        if not metrics_file.exists():
            parser.error(f"metrics file not found: {metrics_file}")

    cost: Decimal | None = None
    cost_response = _load(args.cost)
    if cost_response is not None:
        try:
            cost = month_to_date_cost(cost_response)
        except (KeyError, TypeError, ValueError) as error:
            # An unreadable cost response must not stop the usage checks below.
            print(f"Cost Management response unreadable, treating cost as unavailable: {error!r}")
    decision = evaluate(
        cost=cost,
        replica_minutes=metric_total(_load(args.replicas) or {}),
        requests=metric_total(_load(args.requests) or {}),
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
