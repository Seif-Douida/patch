"""uptime.yml's checks (Phase 7 design D9): pp-api's /health, and how old the published data is.

The workflow calls /health (two tries of 60 s: a cold start takes ~23 s) and downloads the
published site's `/data/index.json`, what visitors see (so a failed publish counts too); no
Azure login, no SQL. This module turns both into the `api_down` and `data_stale` outcomes and
writes the issue plan the workflow applies. Numbers only.

    python -m patchpulse.monitoring.uptime --health ok|down --index index.json --out DIR
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from patchpulse.monitoring.alerts import (
    AlertRule,
    IssuePlan,
    Outcome,
    apply_outcomes,
    load_rules,
)

INDEX_PATH = "/data/index.json"  # under the site's address (publish-site.yml: public/data)
HEALTH_TRIES = 2
HEALTH_TIMEOUT_S = 60


def data_age_hours(index: Mapping[str, Any], now: datetime) -> float | None:
    """Hours since the published data was generated; None without a readable `generated_at`."""
    try:
        generated = datetime.fromisoformat(str(index["generated_at"]).replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return None
    return (now - generated.astimezone(UTC)).total_seconds() / 3600


def uptime_outcomes(
    *,
    health_ok: bool,
    age_hours: float | None,
    rules: Mapping[str, AlertRule],
    index_read: bool = True,
) -> list[Outcome]:
    """`index_read` False: the site's index couldn't be downloaded or parsed (the site is gone or
    a publish broke it), which is stale data too (final review I2)."""
    down = Outcome(
        "api_down",
        not health_ok,
        f"/health: {'answered' if health_ok else 'no answer in'} "
        f"{HEALTH_TRIES} tries of {HEALTH_TIMEOUT_S} s",
    )
    threshold = rules["data_stale"].threshold
    if not index_read:
        stale = Outcome("data_stale", True, f"{INDEX_PATH}: could not be read")
    elif age_hours is None or threshold is None:
        stale = Outcome("data_stale", None, "data age: not measured")
    else:
        stale = Outcome(
            "data_stale",
            age_hours > threshold,
            f"data generated {age_hours:.1f} h ago (threshold {threshold:.0f} h)",
        )
    return [down, stale]


def _read_index(path: Path) -> Mapping[str, Any] | None:
    """The downloaded index; None when it's missing or not a JSON object."""
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m patchpulse.monitoring.uptime")
    parser.add_argument("--health", choices=["ok", "down"], required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    rules = load_rules()
    index = _read_index(args.index)
    age = None if index is None else data_age_hours(index, datetime.now(UTC))
    outcomes = uptime_outcomes(
        health_ok=args.health == "ok", age_hours=age, rules=rules, index_read=index is not None
    )
    plan = IssuePlan()
    apply_outcomes(plan, rules, outcomes)
    plan.write(args.out)
    for outcome in outcomes:
        print(f"{outcome.rule_id}: {outcome.body}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
