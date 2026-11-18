"""Phase 7 Task 14: uptime.yml's checks (design D9) - pp-api's /health and the published data's
age - and the issue plan the workflows apply."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from patchpulse.monitoring.alerts import Outcome, load_rules
from patchpulse.monitoring.uptime import data_age_hours, main, uptime_outcomes

RULES = load_rules()
NOW = datetime(2026, 11, 18, 6, 17, tzinfo=UTC)


def by_rule(outcomes: list[Outcome]) -> dict[str, bool | None]:
    return {outcome.rule_id: outcome.breached for outcome in outcomes}


def test_a_healthy_api_and_fresh_data_resolve_both() -> None:
    outcomes = uptime_outcomes(health_ok=True, age_hours=27.5, rules=RULES)

    assert by_rule(outcomes) == {"api_down": False, "data_stale": False}


def test_two_failed_tries_raise_api_down() -> None:
    outcomes = uptime_outcomes(health_ok=False, age_hours=27.5, rules=RULES)

    assert by_rule(outcomes) == {"api_down": True, "data_stale": False}
    (down,) = [o for o in outcomes if o.rule_id == "api_down"]
    assert "2 tries of 60 s" in down.body


def test_data_older_than_sixty_hours_is_stale() -> None:
    index = {"data_version": 61, "generated_at": "2026-11-15T18:00:00Z"}

    age = data_age_hours(index, NOW)

    assert age is not None
    assert round(age, 2) == 60.28
    assert by_rule(uptime_outcomes(health_ok=True, age_hours=age, rules=RULES))["data_stale"]
    assert not by_rule(uptime_outcomes(health_ok=True, age_hours=59.9, rules=RULES))["data_stale"]
    # No readable timestamp is "not measured": the issue is left as it is.
    assert data_age_hours({"data_version": 61}, NOW) is None
    assert (
        by_rule(uptime_outcomes(health_ok=True, age_hours=None, rules=RULES))["data_stale"] is None
    )


def test_main_writes_the_issue_plan(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"generated_at": "2026-11-10T03:00:00Z"}), encoding="utf-8")
    out = tmp_path / "plan"

    assert main(["--health", "down", "--index", str(index), "--out", str(out)]) == 0

    rows = [line.split("\t") for line in (out / "issues.tsv").read_text().splitlines()]
    assert [(action, label, title) for action, label, title, _ in rows] == [
        ("alert", "alert", "pp-api down"),
        ("alert", "alert", "data stale"),
    ]
    for *_, body in rows:
        assert (out / body).read_text(encoding="utf-8").strip()


def test_an_unreadable_site_raises_data_stale(tmp_path: Path) -> None:
    # Final review I2 (18 Nov): a site that can't be read (gone, a broken publish) is the
    # stalest data of all; it used to count as "not measured" and never alerted.
    out = tmp_path / "plan"

    assert (
        main(["--health", "ok", "--index", str(tmp_path / "missing.json"), "--out", str(out)]) == 0
    )

    rows = [line.split("\t") for line in (out / "issues.tsv").read_text().splitlines()]
    assert [(action, title) for action, _, title, _ in rows] == [
        ("resolve", "pp-api down"),
        ("alert", "data stale"),
    ]


def test_an_index_without_its_date_leaves_data_stale_alone(tmp_path: Path) -> None:
    out = tmp_path / "plan"
    index = tmp_path / "index.json"
    index.write_text('{"games": []}', encoding="utf-8")

    assert main(["--health", "ok", "--index", str(index), "--out", str(out)]) == 0

    rows = [line.split("\t") for line in (out / "issues.tsv").read_text().splitlines()]
    assert [(action, title) for action, _, title, _ in rows] == [("resolve", "pp-api down")]
