"""The workflows that make up the kill switch (ADR-017).

The kill switch can only switch off pp-api, so its cost check is scoped to the rg-patchpulse
resource group. Spending elsewhere in the subscription is the $1 budget's job; counting it here
disabled pp-api over another project's NAT gateway on 2026-09-28.

The calibration and drill of 2026-09-28 pinned the rest: the Replicas metric must be read as
per-minute averages, and a trip must stop pp-api, because disabling ingress alone left a replica
running for 4.5 hours.
"""

import re
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def read(workflow: str) -> str:
    return (WORKFLOWS / workflow).read_text(encoding="utf-8")


def shell_lines(workflow: str) -> list[str]:
    """The workflow's lines, with shell backslash continuations joined."""
    return read(workflow).replace("\\\n", " ").splitlines()


def test_cost_query_is_scoped_to_the_patchpulse_resource_group() -> None:
    text = read("cost-guard.yml")
    (url,) = re.findall(r"https://management\.azure\.com/\S*CostManagement/query\S*", text)
    assert "/resourceGroups/$RG/" in url, url
    assert re.search(r"^\s+RG: rg-patchpulse$", text, flags=re.MULTILINE)


def test_guard_is_scheduled_every_15_minutes_off_the_hour() -> None:
    # GitHub ran 2 of ~19 hourly schedules on 2026-09-28: scheduled runs may be dropped under
    # load, and the start of the hour is the busiest time.
    (minutes,) = re.findall(r'^\s+- cron: "(\S+) \* \* \* \*"$', read("cost-guard.yml"), re.M)
    slots = sorted(int(minute) for minute in minutes.split(","))
    assert 0 not in slots
    next_slots = [*slots[1:], slots[0] + 60]  # the last slot wraps to the next hour's first
    gaps = [later - earlier for earlier, later in zip(slots, next_slots, strict=True)]
    assert max(gaps) <= 15, slots


def test_alert_issue_is_not_recommented_while_pp_api_stays_stopped() -> None:
    # Every 15 minutes a persisting trip would otherwise add ~100 comments a day.
    text = read("cost-guard.yml")
    assert "already_stopped=" in text
    issue_step = text.split("- name: Open or update the alert issue", 1)[1]
    assert "steps.trip.outputs.already_stopped" in issue_step


def test_cost_query_is_retried_when_throttled() -> None:
    # Cost Management answered 429 "Too many requests" on 2026-09-28, which blinded the cost check.
    text = read("cost-guard.yml")
    step = text.split("id: cost\n", 1)[1].split("\n      - name:", 1)[0]
    assert "CostManagement/query" in step
    assert re.search(r"for attempt in [\d ]+; do", step)
    assert "sleep" in step


def test_replicas_are_read_as_per_minute_averages() -> None:
    # Azure samples Replicas about twice a minute, so hourly Totals double-count.
    (query,) = [line for line in shell_lines("cost-guard.yml") if "az monitor metrics list" in line]
    assert "--metrics Replicas" in query
    assert "--aggregation Average" in query
    assert "--interval 1m" in query


def test_trip_stops_pp_api_after_disabling_ingress_and_verifies_it() -> None:
    text = read("cost-guard.yml")
    (stop,) = [line for line in shell_lines("cost-guard.yml") if "/stop?api-version=" in line]
    assert "az rest --method post" in stop
    # Disabling ingress restarts the app, so the stop must come after it.
    assert text.index("az containerapp ingress disable") < text.index("/stop?api-version=")
    assert "properties.runningStatus" in text
    assert "properties.replicas" in text


def test_infra_deploy_restarts_a_stopped_pp_api() -> None:
    text = read("infra.yml")
    (start,) = [line for line in shell_lines("infra.yml") if "/start?api-version=" in line]
    assert "az rest --method post" in start
    assert text.index("az deployment group create") < text.index("/start?api-version=")


def test_infra_and_api_image_deploys_never_overlap() -> None:
    # infra.yml reads the image pp-api is running, then deploys it again a few minutes later; an
    # api-image roll in between would be silently undone.
    for workflow in ("infra.yml", "api-image.yml"):
        deploy_job = read(workflow).split("\n  deploy:\n", 1)[1]
        assert re.search(r"^    concurrency:\n      group: pp-api-deploy\n", deploy_job, re.M), (
            workflow
        )


def test_api_image_refuses_to_update_a_stopped_pp_api() -> None:
    # Recovery is infra's job; updating the image of a stopped app could restart it unguarded.
    text = read("api-image.yml")
    assert text.index("properties.runningStatus") < text.index("az containerapp update")
