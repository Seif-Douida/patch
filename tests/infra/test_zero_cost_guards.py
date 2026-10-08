"""$0 guardrails for the Bicep templates (ADR-017).

Compiles infra/main.bicep to ARM JSON and asserts every setting that keeps PatchPulse inside
Azure's permanent free tiers. Guard values must be literals in the Bicep modules, so a
parameterised value fails here instead of silently passing.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from patchpulse.monitoring.cost_guard import MAX_JOB_SECONDS

MAIN_BICEP = Path(__file__).resolve().parents[2] / "infra" / "main.bicep"

# Every resource type the templates may create. Adding a type is a deliberate, reviewed change.
ALLOWED_TYPES = frozenset(
    {
        "microsoft.resources/deployments",
        "microsoft.operationalinsights/workspaces",
        "microsoft.insights/components",
        "microsoft.insights/actiongroups",  # ADR-017 option B
        "microsoft.insights/metricalerts",  # ADR-017 option B
        "microsoft.app/managedenvironments",
        "microsoft.app/containerapps",
        "microsoft.app/jobs",  # ADR-009
        "microsoft.sql/servers",
        "microsoft.sql/servers/databases",
        "microsoft.sql/servers/firewallrules",
        "microsoft.web/staticsites",
    }
)

_JSON_LITERAL = re.compile(r"^\[json\('(-?[0-9.]+)'\)\]$")

# The Container Apps free grant, per subscription per month; pp-api and the jobs share it.
GRANT_VCPU_SECONDS = 180_000
GRANT_GIB_SECONDS = 360_000
# A fixed minute and hour, every day: one run a day, whatever the timeout math assumes.
_DAILY_CRON = re.compile(r"^[0-9]{1,2} [0-9]{1,2} \* \* \*$")
_TRIGGER_CONFIG = {
    "Schedule": "scheduleTriggerConfig",
    "Manual": "manualTriggerConfig",
    "Event": "eventTriggerConfig",
}


def bicep_build_command(main: Path) -> list[str] | None:
    """Command that compiles `main` to ARM JSON on stdout, or None when no compiler is installed."""
    bicep = shutil.which("bicep")
    if bicep:
        return [bicep, "build", str(main), "--stdout"]
    az = shutil.which("az")
    if az:
        return [az, "bicep", "build", "--file", str(main), "--stdout"]
    return None


def require_compiler(command: list[str] | None, *, ci: bool) -> list[str]:
    """Fail in CI (the guard must run there) but skip on a dev machine without Bicep."""
    if command is not None:
        return command
    if ci:
        pytest.fail("Bicep compiler not found in CI; the $0 guard must run.")
    pytest.skip("Bicep not installed locally (install Azure CLI, then run `az bicep install`).")


def literal_number(value: Any) -> float:
    """Read a number Bicep compiled either as a JSON number or as "[json('0.5')]"."""
    if isinstance(value, bool):
        raise AssertionError(f"expected a number, got {value!r}")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and (match := _JSON_LITERAL.match(value)):
        return float(match.group(1))
    raise AssertionError(f"guard value must be a literal in Bicep, got {value!r}")


def gibibytes(memory: Any) -> float:
    match = re.fullmatch(r"([0-9.]+)Gi", memory) if isinstance(memory, str) else None
    if match is None:
        raise AssertionError(f"memory must be a literal like '2Gi', got {memory!r}")
    return float(match.group(1))


def trigger_config(job: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = job["properties"]["configuration"]
    trigger: dict[str, Any] = config[_TRIGGER_CONFIG[config["triggerType"]]]
    return trigger


def iter_resources(template: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield every resource in an ARM template, descending into nested module deployments."""
    resources = template.get("resources", [])
    for resource in resources.values() if isinstance(resources, dict) else resources:
        yield resource
        nested = resource.get("properties", {}).get("template")
        if resource["type"].lower() == "microsoft.resources/deployments" and nested:
            yield from iter_resources(nested)


def of_type(resources: list[dict[str, Any]], resource_type: str) -> list[dict[str, Any]]:
    found = [r for r in resources if r["type"].lower() == resource_type]
    assert found, f"no {resource_type} in the template"
    return found


@pytest.fixture(scope="module")
def resources() -> list[dict[str, Any]]:
    command = require_compiler(bicep_build_command(MAIN_BICEP), ci=bool(os.environ.get("CI")))
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return list(iter_resources(json.loads(result.stdout)))


def test_only_allowlisted_resource_types(resources: list[dict[str, Any]]) -> None:
    unexpected = {r["type"] for r in resources if r["type"].lower() not in ALLOWED_TYPES}
    assert not unexpected, f"new resource types need an ADR and an allowlist entry: {unexpected}"


def test_sql_database_is_free_offer_that_pauses_when_exhausted(
    resources: list[dict[str, Any]],
) -> None:
    (database,) = of_type(resources, "microsoft.sql/servers/databases")
    props = database["properties"]
    assert props["useFreeLimit"] is True
    assert props["freeLimitExhaustionBehavior"] == "AutoPause"
    assert database["sku"]["name"] == "GP_S_Gen5"
    assert literal_number(database["sku"]["capacity"]) <= 2
    assert literal_number(props["minCapacity"]) == 0.5
    # Azure only allows the default delay (60 min) on a free database with AutoPause
    # (ProvisioningDisabled on 2026-09-28 with 15). Each wake costs >= 60 x 0.5 = 1,800 vCore-s.
    assert literal_number(props["autoPauseDelay"]) == 60
    assert literal_number(props["maxSizeBytes"]) <= 32 * 1024**3
    assert props["requestedBackupStorageRedundancy"] == "Local"
    assert props["zoneRedundant"] is False


def test_sql_firewall_rules_open_single_addresses_only(resources: list[dict[str, Any]]) -> None:
    for rule in of_type(resources, "microsoft.sql/servers/firewallrules"):
        props = rule["properties"]
        assert props["startIpAddress"] == props["endIpAddress"], rule["name"]


def test_log_analytics_daily_cap_fits_free_ingestion(resources: list[dict[str, Any]]) -> None:
    (workspace,) = of_type(resources, "microsoft.operationalinsights/workspaces")
    props = workspace["properties"]
    # A commitment tier (CapacityReservation) bills its reservation whatever the daily cap says.
    assert props["sku"]["name"] == "PerGB2018"
    assert "capacityReservationLevel" not in props["sku"]
    cap_gb = literal_number(props["workspaceCapping"]["dailyQuotaGb"])
    assert 0 < cap_gb <= 0.15
    assert cap_gb * 31 < 5  # 5 GB free ingestion per billing account per month
    assert literal_number(props["retentionInDays"]) <= 31


def test_container_apps_environment_is_consumption_only(resources: list[dict[str, Any]]) -> None:
    (environment,) = of_type(resources, "microsoft.app/managedenvironments")
    profiles = environment["properties"]["workloadProfiles"]
    assert [p["workloadProfileType"] for p in profiles] == ["Consumption"]
    assert environment["properties"]["zoneRedundant"] is False


def test_container_apps_scale_to_zero_with_capped_size(resources: list[dict[str, Any]]) -> None:
    for app in of_type(resources, "microsoft.app/containerapps"):
        template = app["properties"]["template"]
        assert literal_number(template["scale"]["minReplicas"]) == 0
        assert literal_number(template["scale"]["maxReplicas"]) <= 1
        for container in template["containers"]:
            assert literal_number(container["resources"]["cpu"]) <= 0.5
            assert container["resources"]["memory"] == "1Gi"


def test_jobs_have_capped_size_timeout_and_no_retries(resources: list[dict[str, Any]]) -> None:
    for job in of_type(resources, "microsoft.app/jobs"):
        config = job["properties"]["configuration"]
        # The cost guard caps every execution at this (cost_guard.MAX_JOB_SECONDS): keep them equal.
        assert literal_number(config["replicaTimeout"]) <= MAX_JOB_SECONDS, job["name"]
        # A retry doubles a failed night's usage; the next night catches up instead (ADR-009).
        assert literal_number(config["replicaRetryLimit"]) == 0, job["name"]
        assert literal_number(trigger_config(job)["parallelism"]) == 1, job["name"]
        assert literal_number(trigger_config(job)["replicaCompletionCount"]) == 1, job["name"]
        for container in job["properties"]["template"]["containers"]:
            assert literal_number(container["resources"]["cpu"]) <= 1, job["name"]
            assert gibibytes(container["resources"]["memory"]) <= 2, job["name"]


def test_jobs_run_only_on_a_schedule_or_by_hand(resources: list[dict[str, Any]]) -> None:
    # An event trigger scales out with its queue; nothing would bound how often it runs.
    for job in of_type(resources, "microsoft.app/jobs"):
        config = job["properties"]["configuration"]
        assert config["triggerType"] in {"Schedule", "Manual"}, job["name"]
        assert "eventTriggerConfig" not in config, job["name"]


def test_scheduled_jobs_run_once_a_day(resources: list[dict[str, Any]]) -> None:
    scheduled = [
        job
        for job in of_type(resources, "microsoft.app/jobs")
        if job["properties"]["configuration"]["triggerType"] == "Schedule"
    ]
    assert scheduled, "expected the nightly job"
    for job in scheduled:
        cron = trigger_config(job)["cronExpression"]
        assert _DAILY_CRON.match(cron), f"{job['name']}: {cron!r} is not a literal daily schedule"


def test_worst_case_scheduled_job_usage_is_under_half_the_grant(
    resources: list[dict[str, Any]],
) -> None:
    # Every scheduled run hits its timeout, every day of a 31-day month (one run a day, above).
    vcpu_seconds = gib_seconds = 0.0
    for job in of_type(resources, "microsoft.app/jobs"):
        config = job["properties"]["configuration"]
        if config["triggerType"] != "Schedule":
            continue
        month = 31 * literal_number(config["replicaTimeout"])
        for container in job["properties"]["template"]["containers"]:
            vcpu_seconds += month * literal_number(container["resources"]["cpu"])
            gib_seconds += month * gibibytes(container["resources"]["memory"])

    assert vcpu_seconds <= GRANT_VCPU_SECONDS / 2
    assert gib_seconds <= GRANT_GIB_SECONDS / 2


def test_job_credentials_come_from_container_app_secrets(resources: list[dict[str, Any]]) -> None:
    for job in of_type(resources, "microsoft.app/jobs"):
        for container in job["properties"]["template"]["containers"]:
            for variable in container.get("env", []):
                if re.search(r"PASSWORD|SALT|PRIVATE_KEY", variable["name"]):
                    assert "secretRef" in variable, f"{job['name']}: {variable['name']}"
                    assert "value" not in variable, f"{job['name']}: {variable['name']}"


def test_nightly_job_publishes_with_the_github_app(resources: list[dict[str, Any]]) -> None:
    # ADR-018: the key reaches the job only as a Container Apps secret; --publish makes a missing
    # setting fail the night instead of skipping the publish.
    (nightly,) = [j for j in of_type(resources, "microsoft.app/jobs") if j["name"] == "pp-nightly"]
    (container,) = nightly["properties"]["template"]["containers"]
    assert container["args"] == ["nightly", "--publish"]
    env = {variable["name"]: variable for variable in container["env"]}
    assert "secretRef" in env["PP_GITHUB_APP_PRIVATE_KEY"]
    assert env["PP_GITHUB_REPOSITORY"]["value"] == "Seif-Douida/patch"
    assert "PP_GITHUB_APP_ID" in env


def test_metric_alerts_stay_inside_the_free_time_series(resources: list[dict[str, Any]]) -> None:
    # The first 10 monitored metric time series a month are free. One resource, one metric and no
    # dimension split is one time series; a dimension would multiply them.
    series = 0
    for alert in of_type(resources, "microsoft.insights/metricalerts"):
        properties = alert["properties"]
        assert len(properties["scopes"]) == 1, alert["name"]
        criteria = properties["criteria"]["allOf"]
        for criterion in criteria:
            assert not criterion.get("dimensions"), alert["name"]
        series += len(criteria)
    assert series <= 10


def test_action_groups_only_send_email(resources: list[dict[str, Any]]) -> None:
    # SMS and voice calls are billed from the first one; email is free up to 1,000 a month.
    for group in of_type(resources, "microsoft.insights/actiongroups"):
        receivers = {
            name
            for name, value in group["properties"].items()
            if name.endswith("Receivers") and value
        }
        assert receivers == {"emailReceivers"}, group["name"]


def test_backup_alert_matches_the_cost_guard_burn_rate_rule(
    resources: list[dict[str, Any]],
) -> None:
    # cost-guard runs only every few hours (measured 2026-10-05), so Azure itself watches the same
    # rule and emails: a pp-api replica up for 45 of the last 60 minutes (max 1 replica).
    (alert,) = of_type(resources, "microsoft.insights/metricalerts")
    properties = alert["properties"]
    (criterion,) = properties["criteria"]["allOf"]
    assert criterion["metricName"] == "Replicas"
    assert criterion["timeAggregation"] == "Average"
    assert criterion["operator"] == "GreaterThanOrEqual"
    assert literal_number(criterion["threshold"]) == 0.75
    assert properties["windowSize"] == "PT1H"
    assert properties["enabled"] is True
    assert properties["actions"], "the alert must notify someone"


def test_static_web_app_is_free_plan(resources: list[dict[str, Any]]) -> None:
    (site,) = of_type(resources, "microsoft.web/staticsites")
    assert site["sku"]["name"] == "Free"


def test_literal_number_reads_bicep_decimal_expressions() -> None:
    assert literal_number("[json('0.15')]") == 0.15
    assert literal_number(2) == 2.0


def test_gibibytes_reads_literal_memory_only() -> None:
    assert gibibytes("2Gi") == 2.0
    assert gibibytes("0.5Gi") == 0.5
    with pytest.raises(AssertionError, match="literal"):
        gibibytes("[parameters('memory')]")


def test_literal_number_rejects_parameterised_guard_values() -> None:
    with pytest.raises(AssertionError, match="literal"):
        literal_number("[parameters('minReplicas')]")


def test_missing_compiler_fails_in_ci_and_skips_locally() -> None:
    with pytest.raises(pytest.fail.Exception):
        require_compiler(None, ci=True)
    with pytest.raises(pytest.skip.Exception):
        require_compiler(None, ci=False)
