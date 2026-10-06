"""The workflows that make up the kill switch (ADR-017).

The kill switch can only switch off pp-api, so its cost check is scoped to the rg-patchpulse
resource group. Spending elsewhere in the subscription is the $1 budget's job; counting it here
disabled pp-api over another project's NAT gateway on 2026-09-28.

The calibration and drill of 2026-09-28 pinned the rest: the Replicas metric must be read as
per-minute averages, and a trip must stop pp-api, because disabling ingress alone left a replica
running for 4.5 hours.

deploy.yml (Phase 1, replacing api-image.yml) must never undo a trip, and must migrate the
database before the code that needs the new schema runs.
"""

import re
from pathlib import Path

from patchpulse.publish.github import FAILURE_TITLE

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def read(workflow: str) -> str:
    return (WORKFLOWS / workflow).read_text(encoding="utf-8")


def shell_lines(workflow: str) -> list[str]:
    """The workflow's lines, with shell backslash continuations joined."""
    return read(workflow).replace("\\\n", " ").splitlines()


def step(workflow: str, name: str) -> str:
    """The text of the step called `name`, up to the next step."""
    return read(workflow).split(f"- name: {name}", 1)[1].split("\n      - name:", 1)[0]


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


def test_infra_and_deploy_share_the_pp_api_deploy_group() -> None:
    # infra.yml reads the images pp-api and the jobs are running, then deploys them again a few
    # minutes later; a deploy.yml roll in between would be silently undone.
    for workflow in ("infra.yml", "deploy.yml"):
        deploy_job = read(workflow).split("\n  deploy:\n", 1)[1]
        assert re.search(r"^    concurrency:\n      group: pp-api-deploy\n", deploy_job, re.M), (
            workflow
        )


def test_deploy_refuses_a_stopped_pp_api() -> None:
    # Recovery is infra's job; updating the image of a stopped app could restart it unguarded.
    text = read("deploy.yml")
    assert text.index("properties.runningStatus") < text.index("az containerapp update")
    assert text.index("properties.runningStatus") < text.index("az containerapp job start")


def test_api_image_workflow_is_replaced_by_deploy() -> None:
    # Two workflows rolling pp-api would race each other.
    assert not (WORKFLOWS / "api-image.yml").exists()


def test_deploy_builds_both_images_for_this_commit() -> None:
    text = read("deploy.yml")
    assert "Dockerfile.jobs" in text
    assert "patchpulse-jobs" in text
    assert "patchpulse-api" in text
    assert "type=sha,format=long,prefix=" in text


def test_deploy_runs_migrations_before_updating_the_api() -> None:
    text = read("deploy.yml")
    migrate = step("deploy.yml", "Run migrations")
    assert 'az containerapp job update -g "$RG" -n pp-migrate' in migrate
    assert 'az containerapp job start -g "$RG" -n pp-migrate' in migrate
    assert "Succeeded) exit 0" in migrate
    assert re.search(r"Failed\|", migrate)
    assert text.index("- name: Run migrations") < text.index("az containerapp update")


def test_deploy_rolls_back_the_api_on_a_failed_smoke_test() -> None:
    rollback = step("deploy.yml", "Roll back pp-api")
    assert "failure()" in rollback
    assert "steps.smoke.outcome == 'failure'" in rollback
    assert "steps.api.outputs.previous_image" in rollback
    assert "az containerapp update" in rollback


def test_nightly_job_gets_the_new_image_only_after_the_api_is_healthy() -> None:
    text = read("deploy.yml")
    assert text.index("- name: Smoke test") < text.index("-n pp-nightly --image")


def test_infra_keeps_the_running_jobs_image() -> None:
    text = read("infra.yml")
    assert "az containerapp job show" in text
    assert "JOBS_IMAGE=" in text
    assert "patchpulse-jobs:latest" in text


def test_infra_refuses_to_deploy_without_the_phase_1_secrets() -> None:
    # An empty secret would reach Azure as an empty pp_writer password, hash salt or alert address.
    text = read("infra.yml")
    check = step("infra.yml", "Require the Phase 1 secrets")
    for secret in (
        "SQL_WRITER_PASSWORD",
        "AUTHOR_HASH_SALT",
        "ALERT_EMAIL",
        "PP_GITHUB_APP_PRIVATE_KEY",
    ):
        assert f"secrets.{secret}" in check
    assert "vars.PP_GITHUB_APP_ID" in check
    assert "exit 1" in check
    assert text.index("- name: Require the Phase 1 secrets") < text.index(
        "az deployment group what-if"
    )


def test_infra_passes_the_alert_address_to_what_if_and_deploy() -> None:
    assert read("infra.yml").count("ALERT_EMAIL: ${{ secrets.ALERT_EMAIL }}") == 3


def test_infra_passes_the_github_app_to_what_if_and_deploy() -> None:
    text = read("infra.yml")
    assert text.count("PP_GITHUB_APP_PRIVATE_KEY: ${{ secrets.PP_GITHUB_APP_PRIVATE_KEY }}") == 3
    assert text.count("PP_GITHUB_APP_ID: ${{ vars.PP_GITHUB_APP_ID }}") == 3


def test_guard_counts_both_jobs_executions() -> None:
    # The jobs share pp-api's vCPU grant (ADR-009); leaving them out would under-count the month.
    reading = step("cost-guard.yml", "Job executions this month")
    assert "az containerapp job execution list" in reading
    for job in ("pp-nightly", "pp-migrate"):
        assert job in reading
    evaluate = step("cost-guard.yml", "Evaluate")
    assert evaluate.count("--job-executions") == 2


def test_trip_deletes_the_nightly_job_and_verifies_it_is_gone() -> None:
    # A scheduled job can't be stopped, only deleted; infra.yml recreates it.
    trip = step("cost-guard.yml", "Trip - delete pp-nightly")
    assert "steps.guard.outputs.tripped == 'true'" in trip
    assert "inputs.dry_run" in trip
    assert 'az containerapp job delete -g "$RG" -n pp-nightly --yes' in trip
    assert trip.count("az containerapp job list") >= 2  # before, and after to verify
    assert "exit 1" in trip


def test_nightly_job_is_deleted_even_if_stopping_pp_api_fails() -> None:
    text = read("cost-guard.yml")
    assert text.index("- name: Trip - delete pp-nightly") < text.index("- name: Trip - stop pp-api")
    issue = step("cost-guard.yml", "Open or update the alert issue")
    assert "steps.trip_jobs.outcome" in issue


def test_guard_reports_a_failed_night_that_could_not_report_itself() -> None:
    # 2026-10-06: a broken app key failed the night and kept it from opening its own issue.
    reading = step("cost-guard.yml", "Job executions this month")
    assert "id: executions" in reading
    check = step("cost-guard.yml", "Check the last nightly run")
    assert "steps.executions.outcome == 'success'" in check
    assert "!cancelled()" in check  # also after a trip, or a failed trip step
    assert "patchpulse.monitoring.nightly_watch pp-nightly-executions.json" in check
    issue = step("cost-guard.yml", "Open the nightly failure issue")
    assert "steps.nightly.outputs.failed == 'true'" in issue
    # The title the publisher closes on the next successful night.
    assert f'title="{FAILURE_TITLE}"' in issue
    # One issue: none if one is open. The list API, unlike search, sees an issue opened seconds ago.
    assert "--state open" in issue
    assert "--search" not in issue
    assert "--body-file nightly-issue.md" in issue
    assert "${{" not in issue.split("run: |", 1)[1]  # Azure data reaches the shell only as a file


def test_publish_site_listens_for_the_dispatch_event() -> None:
    # The nightly job's site-data commit can't trigger a workflow; its repository_dispatch does.
    text = read("publish-site.yml")
    assert re.search(r"repository_dispatch:\n\s+types: \[site-data-updated\]", text)
    assert "workflow_dispatch:" in text
    assert '- "web/**"' in text


def test_publish_site_checks_out_the_site_data_or_fails_clearly() -> None:
    text = read("publish-site.yml")
    check = step("publish-site.yml", "Require the site-data branch")
    assert "git ls-remote --exit-code" in check
    assert "::error::" in check
    assert "ref: site-data" in text
    assert "path: web/public/data" in text


def test_swa_token_is_fetched_at_run_time_and_masked() -> None:
    # No stored deployment token: it is read with the OIDC identity on each run.
    fetch = step("publish-site.yml", "Fetch the Static Web App deployment token")
    assert "az staticwebapp secrets list" in fetch
    assert fetch.index("::add-mask::") < fetch.index("GITHUB_OUTPUT")
    assert "secrets.AZURE_STATIC_WEB_APPS" not in read("publish-site.yml")


def test_publish_site_never_puts_the_dispatch_payload_in_a_shell() -> None:
    # Whoever can send a dispatch controls client_payload; in a run: script it would be code.
    assert "client_payload" not in read("publish-site.yml")


def test_publish_site_uploads_the_prebuilt_site_only() -> None:
    deploy = step("publish-site.yml", "Deploy to the Static Web App")
    assert "Azure/static-web-apps-deploy@v1" in deploy
    assert "app_location: web/out" in deploy
    assert "skip_app_build: true" in deploy
    assert "skip_api_build: true" in deploy
