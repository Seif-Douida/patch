"""`patchpulse-agent`: the agent's commands on Seif's machine (Phase 5).

    backfill-reports   Write a report for every past patch bundle that has none (design D10).

`backfill-reports` signs in to Azure SQL as the person at the keyboard (as `patchpulse-ml pull`
does), builds the agent's bundle from the marts, and writes the missing reports newest first
through the same graph and store as the nightly job, as one run in ops.pipeline_run. It never
stores a basic report: when the day's Gemini quota is used up, or no model answers, it stops and a
later run picks up where it left off (stored reports are skipped). The database stays awake while
it runs, so it needs --yes.

Exit codes: 0 done (stopping at the quota included), 2 refused or bad configuration.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol

import httpx2
from sqlalchemy import Connection, Engine

from patchpulse.agent.bundle import Bundle, build_bundle
from patchpulse.agent.graph import AgentDeps, ReportRequest, run_report
from patchpulse.agent.llm_cache import AGENT_CACHE_FILE, ReplayCache
from patchpulse.agent.report import Report
from patchpulse.agent.router import ModelRouter, make_agent_clients
from patchpulse.agent.search import ReviewSearch, open_pinned_search
from patchpulse.agent.store import (
    latest_reports,
    report_fingerprint,
    report_inputs,
    reports_due,
    write_report,
)
from patchpulse.config import LabSettings, get_lab_settings
from patchpulse.db.engine import connect_with_resume_retry, make_entra_engine
from patchpulse.ingest.http import RateLimiter, SteamHttp
from patchpulse.labeling.limits import USAGE_FILE
from patchpulse.labeling.teachers import load_model_config
from patchpulse.pipeline.runs import finish_run, start_run

log = logging.getLogger("patchpulse.agent")

RUN_KIND = "reports_backfill"
QUOTA_SPENT = "the day's model quota is used up"
NO_MODEL = "no model answered"
PROGRESS_EVERY = 10
ENCODER_DIR = Path("build/encoder")  # `patchpulse-ml retrieval encoder download --to build/encoder`
BACKFILL_COST = (
    "`backfill-reports` wakes Azure SQL and keeps it awake while it runs (about an hour for every "
    "past patch, then the 60-minute pause delay): about 4,000-6,000 of the month's 100,000 free "
    "vCore-s. Each report also takes about 3 Gemini free-tier requests (a day's 1,000 cover about "
    "300). Run it again with --yes to go ahead."
)


class BundleBuilder(Protocol):
    def __call__(self, connection: Connection, *, run_id: int, today: date) -> Bundle: ...


@dataclass(frozen=True)
class BackfillOutcome:
    missing: int  # complete bundle horizons without a report when the run started
    written: int
    skipped: int  # reports that failed or came out basic with models answering; tried next run
    stopped: str | None  # why the run ended early, if it did


def backfill_reports(
    engine: Engine,
    deps_for: Callable[[Bundle], AgentDeps],
    *,
    today: date,
    build: BundleBuilder = build_bundle,
    out: Callable[[str], None] = print,
    every: int = PROGRESS_EVERY,
) -> BackfillOutcome:
    """Write the missing reports, newest first, as one run; `deps_for` gives the agent its models,
    search and Steam for the bundle."""
    with engine.begin() as connection:
        run_id = start_run(connection, kind=RUN_KIND, image_version=None)
    with engine.connect() as connection:
        bundle = build(connection, run_id=run_id, today=today)
        stored = latest_reports(connection)
    missing = [key for key in reports_due(bundle, {}) if key not in stored]
    deps = deps_for(bundle) if missing else None
    written = skipped = 0
    stopped: str | None = None
    for done, (bundle_id, horizon) in enumerate(missing, start=1):
        if deps is None:
            break
        try:
            result = run_report(ReportRequest.for_bundle(bundle_id, horizon), deps)
        except Exception as error:
            log.warning("%s-%dd failed: %s: %s", bundle_id, horizon, type(error).__name__, error)
            skipped += 1
            result = None
        if isinstance(result, Report) and result.basic:
            if deps.router.quota_spent():
                stopped = QUOTA_SPENT
                break
            if not result.model_ids:
                stopped = NO_MODEL
                break
            log.warning("%s: a basic report (the draft didn't hold); skipped", result.report_id)
            skipped += 1
        elif isinstance(result, Report):
            fingerprint = report_fingerprint(report_inputs(bundle, bundle_id, horizon))
            with engine.begin() as connection:
                written += int(
                    write_report(connection, result, fingerprint=fingerprint, run_id=run_id)
                )
        elif result is not None:
            log.warning("%s-%dd: refused (%s); skipped", bundle_id, horizon, result.reason)
            skipped += 1
        if done % every == 0 and done < len(missing):
            out(f"{done} of {len(missing)} reports done")
        if deps.router.quota_spent():
            stopped = QUOTA_SPENT
            break
    outcome = BackfillOutcome(len(missing), written, skipped, stopped)
    with engine.begin() as connection:
        finish_run(connection, run_id, status="succeeded", stages=asdict(outcome), error=None)
    summary = f"{written} of {len(missing)} missing reports written"
    if stopped is None:
        out(f"{summary}." + (f" {skipped} skipped: run the command again." if skipped else ""))
    else:
        out(f"Stopped: {stopped}. {summary}; run the command again tomorrow for the rest.")
    return outcome


def _live_deps(
    stack: ExitStack, settings: LabSettings, encoder_dir: Path
) -> Callable[[Bundle], AgentDeps]:
    """The Gemini chain of config/models.yaml (counting requests with Phase 2's labeling, in
    data/local), the pinned review search, and Steam for patch notes."""
    if settings.gemini_api_key is None:
        raise ValueError("backfill-reports needs PP_GEMINI_API_KEY")
    models = load_model_config()
    if models.agent is None:
        raise ValueError("config/models.yaml has no agent section")

    def now() -> datetime:
        return datetime.now(UTC)

    http = stack.enter_context(httpx2.Client())
    clients = make_agent_clients(
        models,
        settings.gemini_api_key.get_secret_value(),
        http,
        state_path=settings.data_dir / USAGE_FILE,
        clock=now,
        sleep=time.sleep,
    )
    router = ModelRouter(models.agent.chain, clients, ReplayCache(Path(AGENT_CACHE_FILE), "live"),
                         clock=now)  # fmt: skip
    search: ReviewSearch | None = None
    if settings.qdrant_url is not None and encoder_dir.is_dir():
        search = open_pinned_search(settings.qdrant_url, settings.qdrant_api_key, encoder_dir)
        stack.callback(search.client.close)
    else:
        log.warning("no review search (PP_QDRANT_URL or %s missing): reports cite no reviews",
                    encoder_dir)  # fmt: skip
    steam = SteamHttp(stack.enter_context(httpx2.Client()), limiter=RateLimiter())

    def deps(bundle: Bundle) -> AgentDeps:
        return AgentDeps(bundle=bundle, router=router, search=search, steam=steam)

    return deps


def configure_logging() -> None:
    """INFO for PatchPulse; azure-identity's and the HTTP clients' lines only on warnings (the
    Qdrant cluster's URL, a secret, is in httpx's request lines)."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("patchpulse").setLevel(logging.INFO)
    for noisy in ("azure", "httpx", "httpcore", "httpx2", "httpcore2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _backfill(settings: LabSettings, *, encoder_dir: Path) -> int:
    if settings.pull_host is None or settings.gemini_api_key is None:
        print(
            "patchpulse-agent: backfill-reports needs PP_PULL_HOST and PP_GEMINI_API_KEY",
            file=sys.stderr,
        )
        return 2
    print("patchpulse-agent: waking Azure SQL for this run", file=sys.stderr)
    engine = make_entra_engine(settings.pull_host, settings.pull_database)
    try:
        connect_with_resume_retry(engine).close()  # wake a paused database first
        with ExitStack() as stack:
            backfill_reports(
                engine,
                _live_deps(stack, settings, encoder_dir),
                today=datetime.now(UTC).date(),
            )
    finally:
        engine.dispose()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="patchpulse-agent", description="The agent's commands on Seif's machine (Phase 5)."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    backfill = commands.add_parser(
        "backfill-reports",
        help="write a report for every past patch bundle without one (wakes the database)",
    )
    backfill.add_argument("--yes", action="store_true", help="go ahead and wake the database")
    backfill.add_argument(
        "--encoder-dir",
        type=Path,
        default=ENCODER_DIR,
        help=f"the pinned review encoder (default {ENCODER_DIR})",
    )
    args = parser.parse_args(argv)
    configure_logging()
    if not args.yes:
        print(f"patchpulse-agent: {BACKFILL_COST}", file=sys.stderr)
        return 2
    return _backfill(get_lab_settings(), encoder_dir=args.encoder_dir)


if __name__ == "__main__":
    sys.exit(main())
