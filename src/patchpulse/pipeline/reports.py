"""The nightly `reports` stage's work and the API's files (Phase 5 plan Task 12, design D2 and
D10); the runner wraps them.

- The reports: the due ones (agent/store.py: no report yet, a basic one, or changed evidence), the
  latest windows first, each through the agent's graph on tonight's bundle, until the runner's
  deadline (PP_REPORTS_BUDGET_MINUTES, within the night). A report started before the deadline
  finishes. One that raises is a `report_failed` warning and the rest go on: otherwise the newest
  could block every older one night after night.
- The API's files on the orphan `agent-data` branch: the bundle, and reports.json with the public
  copy each bundle shows (the 14-day one once it exists) and the patch-to-bundle map.

Review text never reaches those files: the bundle holds aggregates only, and a public copy has no
quote spans, review ids or withheld text.
"""

from __future__ import annotations

import functools
import json
import logging
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx2
from sqlalchemy import Engine

from patchpulse.agent.bundle import Bundle
from patchpulse.agent.graph import AgentDeps, Refusal, ReportRequest, run_report
from patchpulse.agent.llm_cache import AGENT_CACHE_FILE, ReplayCache
from patchpulse.agent.report import Report, public_copy
from patchpulse.agent.router import ModelRouter, make_agent_clients
from patchpulse.agent.search import ReviewSearch, open_pinned_search
from patchpulse.agent.store import (
    Key,
    StoredReport,
    latest_reports,
    report_fingerprint,
    report_inputs,
    reports_due,
    shown_reports,
    write_report,
)
from patchpulse.config import Settings, scrub_secrets
from patchpulse.ingest.http import SteamHttp
from patchpulse.ingest.load import record_dq
from patchpulse.labeling.teachers import load_model_config

log = logging.getLogger("patchpulse.pipeline.reports")

Agent = Callable[[ReportRequest], Report | Refusal]


@dataclass(frozen=True)
class ReportsOutcome:
    due: int  # reports due tonight
    written: int  # rows stored
    basic: int  # reports that came out basic tonight (no model answered)
    failed: int  # reports that raised; due again the next night
    remaining: int  # due but left for the next night by the deadline


def reports_file(bundle: Bundle, stored: Mapping[Key, StoredReport], *, data_version: int) -> bytes:
    """reports.json: the public copy of the report each bundle shows, made from the stored
    report as the site export makes it, and each patch's bundle."""
    patches = {
        p.patch_id: p.bundle_id for g in bundle.games for p in g.patches if p.bundle_id is not None
    }
    document = {
        "data_version": data_version,
        "reports": {b: public_copy(row.report) for b, row in shown_reports(stored).items()},
        "patches": dict(sorted(patches.items())),
    }
    return (json.dumps(document, ensure_ascii=False) + "\n").encode("utf-8")


def _search(stack: ExitStack, settings: Settings) -> ReviewSearch | None:
    """The pinned collection and encoder, as the index stage uses them; none without Qdrant."""
    if settings.qdrant_url is None or settings.encoder_dir is None:
        log.info("reports: no review search (PP_QDRANT_URL or PP_ENCODER_DIR unset)")
        return None
    search = open_pinned_search(settings.qdrant_url, settings.qdrant_api_key, settings.encoder_dir)
    stack.callback(search.client.close)
    return search


def _live_agent(stack: ExitStack, settings: Settings, bundle: Bundle, steam: SteamHttp) -> Agent:
    """The graph with the Gemini chain of config/models.yaml, live (no cache)."""
    if settings.gemini_api_key is None:
        raise ValueError("the reports stage needs PP_GEMINI_API_KEY")
    models = load_model_config()
    if models.agent is None:
        raise ValueError("config/models.yaml has no agent section")

    def now() -> datetime:
        return datetime.now(UTC)

    http = stack.enter_context(httpx2.Client())
    # The free tier's per-day counts, for this run only: the job's file system is gone by morning.
    state = Path(stack.enter_context(tempfile.TemporaryDirectory())) / "gemini-requests.json"
    clients = make_agent_clients(
        models,
        settings.gemini_api_key.get_secret_value(),
        http,
        state_path=state,
        clock=now,
        sleep=time.sleep,
    )
    router = ModelRouter(
        models.agent.chain, clients, ReplayCache(Path(AGENT_CACHE_FILE), "live"), clock=now
    )
    deps = AgentDeps(bundle=bundle, router=router, search=_search(stack, settings), steam=steam)
    return functools.partial(run_report, deps=deps)


def run_reports(
    engine: Engine,
    settings: Settings,
    bundle: Bundle,
    *,
    run_id: int,
    steam: SteamHttp,
    clock: Callable[[], float],
    deadline: float,
    agent: Agent | None = None,
) -> ReportsOutcome:
    """Write the due reports, newest first, each in its own transaction, until `deadline` (on
    `clock`). `agent` replaces the live graph in tests."""
    began = time.monotonic()
    with engine.connect() as connection:
        due = reports_due(bundle, latest_reports(connection))
    appids = {p.bundle_id: g.appid for g in bundle.games for p in g.patches if p.bundle_id}
    written = basic = failed = started = 0
    with ExitStack() as stack:
        if due and agent is None:
            agent = _live_agent(stack, settings, bundle, steam)
        for bundle_id, horizon in due:
            if agent is None or clock() >= deadline:
                break
            started += 1
            try:
                result = agent(ReportRequest.for_bundle(bundle_id, horizon))
                if isinstance(result, Refusal):
                    raise RuntimeError(f"refused: {result.reason}")
                fingerprint = report_fingerprint(report_inputs(bundle, bundle_id, horizon))
                with engine.begin() as connection:
                    stored = write_report(
                        connection, result, fingerprint=fingerprint, run_id=run_id
                    )
            except Exception as error:
                failed += 1
                detail = scrub_secrets(
                    f"{bundle_id}-{horizon}d: {type(error).__name__}: {error}", settings
                )
                log.warning("reports: %s", detail)
                with engine.begin() as connection:
                    record_dq(
                        connection,
                        run_id=run_id,
                        check="report_failed",
                        appid=appids.get(bundle_id),
                        severity="warn",
                        detail=detail,
                    )
                continue
            written += int(stored)
            basic += int(result.basic)
    outcome = ReportsOutcome(
        due=len(due), written=written, basic=basic, failed=failed, remaining=len(due) - started
    )
    log.info(
        "reports: %d due, %d written, %d basic, %d failed, %d left for the next night (%.1f s)",
        outcome.due,
        outcome.written,
        outcome.basic,
        outcome.failed,
        outcome.remaining,
        time.monotonic() - began,
    )
    return outcome
