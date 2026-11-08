"""Running an arm (Phase 6 design D2): every case once, as a visitor's question on its game, on the
frozen data bundle, through the eval cache.

A case already stored is skipped, so a run stopped by the day's quota or an hour of outage picks up
where it stopped, and the cache answers every prompt it already holds (fill mode). A case
interrupted by the quota or an outage is never saved: its answers would be a basic report or a
refusal that only the outage wrote. In replay mode nothing is asked; what the cache can't answer
is named in the case's misses.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from patchpulse.agent.bundle import Bundle
from patchpulse.agent.cli import OUTAGE, OUTAGE_LIMIT, OUTAGE_PAUSE_S, QUOTA_SPENT
from patchpulse.agent.graph import AgentDeps, ReportRequest, run_recorded
from patchpulse.agent.llm_cache import ReplayCache
from patchpulse.agent.router import ModelClient, ModelRouter
from patchpulse.agent.search import ReviewSearch
from patchpulse.agent.tools import ExternalCalls
from patchpulse.evals.arms import Arm
from patchpulse.evals.records import EVALS_DATA, StoredCase, require_local, runs_dir, save_case
from patchpulse.ingest.http import SteamHttp
from patchpulse.retrieval.search import SearchFilters, query_filter

log = logging.getLogger("patchpulse.evals")


@dataclass(frozen=True)
class Case:
    """One question an arm asks: a golden incident's, or an injection case's on its host."""

    case_id: str
    appid: int
    question: str
    # Wraps the eval cache for this case only (the injection set plants its review this way).
    wrap: Callable[[ExternalCalls], ExternalCalls] | None = None


@dataclass(frozen=True)
class ArmOutcome:
    recorded: int
    already: int
    missed: dict[str, list[str]] = field(default_factory=dict)  # case -> what replay lacked
    stopped: str | None = None  # QUOTA_SPENT or OUTAGE


def make_router(
    arm: Arm,
    clients: Mapping[str, ModelClient],
    cache: ReplayCache,
    *,
    clock: Callable[[], datetime],
) -> ModelRouter:
    """The arm's single-model chain over the clients built for the agent's chain."""
    return ModelRouter(list(arm.chain), clients, cache, clock=clock)


def _day_key(day: date) -> int:
    return day.year * 10_000 + day.month * 100 + day.day


def indexed_reviews(
    search: ReviewSearch | None, appid: int, window: tuple[date, date] | None, data_version: int
) -> int | None:
    """How many reviews the index held for the scoped window, with the agent's own filter; None
    without a search or a window, or when the cluster doesn't answer (it's informational)."""
    if search is None or window is None:
        return None
    filters = SearchFilters(
        start_day=_day_key(window[0]), end_day=_day_key(window[1]), as_of_run_id=data_version
    )
    try:
        counted: Any = search.client.count(
            collection_name=search.collection,
            count_filter=query_filter(appid, filters),
            exact=True,
        )
        return int(counted.count)
    except Exception as error:
        # The type only: a message can carry the cluster's URL, a secret.
        log.warning("couldn't count the window's indexed reviews (%s)", type(error).__name__)
        return None


def run_arm(
    arm: Arm,
    cases: Sequence[Case],
    *,
    bundle: Bundle,
    router: ModelRouter,
    cache: ReplayCache,
    search: ReviewSearch | None,
    steam: SteamHttp | None,
    sleep: Callable[[float], None] = time.sleep,
    root: Path = EVALS_DATA,
) -> ArmOutcome:
    folder = runs_dir(arm.name, root)
    require_local(folder)
    recorded = already = 0
    missed: dict[str, list[str]] = {}
    outage_since: datetime | None = None
    for case in cases:
        path = folder / f"{case.case_id}.json"
        if path.exists():
            already += 1
            continue
        deps = AgentDeps(
            bundle=bundle,
            router=router,
            search=search,
            steam=steam,
            external=case.wrap(cache) if case.wrap is not None else cache,
            search_mode=arm.search_mode,
            causal_tool=arm.causal_tool,
        )
        while True:
            cache.misses.clear()
            outages = router.outages
            record = run_recorded(ReportRequest.for_question(case.question, appid=case.appid), deps)
            if router.quota_spent():
                return ArmOutcome(recorded, already, missed, QUOTA_SPENT)
            if router.outages == outages:
                break
            # Every model failed at some step: Gemini's outage, not this case. Ask again later.
            now = router.clock()
            outage_since = outage_since or now
            if now - outage_since >= OUTAGE_LIMIT:
                return ArmOutcome(recorded, already, missed, OUTAGE)
            ready = router.next_ready()
            wait = float(OUTAGE_PAUSE_S)
            if ready is not None:
                wait = max(wait, (ready - now).total_seconds())
            log.warning("%s: every model failed; waiting %.0f s before asking again",
                        case.case_id, wait)  # fmt: skip
            sleep(wait)
        outage_since = None
        misses = list(cache.misses)
        stored = StoredCase.from_record(
            record,
            case=case.case_id,
            arm=arm.name,
            data_version=bundle.data_version,
            misses=misses,
            indexed_reviews=indexed_reviews(search, case.appid, record.window, bundle.data_version),
        )
        save_case(stored, path)
        recorded += 1
        if misses:
            missed[case.case_id] = misses
    return ArmOutcome(recorded, already, missed, None)
