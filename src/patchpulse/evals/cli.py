"""`patchpulse-evals`: Phase 6's eval commands on Seif's machine (design D1).

- `data --fetch`: fetch and pin the night's published data bundle (no database).
- `freeze`: freeze the eval datasets before the first graded run.
- `run --arm <name>`: run an arm's cases on the pinned bundle through the eval cache. `fill` (the
  default) answers what the cache holds and asks Gemini for the rest; `replay` asks nothing.
- `report`: write reports/evals/<date>.md and .json from the stored runs (`--mlflow` logs it).

Logging is configured as `patchpulse-agent`'s, so the Qdrant cluster's URL never reaches a line.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2

from patchpulse.agent.cli import ENCODER_DIR, QUOTA_SPENT, configure_logging
from patchpulse.agent.llm_cache import CacheMode, ReplayCache
from patchpulse.agent.patch_notes import agent_steam
from patchpulse.agent.prompts import PROMPT_VERSION
from patchpulse.agent.router import make_agent_clients, next_pacific_midnight
from patchpulse.agent.search import ReviewSearch, open_pinned_search
from patchpulse.api.bundle_source import http_fetch
from patchpulse.api.settings import AGENT_DATA_URL
from patchpulse.config import get_lab_settings
from patchpulse.evals.arms import ARMS, Arm
from patchpulse.evals.data import DATA_PIN, fetch_data, load_pinned_bundle, read_data_pin
from patchpulse.evals.frozen import FROZEN_FILE, freeze, require_frozen
from patchpulse.evals.incidents import load_incidents
from patchpulse.evals.records import CACHE_FILE, EVALS_DATA, load_arm
from patchpulse.evals.report import RunInfo, build_results, log_eval_run, write_report
from patchpulse.evals.runner import ArmOutcome, Case, make_router, run_arm
from patchpulse.evals.scoring import SCORING_VERSION
from patchpulse.ingest.http import RateLimiter
from patchpulse.labeling.limits import USAGE_FILE
from patchpulse.labeling.teachers import load_model_config

log = logging.getLogger("patchpulse.evals")


def outcome_line(arm: Arm, outcome: ArmOutcome, reset: datetime | None) -> str:
    line = (
        f"{arm.name}: {outcome.recorded} recorded, {outcome.already} already there, "
        f"{len(outcome.missed)} with misses"
    )
    if outcome.missed:
        line += f" ({', '.join(sorted(outcome.missed))})"
    if outcome.stopped == QUOTA_SPENT and reset is not None:
        line += (
            f". Stopped: {outcome.stopped}; run it again after midnight Pacific "
            f"({reset:%H:%M} UTC) for the rest"
        )
    elif outcome.stopped is not None:
        line += f". Stopped: {outcome.stopped} (high demand); run it again later for the rest"
    return line + "."


def _cases(wanted: str | None) -> list[Case]:
    incidents = load_incidents()
    if wanted is None:
        return [Case(i.id, i.appid, i.question) for i in incidents]
    ids = [part.strip() for part in wanted.split(",") if part.strip()]
    known = {i.id: i for i in incidents}
    unknown = [case_id for case_id in ids if case_id not in known]
    if unknown:
        raise ValueError(f"no incident {', '.join(unknown)}")
    return [Case(known[i].id, known[i].appid, known[i].question) for i in ids]


def _run(arm: Arm, *, wanted: str | None, mode: CacheMode, encoder_dir: Path) -> int:
    if arm.injection:
        raise ValueError("the injection arm needs the injection set (evals/injection/cases.yaml)")
    if arm.name == "main" and mode != "replay":
        require_frozen()  # the graded run reads only frozen datasets
    bundle = load_pinned_bundle(read_data_pin())
    cases = _cases(wanted)
    settings = get_lab_settings()

    def now() -> datetime:
        return datetime.now(UTC)

    with ExitStack() as stack:
        cache = ReplayCache(EVALS_DATA / CACHE_FILE, mode)
        clients: dict[str, Any] = {}
        search: ReviewSearch | None = None
        steam = None
        if mode != "replay":
            if settings.gemini_api_key is None:
                raise ValueError("a live run needs PP_GEMINI_API_KEY")
            http = stack.enter_context(httpx2.Client())
            clients = dict(
                make_agent_clients(
                    load_model_config(),
                    settings.gemini_api_key.get_secret_value(),
                    http,
                    state_path=settings.data_dir / USAGE_FILE,
                    clock=now,
                    sleep=time.sleep,
                )
            )
            if settings.qdrant_url is not None and encoder_dir.is_dir():
                search = open_pinned_search(
                    settings.qdrant_url, settings.qdrant_api_key, encoder_dir
                )
                stack.callback(search.client.close)
            else:
                log.warning("no review search (PP_QDRANT_URL or %s missing): runs cite no reviews",
                            encoder_dir)  # fmt: skip
            steam = agent_steam(stack.enter_context(httpx2.Client()), RateLimiter())
        router = make_router(arm, clients, cache, clock=now)
        outcome = run_arm(
            arm, cases, bundle=bundle, router=router, cache=cache, search=search, steam=steam
        )
        reset = next_pacific_midnight(router.clock()).astimezone(UTC)
    print(outcome_line(arm, outcome, reset))
    return 0


def _commit() -> str:
    """The checkout's commit, short; "unknown" without git."""
    git = shutil.which("git")
    if git is None:
        return "unknown"
    found = subprocess.run(  # noqa: S603 - a found executable, arguments from code
        [git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return found.stdout.strip() or "unknown"


def _report(*, date: str | None, to_mlflow: bool) -> int:
    arms = {name: load_arm(name) for name in ("main", "a2", "a3", "a4")}
    arms = {name: stored for name, stored in arms.items() if stored}
    if "main" not in arms:
        raise ValueError("no stored main run: run `patchpulse-evals run --arm main` first")
    pin = read_data_pin()
    info = RunInfo(
        date=date or datetime.now(UTC).date().isoformat(),
        data_version=pin.data_version,
        scoring_version=SCORING_VERSION,
        prompt_version=PROMPT_VERSION,
        rubric_version=None,
        chains={name: ARMS[name].chain for name in arms},
        judge_chain=None,
        commit=_commit(),
    )
    results = build_results(load_incidents(), arms, info)
    markdown, document = write_report(results)
    print(f"Wrote {markdown} and {document}.")
    if to_mlflow:
        print(f"Logged to MLflow, run {log_eval_run(results, markdown)}.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="patchpulse-evals", description="Phase 6's evals on Seif's machine (no database)."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help=f"fetch and pin the published data bundle ({DATA_PIN})")
    data.add_argument("--fetch", action="store_true", required=True, help="fetch it now")
    commands.add_parser("freeze", help=f"freeze the eval datasets ({FROZEN_FILE})")
    run = commands.add_parser("run", help="run an arm's cases through the eval cache")
    run.add_argument("--arm", choices=sorted(ARMS), required=True)
    run.add_argument("--cases", help="comma-separated incident ids (default: every incident)")
    run.add_argument("--mode", choices=("fill", "replay", "record"), default="fill",
                     help="fill: ask only what the cache lacks (default); replay: ask nothing; "
                          "record: ask everything again")  # fmt: skip
    run.add_argument(
        "--encoder-dir",
        type=Path,
        default=ENCODER_DIR,
        help=f"the pinned review encoder (default {ENCODER_DIR})",
    )
    report = commands.add_parser("report", help="write the eval report from the stored runs")
    report.add_argument("--date", help="the report's date (default: today, UTC)")
    report.add_argument("--mlflow", action="store_true", help="also log it to MLflow (DagsHub)")
    args = parser.parse_args(argv)
    configure_logging()
    try:
        if args.command == "data":
            pin = fetch_data(http_fetch(AGENT_DATA_URL))
            print(f"Data version {pin.data_version} pinned in {DATA_PIN}.")
            return 0
        if args.command == "report":
            return _report(date=args.date, to_mlflow=args.mlflow)
        if args.command == "freeze":
            digest = freeze()
            print(f"Frozen in {FROZEN_FILE} (sha256 {digest[:12]}).")
            return 0
        return _run(ARMS[args.arm], wanted=args.cases, mode=args.mode, encoder_dir=args.encoder_dir)
    except ValueError as error:
        print(f"patchpulse-evals: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
