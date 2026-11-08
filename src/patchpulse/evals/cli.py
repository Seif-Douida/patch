"""`patchpulse-evals`: Phase 6's eval commands on Seif's machine (design D1).

- `data --fetch`: fetch and pin the night's published data bundle (no database).
- `freeze`: freeze the eval datasets before the first graded run.
- `run --arm <name>`: run an arm's cases on the pinned bundle through the eval cache. `fill` (the
  default) answers what the cache holds and asks Gemini for the rest; `replay` asks nothing.
  The injection arm plants each payload in a host incident's run (the main arm's runs pick the
  hosts) and prints how many runs were exposed.
- `judge --arms main,a2,a4`: judge the stored runs' citations and figures (design D5). Seif
  runs it: Claude never sends review text to the self-hosted server (ADR-020).
- `label-sample`: draw Seif's blind calibration sample and export it for the labeling app.
- `kappa`: the judge's agreement with Seif's labels, to reports/evals/judge-agreement.md.
- `report`: write reports/evals/<date>.md and .json from the stored runs and the judge's
  cached verdicts (`--mlflow` logs it).

Logging is configured as `patchpulse-agent`'s, so the Qdrant cluster's URL never reaches a line.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from collections import Counter
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
from patchpulse.agent.router import ModelRouter, make_agent_clients, next_pacific_midnight
from patchpulse.agent.search import ReviewSearch, open_pinned_search
from patchpulse.api.bundle_source import http_fetch
from patchpulse.api.settings import AGENT_DATA_URL
from patchpulse.config import get_lab_settings
from patchpulse.evals.arms import ARMS, Arm
from patchpulse.evals.data import DATA_PIN, fetch_data, load_pinned_bundle, read_data_pin
from patchpulse.evals.frozen import FROZEN_FILE, freeze, require_frozen
from patchpulse.evals.incidents import load_incidents
from patchpulse.evals.injection import (
    InjectionCase,
    assign_hosts,
    check_injection,
    injection_metrics,
    injection_run_cases,
    load_injection_cases,
)
from patchpulse.evals.judge import (
    AGREEMENT_PATH,
    RUBRIC_PATH,
    JudgeAgreement,
    JudgeItem,
    JudgeRun,
    agreement_estimates,
    calibration_sample,
    judge,
    judge_agreement,
    judge_chain,
    judge_items,
    make_judge_router,
    render_judge_agreement,
    rubric_version,
)
from patchpulse.evals.judge_audit import LABELS_PATH, export_audit, read_labels
from patchpulse.evals.records import CACHE_FILE, EVALS_DATA, StoredCase, load_arm
from patchpulse.evals.report import RunInfo, build_results, log_eval_run, write_report
from patchpulse.evals.runner import ArmOutcome, Case, make_router, run_arm
from patchpulse.evals.scoring import SCORING_VERSION
from patchpulse.ingest.http import RateLimiter
from patchpulse.labeling.limits import USAGE_FILE
from patchpulse.labeling.teachers import load_model_config

log = logging.getLogger("patchpulse.evals")

JUDGED_ARMS = "main,a2,a4"  # design D5: the arms whose citations the report compares


def _now() -> datetime:
    return datetime.now(UTC)


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


def _injection_set(wanted: str | None) -> tuple[list[InjectionCase], list[Case]]:
    """The injection cases asked for, and their runs on their hosts' questions."""
    injection = load_injection_cases()
    main = load_arm("main")
    if not main:
        raise ValueError("the injection arm plants its payloads in the main arm's runs: "
                         "run --arm main first")  # fmt: skip
    hosts = assign_hosts(injection, main)
    if wanted is not None:
        ids = [part.strip() for part in wanted.split(",") if part.strip()]
        known = {case.id for case in injection}
        unknown = [case_id for case_id in ids if case_id not in known]
        if unknown:
            raise ValueError(f"no injection case {', '.join(unknown)}")
        injection = [case for case in injection if case.id in ids]
    return injection, injection_run_cases(injection, hosts, load_incidents())


def exposure_line(cases: list[InjectionCase]) -> str:
    stored = load_arm("injection")
    outcomes = [check_injection(case, stored[case.id]) for case in cases if case.id in stored]
    unexposed = sorted(o.case_id for o in outcomes if not o.exposed)
    line = f"Exposed: {len(outcomes) - len(unexposed)} of {len(outcomes)}"
    if unexposed:
        line += f" (not exposed, so not counted: {', '.join(unexposed)})"
    return line + "."


def _run(arm: Arm, *, wanted: str | None, mode: CacheMode, encoder_dir: Path) -> int:
    if arm.name in ("main", "injection") and mode != "replay":
        require_frozen()  # the graded runs read only frozen datasets
    injection: list[InjectionCase] = []
    if arm.injection:
        injection, cases = _injection_set(wanted)
    else:
        cases = _cases(wanted)
    bundle = load_pinned_bundle(read_data_pin())
    settings = get_lab_settings()
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
                    clock=_now,
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
        router = make_router(arm, clients, cache, clock=_now)
        outcome = run_arm(
            arm, cases, bundle=bundle, router=router, cache=cache, search=search, steam=steam
        )
        reset = next_pacific_midnight(router.clock()).astimezone(UTC)
    print(outcome_line(arm, outcome, reset))
    if arm.injection:
        print(exposure_line(injection))
    return 0


def judge_line(run: JudgeRun) -> str:
    total = len(run.verdicts) + run.unanswered
    line = (
        f"Judged {total} items: {run.asked} asked, {run.cached} from the cache, {run.blank} "
        f"with no evidence (unsupported), {run.unanswered} unanswered"
    )
    if run.unanswered:
        line += "; run it again later for the rest"
    return line + "."


def _judge(wanted: str) -> int:
    names = [part.strip() for part in wanted.split(",") if part.strip()]
    unknown = [name for name in names if name not in ARMS]
    if unknown:
        raise ValueError(f"no arm {', '.join(unknown)}")
    require_frozen()  # the rubric is frozen with the datasets
    items = [i for name in names for case in load_arm(name).values() for i in judge_items(case)]
    if not items:
        raise ValueError(f"no stored claims in {', '.join(names)}: run --arm main first")
    with httpx2.Client() as http:
        cache = ReplayCache(EVALS_DATA / CACHE_FILE, "fill")
        router = make_judge_router(load_model_config(), get_lab_settings(), http, cache,
                                   clock=_now)  # fmt: skip
        run = judge(items, router)
    print(judge_line(run))
    return 0


def _replayed_verdicts(arms: dict[str, dict[str, StoredCase]]) -> dict[str, bool] | None:
    """The judge's cached verdicts on the stored runs; None when it hasn't answered any."""
    items = [i for stored in arms.values() for case in stored.values() for i in judge_items(case)]
    cache = ReplayCache(EVALS_DATA / CACHE_FILE, "replay")
    run = judge(items, ModelRouter(judge_chain(load_model_config()), {}, cache, clock=_now))
    return run.verdicts if run.cached else None


def _main_items() -> list[JudgeItem]:
    items = [item for case in load_arm("main").values() for item in judge_items(case)]
    if not items:
        raise ValueError("no stored claims in the main arm: run `patchpulse-evals run --arm main`")
    return items


def _label_sample() -> int:
    sample = calibration_sample(_main_items())
    path = export_audit(sample)
    counts = Counter(item.kind for item in sample.items)
    line = f"Exported {counts['citation']} citation and {counts['figure']} figure items to {path}"
    if sample.short:
        line += f" ({', '.join(f'{k} {n} short' for k, n in sorted(sample.short.items()))})"
    print(line + ". Label them in the labeling app's Judge audit mode.")
    return 0


def _agreement() -> JudgeAgreement | None:
    """The judge's agreement with Seif's labels, on its cached verdicts; None before labels."""
    labels = read_labels(LABELS_PATH)
    if not labels:
        return None
    return judge_agreement(labels, _replayed_verdicts({"main": load_arm("main")}) or {})


def _kappa() -> int:
    agreement = _agreement()
    if agreement is None:
        raise ValueError(f"no labels in {LABELS_PATH}: label the sample in the app first")
    AGREEMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AGREEMENT_PATH.write_text(render_judge_agreement(agreement), encoding="utf-8", newline="\n")
    kappas = ", ".join(
        f"{name} {'-' if row.kappa is None else f'{row.kappa:.2f}'}"
        for name, row in agreement.rows.items()
    )
    print(f"Wrote {AGREEMENT_PATH}: {'passed' if agreement.passed else 'failed'} ({kappas}).")
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
    verdicts = _replayed_verdicts(arms)
    info = RunInfo(
        date=date or datetime.now(UTC).date().isoformat(),
        data_version=pin.data_version,
        scoring_version=SCORING_VERSION,
        prompt_version=PROMPT_VERSION,
        rubric_version=rubric_version() if RUBRIC_PATH.exists() else None,
        chains={name: ARMS[name].chain for name in arms},
        judge_chain=tuple(judge_chain(load_model_config())) if verdicts else None,
        commit=_commit(),
    )
    agreement = _agreement()
    injected = load_arm("injection")
    injection = None
    if injected:
        cases = {case.id: case for case in load_injection_cases()}
        outcomes = [check_injection(cases[k], v) for k, v in sorted(injected.items())
                    if k in cases]  # fmt: skip
        injection = injection_metrics(outcomes)
    results = build_results(
        load_incidents(),
        arms,
        info,
        verdicts=verdicts,
        injection=injection,
        judge=None if agreement is None else agreement_estimates(agreement),
    )
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
    judged = commands.add_parser("judge", help="judge the stored runs' citations and figures")
    judged.add_argument("--arms", default=JUDGED_ARMS,
                        help=f"comma-separated arms (default {JUDGED_ARMS})")  # fmt: skip
    commands.add_parser("label-sample", help="draw and export the judge's calibration sample")
    commands.add_parser("kappa", help=f"the judge's agreement with Seif ({AGREEMENT_PATH})")
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
        if args.command == "judge":
            return _judge(args.arms)
        if args.command == "label-sample":
            return _label_sample()
        if args.command == "kappa":
            return _kappa()
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
