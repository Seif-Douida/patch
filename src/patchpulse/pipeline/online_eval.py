"""Online evals of the night's reports (Phase 7 design D4, spec §14 "Online agent evals").

Every report the night wrote is scored: the verifier's numbers from the report itself (the share
of drafted claims that shipped, withheld claims by rule, rewordings), and the Phase 6 judge on its
shipped claims (citation support, figure faithfulness), by Google's free Gemma 4 31B. The judge
failed its calibration (Phase 6), so its numbers are named `judge.` and never alert. Everything
goes to `ops.eval_metric` and, as scores, onto the report's Langfuse trace.

A night's rows (scope `night`) pool its reports, and count its basic reports as `gemini_errors`:
the router doesn't keep a count of failed attempts, and a basic report is what an outage or a
spent quota produces (design D7: shown, never alerted).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import httpx2
from sqlalchemy import Connection, Engine, text

from patchpulse.agent.llm_cache import ReplayCache
from patchpulse.agent.report import Report
from patchpulse.agent.router import ModelRouter
from patchpulse.agent.tracing import Tracer
from patchpulse.config import Settings
from patchpulse.evals.judge import METRIC_KEYS, judge, judge_items, judge_metrics
from patchpulse.evals.records import StoredCase
from patchpulse.evals.scoring import claim_stats
from patchpulse.labeling.limits import RequestLimiter
from patchpulse.labeling.teachers import load_model_config
from patchpulse.monitoring.alerts import AlertRule, Outcome
from patchpulse.monitoring.store import MetricRow, write_metrics
from patchpulse.pipeline.reports import ScoredRun

log = logging.getLogger("patchpulse.pipeline.online_eval")

ONLINE_ARM = "online"
JUDGE_MODEL = "gemma-4-31b"  # Google's copy; the friend's server never runs in Azure (ADR-020)
VERIFIER_REPORTS = 10  # the verifier alert pools the last 10 reports
SCORE_PREFIX = "online."

_LAST_REPORTS = """
SELECT TOP (:n) value, n FROM ops.eval_metric
WHERE family = 'online' AND metric = 'pre_verifier_pass_rate' AND scope LIKE 'report:%'
  AND value IS NOT NULL AND n > 0
ORDER BY id DESC
"""


@dataclass(frozen=True)
class OnlineOutcome:
    reports: int
    judged_items: int
    judge_failed: bool
    units_used: int | None = None  # the month's Langfuse units, for the alert


def online_judge_router(
    settings: Settings,
    http: httpx2.Client,
    *,
    state_path: Path,
    cache_path: Path,
    clock: Callable[[], datetime],
    sleep: Callable[[float], None] = time.sleep,
) -> ModelRouter:
    """Google's Gemma 4 31B alone, live (nothing cached across nights). A timeout or an outage
    raises at once, as in the agent's router: with the deadline checked before each item, a stall
    costs the stage one request past its budget, not the night (final review C1)."""
    from patchpulse.labeling.gemini import GeminiClient

    if settings.gemini_api_key is None:
        raise ValueError("the online judge needs PP_GEMINI_API_KEY")
    config = load_model_config().candidates[JUDGE_MODEL].model_copy(update={"use_fraction": 1.0})
    limiter = RequestLimiter(config, state_path=state_path, clock=clock, sleep=sleep)
    client = GeminiClient(
        settings.gemini_api_key.get_secret_value(),
        http,
        limiter,
        config,
        sleep=sleep,
        max_attempts=2,  # the second attempt only waits out a per-minute limit
        retry_errors=False,
    )
    return ModelRouter([JUDGE_MODEL], {JUDGE_MODEL: client}, ReplayCache(cache_path, "live"),
                       clock=clock)  # fmt: skip


def _stored(run: ScoredRun) -> StoredCase:
    result = run.record.result
    version = result.data_version if isinstance(result, Report) else 0
    return StoredCase.from_record(
        run.record, case=run.report_id, arm=ONLINE_ARM, data_version=version, misses=[],
        indexed_reviews=None,
    )  # fmt: skip


def report_metrics(run: ScoredRun, verdicts: dict[str, bool]) -> list[MetricRow]:
    scope = f"report:{run.report_id}"
    today = datetime.now(UTC).date()

    def row(metric: str, value: float | None, n: int | None = None) -> MetricRow:
        return MetricRow("online", metric, scope, today, today, value, None, None, n)

    stats = claim_stats(run.record)
    rows: list[MetricRow] = []
    if stats is not None:
        rate = stats.shipped / stats.drafted if stats.drafted else None
        rows.append(row("pre_verifier_pass_rate", rate, stats.drafted))
        rows += [
            row(f"withheld.{rule}", float(count)) for rule, count in stats.withheld_by_rule.items()
        ]
        rows.append(row("reworded", float(stats.reworded)))
    result = run.record.result
    if isinstance(result, Report):
        passed = result.rule_pass_rates.get("V6")
        rows.append(row("summary_passed", None if passed is None else float(passed)))
    if verdicts:
        judged = judge_metrics({run.report_id: _stored(run)}, verdicts)
        for key in METRIC_KEYS.values():
            estimate = judged.get(key)
            if estimate is not None and estimate.value is not None:
                rows.append(row(f"judge.{key}", float(estimate.value), estimate.n))
    return rows


def run_online_evals(
    engine: Engine,
    runs: Sequence[ScoredRun],
    *,
    router: ModelRouter | None,
    tracer: Tracer,
    run_id: int,
    today: date,
    deadline: float,
    clock: Callable[[], float],
) -> OnlineOutcome:
    judged_items = 0
    judge_failed = router is None
    every: list[MetricRow] = []
    drafted = shipped = basic = 0
    for run in runs:
        verdicts: dict[str, bool] = {}
        if router is not None and clock() < deadline:
            try:
                verdicts = judge(
                    judge_items(_stored(run)), router, stop=lambda: clock() >= deadline
                ).verdicts
            except Exception as error:  # the verifier's numbers still count
                judge_failed = True
                log.warning("online judge: %s failed (%s)", run.report_id, type(error).__name__)
        judged_items += len(verdicts)
        rows = report_metrics(run, verdicts)
        every += rows
        for row in rows:
            if row.value is not None:
                tracer.score(run.trace_id, SCORE_PREFIX + row.metric, row.value)
        stats = claim_stats(run.record)
        if stats is not None:
            drafted += stats.drafted
            shipped += stats.shipped
        result = run.record.result
        basic += int(run.record.basic or (isinstance(result, Report) and result.basic))
    night = [
        MetricRow("online", "reports", "night", today, today, float(len(runs)), None, None, None),
        MetricRow("online", "gemini_errors", "night", today, today, float(basic), None, None, None),
        MetricRow(
            "online",
            "pre_verifier_pass_rate",
            "night",
            today,
            today,
            shipped / drafted if drafted else None,
            None,
            None,
            drafted,
        ),
    ]
    with engine.begin() as connection:
        write_metrics(connection, [*every, *night], run_id=run_id)
    log.info(
        "online evals: %d reports, %d items judged%s",
        len(runs),
        judged_items,
        " (the judge failed)" if judge_failed and runs else "",
    )
    return OnlineOutcome(reports=len(runs), judged_items=judged_items, judge_failed=judge_failed)


def verifier_outcome(connection: Connection, rule: AlertRule) -> Outcome:
    """The last 10 reports' pooled pre-verifier pass rate against the rule's floor."""
    rows = connection.execute(text(_LAST_REPORTS), {"n": VERIFIER_REPORTS}).all()
    if len(rows) < VERIFIER_REPORTS or rule.threshold is None:
        return Outcome(rule.id, None, f"verifier: {len(rows)} reports so far")
    drafted = sum(int(r[1]) for r in rows)
    shipped = sum(float(r[0]) * int(r[1]) for r in rows)
    rate = shipped / drafted
    body = (
        f"pre_verifier_pass_rate {rate:.3f} over the last {len(rows)} reports "
        f"({drafted} drafted claims), floor {rule.threshold:.2f}"
    )
    return Outcome(rule.id, rate < rule.threshold, body)


def langfuse_outcome(units_used: int | None, rule: AlertRule) -> Outcome:
    if units_used is None or rule.threshold is None:
        return Outcome(rule.id, None, "langfuse: units not read")
    body = f"{units_used} Langfuse units this month, threshold {rule.threshold:.0f}"
    return Outcome(rule.id, units_used >= rule.threshold, body)


def run_live_online_evals(
    engine: Engine,
    settings: Settings,
    runs: Sequence[ScoredRun],
    *,
    run_id: int,
    today: date,
    clock: Callable[[], float],
    deadline: float,
) -> OnlineOutcome:
    """The nightly stage: the judge router, and the reports stage's tracer and its month count."""
    import tempfile
    from contextlib import ExitStack
    from dataclasses import replace

    from patchpulse.agent.tracing import LangfuseTracer
    from patchpulse.pipeline.reports import _search, _tracer

    def now() -> datetime:
        return datetime.now(UTC)

    with ExitStack() as stack:
        http = stack.enter_context(httpx2.Client())
        scratch = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        router = (
            online_judge_router(
                settings,
                http,
                state_path=scratch / "gemini-requests.json",
                cache_path=scratch / "judge.sqlite",
                clock=now,
            )
            if settings.gemini_api_key is not None
            else None
        )
        tracer = _tracer(stack, settings, _search(stack, settings))
        outcome = run_online_evals(
            engine, runs, router=router, tracer=tracer, run_id=run_id, today=today,
            deadline=deadline, clock=clock,
        )  # fmt: skip
        units = tracer.units.used() if isinstance(tracer, LangfuseTracer) else None
        return replace(outcome, units_used=units)
