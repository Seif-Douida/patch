"""Phase 7 Task 7: the online evals of the night's reports (design D4)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any

import pytest
from eval_fixtures import aspects_call, claim, hypotheses
from eval_fixtures import report as make_report
from sqlalchemy import Engine

from patchpulse.agent.graph import RunRecord
from patchpulse.agent.llm_cache import ReplayCache
from patchpulse.agent.report import Claim, EvidenceRef, WithheldClaim
from patchpulse.agent.router import AllModelsDownError, ModelRouter
from patchpulse.agent.tools import Trace
from patchpulse.labeling.llm import LLMResponse
from patchpulse.monitoring.alerts import load_rules
from patchpulse.monitoring.store import MetricRow, write_metrics
from patchpulse.pipeline.online_eval import (
    langfuse_outcome,
    report_metrics,
    run_online_evals,
    verifier_outcome,
)
from patchpulse.pipeline.reports import ScoredRun

SENTINEL_CLAIM = "SENTINEL-CLAIM players say the servers melted after the patch"
SENTINEL_REVIEW = "SENTINEL-REVIEW the servers melted and the devs went silent for days"
TODAY = date(2026, 11, 23)


def quoting(claim_id: str, *review_ids: int) -> Claim:
    return claim(claim_id, SENTINEL_CLAIM).model_copy(update={"review_ids": list(review_ids)})


def figure(claim_id: str) -> Claim:
    return claim(claim_id, "Policy complaints rose 2 points.").model_copy(
        update={
            "value": 2.0,
            "unit": "pp",
            "evidence": EvidenceRef(
                call_id="r.1", json_path="aspects.platform_policy.negative.change"
            ),
        }
    )


def scored_run(report_id: str = "p1-7d", trace_id: str | None = "t1") -> ScoredRun:
    withheld = (WithheldClaim(claim=claim("c3", "Sales doubled."), rule="V2"),)
    result = make_report(
        hypotheses(("H1",)), claims=(quoting("c1", 101), figure("c2")), withheld=withheld
    )
    result = result.model_copy(update={"report_id": report_id})
    trace = Trace(calls=[aspects_call(2.0, 1.0)], retrieved={101: SENTINEL_REVIEW})
    record = RunRecord(result, None, result.hypotheses, None, None, trace, False, trace_id)
    return ScoredRun(report_id, record, trace_id)


def now() -> datetime:
    return datetime(2026, 11, 23, 3, tzinfo=UTC)


class FakeJudge:
    def __init__(self, supported: bool = True, *, down: bool = False) -> None:
        self.supported = supported
        self.down = down
        self.calls = 0

    def generate(
        self, system: str, user: str, *, json_schema: Mapping[str, Any] | None, **_: Any
    ) -> LLMResponse:
        self.calls += 1
        if self.down:
            raise AllModelsDownError("no model answered")
        return LLMResponse(
            json.dumps({"supported": self.supported, "reason": "r"}), 100, 10, "STOP"
        )


def router(judge: FakeJudge, tmp_path: Any) -> ModelRouter:
    return ModelRouter(
        ["judge"], {"judge": judge}, ReplayCache(tmp_path / "c.sqlite", "live"), clock=now
    )


class Scores:
    def __init__(self) -> None:
        self.sent: list[tuple[str | None, str, float]] = []

    def start(self, *_: Any, **__: Any) -> Any:
        raise AssertionError("the online evals never start a trace")

    def score(self, trace_id: str | None, name: str, value: float) -> None:
        self.sent.append((trace_id, name, value))

    def close(self) -> None:
        return None


def test_report_metrics_count_the_verifier_and_the_judge() -> None:
    rows = {r.metric: r for r in report_metrics(scored_run(), {"online/p1-7d/c1/r101": False})}

    assert rows["pre_verifier_pass_rate"].value == pytest.approx(2 / 3)
    assert rows["pre_verifier_pass_rate"].n == 3
    assert rows["withheld.V2"].value == 1
    assert rows["reworded"].value == 0
    assert rows["judge.citation_support"].value == pytest.approx(0.0)
    assert all(r.family == "online" and r.scope == "report:p1-7d" for r in rows.values())


@pytest.mark.integration
def test_verifier_numbers_are_stored_even_when_the_judge_is_down(
    migrated_engine: Engine, tmp_path: Any
) -> None:
    outcome = run_online_evals(
        migrated_engine, [scored_run()], router=router(FakeJudge(down=True), tmp_path),
        tracer=Scores(), run_id=5, today=TODAY, deadline=float("inf"), clock=lambda: 0.0,
    )  # fmt: skip

    assert outcome.reports == 1
    assert outcome.judged_items == 0
    from patchpulse.monitoring.store import read_metrics

    with migrated_engine.connect() as connection:
        rows = {r.metric for r in read_metrics(connection, family="online", scope="report:p1-7d")}
    assert "pre_verifier_pass_rate" in rows
    assert "judge.citation_support" not in rows


@pytest.mark.integration
def test_scores_reach_the_reports_trace_and_names_say_judge(
    migrated_engine: Engine, tmp_path: Any
) -> None:
    tracer = Scores()
    run_online_evals(
        migrated_engine, [scored_run(trace_id="t9")], router=router(FakeJudge(), tmp_path),
        tracer=tracer, run_id=5, today=TODAY, deadline=float("inf"), clock=lambda: 0.0,
    )  # fmt: skip

    names = {name for trace_id, name, _ in tracer.sent if trace_id == "t9"}
    assert "online.pre_verifier_pass_rate" in names
    assert "online.judge.citation_support" in names
    assert all(trace_id == "t9" for trace_id, _, _ in tracer.sent)


@pytest.mark.integration
def test_online_metrics_hold_no_text(migrated_engine: Engine, tmp_path: Any) -> None:
    tracer = Scores()
    run_online_evals(
        migrated_engine, [scored_run()], router=router(FakeJudge(), tmp_path),
        tracer=tracer, run_id=5, today=TODAY, deadline=float("inf"), clock=lambda: 0.0,
    )  # fmt: skip

    from sqlalchemy import text

    with migrated_engine.connect() as connection:
        stored = connection.execute(text("SELECT * FROM ops.eval_metric")).all()
    everything = " ".join(str(v) for row in stored for v in row) + " ".join(map(str, tracer.sent))
    assert "SENTINEL" not in everything


def report_row(rid: int, rate: float, drafted: int = 10) -> MetricRow:
    return MetricRow(
        "online",
        "pre_verifier_pass_rate",
        f"report:r{rid}",
        TODAY,
        TODAY,
        rate,
        None,
        None,
        drafted,
    )


@pytest.mark.integration
def test_the_verifier_alert_needs_ten_reports_and_breaches_below_eighty_percent(
    migrated_engine: Engine,
) -> None:
    rule = load_rules()["verifier_dropped"]
    with migrated_engine.begin() as connection:
        write_metrics(connection, [report_row(i, 0.7) for i in range(9)], run_id=1)
        assert verifier_outcome(connection, rule).breached is None
        write_metrics(connection, [report_row(9, 0.7)], run_id=2)
        assert verifier_outcome(connection, rule).breached is True
        write_metrics(connection, [report_row(i, 0.95) for i in range(10, 20)], run_id=3)
        assert verifier_outcome(connection, rule).breached is False


def test_langfuse_units_breach_at_forty_thousand() -> None:
    rule = load_rules()["langfuse_units"]
    assert langfuse_outcome(39_999, rule).breached is False
    assert langfuse_outcome(40_000, rule).breached is True
    assert langfuse_outcome(None, rule).breached is None


def test_the_judge_never_raises_an_alert() -> None:
    rules = load_rules()
    judged = [r for r in rules.values() if "judge" in r.id or "judge" in r.title.lower()]
    assert judged == []


class StallingJudge(FakeJudge):
    """Answers, but each answer takes `seconds` on its own clock (Gemma accepting and stalling)."""

    def __init__(self, seconds: float) -> None:
        super().__init__()
        self.now = 0.0
        self.seconds = seconds

    def generate(self, system: str, user: str, **kwargs: Any) -> LLMResponse:
        self.now += self.seconds
        return super().generate(system, user, **kwargs)


@pytest.mark.integration
def test_a_stalled_judge_stops_at_the_deadline(migrated_engine: Engine, tmp_path: Any) -> None:
    # Final review C1 (18 Nov): the deadline was checked only before each report, so a stalled
    # judge spent every item of a report past it.
    judge = StallingJudge(seconds=100.0)

    outcome = run_online_evals(
        migrated_engine, [scored_run(), scored_run("p2-7d")], router=router(judge, tmp_path),
        tracer=Scores(), run_id=5, today=TODAY, deadline=50.0, clock=lambda: judge.now,
    )  # fmt: skip

    assert judge.calls == 1
    assert outcome.reports == 2  # the verifier's numbers still count for both


def test_the_online_judge_never_retries_a_stalled_request(tmp_path: Any) -> None:
    from pydantic import SecretStr

    from patchpulse.config import Settings
    from patchpulse.pipeline.online_eval import online_judge_router

    settings = Settings.model_construct(gemini_api_key=SecretStr("k-test"))
    judge = online_judge_router(
        settings, None, state_path=tmp_path / "s.json", cache_path=tmp_path / "c.sqlite",  # type: ignore[arg-type]
        clock=now,
    )  # fmt: skip

    (client,) = judge.clients.values()
    assert client.retry_errors is False  # type: ignore[attr-defined]
