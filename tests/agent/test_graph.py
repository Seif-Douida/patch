"""The workflow graph (plan Task 10, design D4): scope, core evidence, rules, the model's tool
calls, draft, verify, summary, finalize; on the fixture bundle with a fake Qdrant, a fake Steam
and scripted model answers through the router and its record/replay cache."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import numpy as np
import pytest
from agent_fixtures import GAME, PATCH_DAY, make_bundle
from fake_clock import FakeClock
from numpy.typing import NDArray

from patchpulse.agent.graph import (
    MAX_MODEL_TOOL_CALLS,
    AgentDeps,
    Refusal,
    ReportRequest,
    run_report,
)
from patchpulse.agent.llm_cache import ReplayCache
from patchpulse.agent.prompts import PROMPT_NAMES, PROMPT_VERSION, load_prompt
from patchpulse.agent.report import Report
from patchpulse.agent.router import ModelRouter
from patchpulse.agent.search import ReviewSearch
from patchpulse.agent.tools import Tools, Trace
from patchpulse.ingest.http import RateLimiter, RetryPolicy, SteamHttp
from patchpulse.labeling.llm import LLMResponse, TeacherUnavailableError

CHAIN = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
CRASHES = "The game crashes on load since the patch and the frame rate halved in every mission."
WINDOW = (PATCH_DAY, PATCH_DAY + timedelta(days=6))
BASELINE = (PATCH_DAY - timedelta(days=28), PATCH_DAY - timedelta(days=1))


def performance_change() -> float:
    tools = Tools(make_bundle(), Trace())
    change: float = tools.aspect_breakdown(GAME, WINDOW, BASELINE)["aspects"]["performance"][
        "negative"
    ]["change"]
    return change


def scope_answer(_: str) -> dict[str, Any]:
    return {"appid": GAME, "start": "2026-09-10", "end": "2026-09-16", "patch_id": "p1"}


def gather_answer(_: str) -> dict[str, Any]:
    searches = [
        {"tool": "search_reviews", "query": f"crashes {i}", "period": "window",
         "voted_up": False, "reason": "stability"}
        for i in range(11)
    ]  # fmt: skip
    return {"calls": [{"tool": "get_patch_notes", "patch_id": "p1", "reason": "notes"}, *searches]}


def call_id(user: str, tool: str) -> str:
    found = re.search(rf"\[(\S+)\] {tool}\b", user)
    assert found, f"no {tool} result in the draft prompt"
    return found.group(1)


def draft_answer(user: str) -> dict[str, Any]:
    aspects, estimate = call_id(user, "aspect_breakdown"), call_id(user, "estimate_patch_effect")
    assert '<review id="101">' in user
    return {
        "findings": [
            {"title": "The patch", "hypothesis": "H1", "claim_ids": ["c2"]},
            {"title": "Performance", "hypothesis": "H3", "claim_ids": ["c1", "c3", "c4"]},
        ],
        "claims": [
            {"claim_id": "c1", "text": "Performance mentions among negative reviews changed.",
             "claim_type": "comparative", "value": performance_change(), "unit": "pp",
             "evidence": {"call_id": aspects,
                          "json_path": "aspects.performance.negative.change"}},
            {"claim_id": "c2", "text": "The patch led to a 12-point fall in the positive share.",
             "claim_type": "causal", "value": -12.25, "unit": "pp",
             "evidence": {"call_id": estimate, "json_path": "horizons.7.effect"}},
            {"claim_id": "c3", "text": "A negative review reports crashes on load.",
             "claim_type": "descriptive", "review_ids": [101],
             "quote_spans": [{"review_id": 101, "text": "crashes on load"}]},
            {"claim_id": "c4", "text": "Performance complaints rose by 40 points.",
             "claim_type": "comparative", "value": 40.0, "unit": "pp",
             "evidence": {"call_id": aspects,
                          "json_path": "aspects.performance.negative.change"}},
        ],
    }  # fmt: skip


def summary_answer(_: str) -> dict[str, Any]:
    change = performance_change()
    return {"summary": f"Performance mentions among negative reviews moved {change:.1f} points."}


ANSWERS: dict[str, Callable[[str], dict[str, Any]]] = {
    "scope": scope_answer,
    "gather": gather_answer,
    "draft": draft_answer,
    "summary": summary_answer,
}


class Scripted:
    """A model that answers each step from its script; the step is known by its system prompt."""

    def __init__(self, answers: dict[str, Callable[[str], dict[str, Any]]] | None = None) -> None:
        self.answers = answers or ANSWERS
        self.steps: list[str] = []
        self.by_prompt = {load_prompt(name).text: name for name in PROMPT_NAMES}

    def generate(self, system: str, user: str, *, json_schema: Any, **_: Any) -> LLMResponse:
        step = self.by_prompt[system]
        self.steps.append(step)
        return LLMResponse(json.dumps(self.answers[step](user)), 900, 120, "STOP")


class Down:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, system: str, user: str, *, json_schema: Any, **_: Any) -> LLMResponse:
        self.calls += 1
        raise TeacherUnavailableError(503, "overloaded")


class FakeEmbedder:
    def embed_queries(self, texts: list[str], *, batch: int = 32) -> NDArray[np.float32]:
        return np.ones((len(texts), 4), dtype=np.float32)


class FakeQdrant:
    def __init__(self) -> None:
        self.calls = 0

    def query_points(self, collection: str, **kwargs: Any) -> Any:
        self.calls += 1
        hits = [(101, 0.9, CRASHES), (102, 0.4, "Servers are stable now.")]
        return SimpleNamespace(
            points=[SimpleNamespace(id=i, score=s, payload={"text": t}) for i, s, t in hits]
        )


class World:
    def __init__(self, tmp_path: Path, model: Any, *, mode: str = "live") -> None:
        self.qdrant = FakeQdrant()
        self.steam_requests = 0

        def steam(request: httpx2.Request) -> httpx2.Response:
            self.steam_requests += 1
            item = {
                "gid": "p1",
                "title": "Patch 1.1.0",
                "contents": "[list][*]Fixed crashes[/list]",
            }
            return httpx2.Response(200, json={"appnews": {"newsitems": [item]}})

        clock = FakeClock(datetime(2026, 10, 11, 3, 0, tzinfo=UTC))
        self.router = ModelRouter(
            CHAIN,
            {CHAIN[0]: model, CHAIN[1]: model},
            ReplayCache(tmp_path / "answers.sqlite", mode),  # type: ignore[arg-type]
            clock=clock,
        )
        self.deps = AgentDeps(
            bundle=make_bundle(),
            router=self.router,
            search=ReviewSearch(self.qdrant, "reviews", FakeEmbedder()),
            steam=SteamHttp(
                httpx2.Client(transport=httpx2.MockTransport(steam)),
                limiter=RateLimiter(min_interval_s=0.0),
                policy=RetryPolicy(max_attempts=1),
            ),
        )

    def run(self, request: ReportRequest) -> Report | Refusal:
        return run_report(request, self.deps)


def report_of(result: Report | Refusal) -> Report:
    assert isinstance(result, Report), result
    return result


def test_a_nightly_report_runs_every_step_without_a_scope_call(tmp_path: Path) -> None:
    model = Scripted()

    report = report_of(World(tmp_path, model).run(ReportRequest.for_bundle("p1", 7)))

    assert model.steps == ["gather", "draft", "summary"]
    assert (report.scope, report.bundle_id, report.horizon, report.question) == (
        "bundle",
        "p1",
        7,
        None,
    )
    assert (report.window.start, report.window.end) == WINDOW
    assert [h.code for h in report.hypotheses] == ["H1", "H2", "H3", "H4", "H5"]
    assert not report.basic
    verified = {c.claim_id: c for c in report.claims}
    assert set(verified) == {"c1", "c2", "c3"}
    assert verified["c2"].reworded  # the 7-day estimate has a volume spike, no verdict yet
    assert [(w.claim.claim_id, w.rule) for w in report.withheld_claims] == [("c4", "V2")]
    assert report.summary.startswith("Performance mentions")
    assert report.confidence == "medium"


def test_an_ask_report_scopes_then_runs(tmp_path: Path) -> None:
    model = Scripted()

    report = report_of(
        World(tmp_path, model).run(
            ReportRequest.for_question("Why did Helldivers 2 reviews drop after patch 1.1.0?")
        )
    )

    assert model.steps == ["scope", "gather", "draft", "summary"]
    assert (report.scope, report.bundle_id, report.horizon) == ("question", "p1", 7)
    assert report.question == "Why did Helldivers 2 reviews drop after patch 1.1.0?"
    assert (report.window.start, report.window.end) == WINDOW


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        ({"refusal": "That isn't about a game's reviews."}, "That isn't about a game's reviews."),
        ({"appid": 1086940, "start": "2026-09-10", "end": "2026-09-16"}, "isn't one of the games"),
        ({"appid": GAME, "start": "2026-11-01", "end": "2026-11-07"}, "outside the data"),
        ({"appid": GAME, "start": "2026-09-16", "end": "2026-09-10"}, "ends before it starts"),
        ({"appid": GAME, "start": "2026-09-10", "end": "2026-09-16", "patch_id": "zz"}, "patch"),
        ({"appid": GAME}, "dates"),
    ],
)
def test_a_question_the_scope_cant_place_is_refused(
    tmp_path: Path, answer: dict[str, Any], reason: str
) -> None:
    model = Scripted({**ANSWERS, "scope": lambda _: answer})

    result = World(tmp_path, model).run(ReportRequest.for_question("Why did reviews drop?"))

    assert isinstance(result, Refusal)
    assert reason in result.reason
    assert model.steps == ["scope"]  # at most the scope call spent


def test_an_empty_or_long_question_is_refused_without_a_model_call(tmp_path: Path) -> None:
    model = Scripted()
    world = World(tmp_path, model)

    for question in ("", "   ", "x" * 1000):
        assert isinstance(world.run(ReportRequest.for_question(question)), Refusal)
    assert model.steps == []


def test_the_model_gets_at_most_eight_tool_calls(tmp_path: Path) -> None:
    world = World(tmp_path, Scripted())

    world.run(ReportRequest.for_bundle("p1", 7))

    assert MAX_MODEL_TOOL_CALLS == 8
    assert world.steam_requests == 1  # the patch notes, asked for first
    assert world.qdrant.calls == MAX_MODEL_TOOL_CALLS - 1  # 11 searches asked for, 7 run


def test_all_models_down_gives_a_basic_report(tmp_path: Path) -> None:
    down = Down()

    report = report_of(World(tmp_path, down).run(ReportRequest.for_bundle("p1", 7)))

    assert report.basic
    assert down.calls == 2  # gather, on both models; nothing more is asked of them
    assert report.model_ids == {}
    assert report.claims
    assert all(c.claim_id.startswith("b") for c in report.claims)
    assert any(c.value is not None for c in report.claims)
    assert [h.code for h in report.hypotheses] == ["H1", "H2", "H3", "H4", "H5"]
    assert "Supported" in report.summary or "Ruled out" in report.summary
    assert report.withheld_claims == []


def test_a_question_with_every_model_down_is_refused(tmp_path: Path) -> None:
    result = World(tmp_path, Down()).run(ReportRequest.for_question("Why did reviews drop?"))

    assert isinstance(result, Refusal)
    assert "unavailable" in result.reason


def test_report_records_versions_and_pass_rates(tmp_path: Path) -> None:
    report = report_of(World(tmp_path, Scripted()).run(ReportRequest.for_bundle("p1", 7)))

    assert report.prompt_version == PROMPT_VERSION
    assert report.data_version == 52
    assert report.model_ids == {"gather": CHAIN[0], "draft": CHAIN[0], "summary": CHAIN[0]}
    assert report.report_id == "p1-7d"
    for rule in ("V1", "V2", "V3", "V4", "V5", "V6", "V7"):
        assert rule in report.rule_pass_rates, rule
    assert report.rule_pass_rates["V2"] == pytest.approx(2 / 3)  # c1, c2 pass; c4's 40 doesn't
    assert report.rule_pass_rates["V6"] == 1.0


def test_a_failed_summary_gets_the_template(tmp_path: Path) -> None:
    model = Scripted({**ANSWERS, "summary": lambda _: {"summary": "Reviews fell 77 points."}})

    report = report_of(World(tmp_path, model).run(ReportRequest.for_bundle("p1", 7)))

    assert "77" not in report.summary
    assert report.rule_pass_rates["V6"] == 0.0
    assert not report.basic


def test_a_recorded_run_replays_identically(tmp_path: Path) -> None:
    recorded = report_of(
        World(tmp_path, Scripted(), mode="record").run(ReportRequest.for_bundle("p1", 7))
    )

    replayed = report_of(
        World(tmp_path, Down(), mode="replay").run(ReportRequest.for_bundle("p1", 7))
    )

    assert replayed == recorded
