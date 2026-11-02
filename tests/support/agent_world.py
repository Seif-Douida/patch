"""The graph tests' world (Phase 5 plan Tasks 10 and 18): the fixture bundle, a fake Qdrant whose
hits carry invented review text, a fake Steam for patch notes, and scripted model answers through
the router and its record/replay cache."""

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
from agent_fixtures import GAME, PATCH_DAY, make_bundle
from fake_clock import FakeClock
from numpy.typing import NDArray

from patchpulse.agent.graph import AgentDeps, Refusal, ReportRequest, run_report
from patchpulse.agent.llm_cache import ReplayCache
from patchpulse.agent.prompts import PROMPT_NAMES, load_prompt
from patchpulse.agent.report import Report
from patchpulse.agent.router import ModelRouter
from patchpulse.agent.search import ReviewSearch
from patchpulse.agent.tools import Tools, Trace
from patchpulse.agent.tracing import NoTracer, Tracer
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
    def __init__(
        self, tmp_path: Path, model: Any, *, mode: str = "live", tracer: Tracer | None = None
    ) -> None:
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
            tracer=tracer or NoTracer(),
        )

    def run(self, request: ReportRequest) -> Report | Refusal:
        return run_report(request, self.deps)


def report_of(result: Report | Refusal) -> Report:
    assert isinstance(result, Report), result
    return result
