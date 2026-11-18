"""The Ask harness (Phase 5 plan Task 16): the API with scripted models, a fake review search and
in-memory limits. Shared by the Ask tests and Phase 7's telemetry tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from agent_fixtures import GAME, make_bundle, sample_report
from api_fixtures import Published, Seconds
from fake_clock import FakeClock
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from patchpulse.agent.bundle import Bundle
from patchpulse.agent.graph import AgentDeps
from patchpulse.agent.llm_cache import ReplayCache
from patchpulse.agent.prompts import PROMPT_NAMES, load_prompt
from patchpulse.agent.router import ModelRouter
from patchpulse.agent.search import ReviewSearch
from patchpulse.api.app import create_app
from patchpulse.api.bundle_source import BundleSource
from patchpulse.api.limits import AskLimits, LimitStore, MemoryLimitStore
from patchpulse.api.settings import ApiSettings
from patchpulse.labeling.llm import LLMResponse

CHAIN = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
NOW = datetime(2026, 10, 25, 12, tzinfo=UTC)
QUOTED = "The game crashes on load since the patch and every mission stutters."
REVIEW_ID = 987654321
Answer = Callable[[str], dict[str, Any]]


def scope(start: str = "2026-09-10", end: str = "2026-09-16") -> Answer:
    return lambda _: {"appid": GAME, "start": start, "end": end, "patch_id": "p1"}


def draft(user: str) -> dict[str, Any]:
    claim = {
        "claim_id": "c1",
        "text": "A negative review reports crashes on load after the patch.",
        "claim_type": "descriptive",
        "review_ids": [REVIEW_ID],
        "quote_spans": [{"review_id": REVIEW_ID, "text": "crashes on load"}],
    }
    return {"findings": [{"title": "Crashes", "hypothesis": "H1", "claim_ids": ["c1"]}],
            "claims": [claim]}  # fmt: skip


ANSWERS: dict[str, Answer] = {
    "scope": scope(),
    "gather": lambda _: {
        "calls": [
            {
                "tool": "search_reviews",
                "query": "crashes",
                "period": "window",
                "reason": "stability",
            }
        ]
    },
    "draft": draft,
    "summary": lambda _: {"summary": "Some negative reviews report crashes after the patch."},
}


class Scripted:
    def __init__(self, **answers: Answer) -> None:
        self.answers = {**ANSWERS, **answers}
        self.steps: list[str] = []
        self.by_prompt = {load_prompt(name).text: name for name in PROMPT_NAMES}

    def generate(self, system: str, user: str, *, json_schema: Any, **_: Any) -> LLMResponse:
        step = self.by_prompt[system]
        self.steps.append(step)
        return LLMResponse(json.dumps(self.answers[step](user)), 900, 120, "STOP")


class FakeEmbedder:
    def embed_queries(self, texts: list[str], *, batch: int = 32) -> NDArray[np.float32]:
        return np.ones((len(texts), 4), dtype=np.float32)


class FakeQdrant:
    def __init__(self, text: str = QUOTED) -> None:
        self.text = text

    def query_points(self, collection: str, **kwargs: Any) -> Any:
        point = SimpleNamespace(id=REVIEW_ID, score=0.9, payload={"text": self.text})
        return SimpleNamespace(points=[point])


class Ask:
    """The API with scripted models, a fake review search and in-memory limits."""

    def __init__(
        self,
        tmp_path: Path,
        model: Scripted,
        *,
        daily_cap: int = 15,
        qdrant: Any = None,
        store: LimitStore | None = None,
    ) -> None:
        self.model = model
        self.published, self.clock = Published(), Seconds()
        self.published.publish(make_bundle(), sample_report("p1", 7))

        def deps(bundle: Bundle) -> AgentDeps:
            router = ModelRouter(
                CHAIN,
                {CHAIN[0]: model, CHAIN[1]: model},
                ReplayCache(tmp_path / "unused.sqlite", "live"),
                clock=FakeClock(NOW),
            )
            search = ReviewSearch(qdrant or FakeQdrant(), "reviews", FakeEmbedder())
            return AgentDeps(bundle=bundle, router=router, search=search)

        limits = AskLimits(
            store or MemoryLimitStore(),
            secret=b"test",
            per_visitor_hour=3,
            daily_cap=daily_cap,
            clock=FakeClock(NOW),
        )
        app = create_app(
            ApiSettings(), BundleSource(self.published, clock=self.clock), limits=limits,
            deps_for=deps,
        )  # fmt: skip
        self.client = TestClient(app)

    def ask(self, question: str, *, appid: int = GAME, visitor: str = "203.0.113.7") -> Any:
        return self.client.post(
            "/ask",
            json={"appid": appid, "question": question},
            headers={"X-Forwarded-For": f"10.0.0.1, {visitor}"},
        )
