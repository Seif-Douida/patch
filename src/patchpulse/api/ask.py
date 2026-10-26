"""What POST /ask needs (Phase 5, design D9): the request, the answer cache, the visitor's
address, and the live agent and limits built from the API's settings.

The agent's models, review search, Steam client and tracer are built on the first question, not at
start, so /health and /reports never wait for the encoder to load. Ask's runs are traced at the
settings' sample (agent/tracing.py); the SDK sends what's left when the process exits. The
visitors' HMAC key is derived from the Gemini key, a secret the API already holds: rotating that
key only resets the counters.
"""

from __future__ import annotations

import hashlib
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
from fastapi import Request
from pydantic import BaseModel, Field, SecretStr

from patchpulse.agent.bundle import Bundle
from patchpulse.agent.graph import AgentDeps
from patchpulse.agent.llm_cache import AGENT_CACHE_FILE, ReplayCache
from patchpulse.agent.router import ModelRouter, make_agent_clients
from patchpulse.agent.search import ReviewSearch, open_pinned_search
from patchpulse.agent.tracing import Tracer, make_tracer
from patchpulse.api.limits import AskLimits, LimitStore, MemoryLimitStore, QdrantLimitStore
from patchpulse.api.settings import ApiSettings
from patchpulse.ingest.http import RateLimiter, SteamHttp
from patchpulse.labeling.teachers import load_model_config

ANSWERS_KEPT = 256
UNAVAILABLE = "Ask is unavailable right now; the nightly reports are still here."

CacheKey = tuple[int, int, str]  # data version, game, normalised question


class AskRequest(BaseModel):
    appid: int
    question: str = Field(max_length=2_000)  # the agent refuses past 500; this bounds the body


def normalise(question: str) -> str:
    return " ".join(question.casefold().split())


class AnswerCache:
    """The latest answers by data version, game and question, in memory (gone at scale-to-zero,
    when the next question would be a new wake anyway)."""

    def __init__(self, kept: int = ANSWERS_KEPT) -> None:
        self.kept = kept
        self._answers: OrderedDict[CacheKey, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: CacheKey) -> dict[str, Any] | None:
        with self._lock:
            answer = self._answers.get(key)
            if answer is not None:
                self._answers.move_to_end(key)
            return answer

    def put(self, key: CacheKey, answer: dict[str, Any]) -> None:
        with self._lock:
            self._answers[key] = answer
            self._answers.move_to_end(key)
            while len(self._answers) > self.kept:
                self._answers.popitem(last=False)


def visitor_address(request: Request) -> str:
    """The address the ingress saw: the last X-Forwarded-For entry (the one Container Apps adds;
    earlier ones are whatever the client sent), else the connection's."""
    forwarded = request.headers.get("x-forwarded-for", "")
    entries = [e.strip() for e in forwarded.split(",") if e.strip()]
    if entries:
        return entries[-1]
    return request.client.host if request.client else "unknown"


def _now() -> datetime:
    return datetime.now(UTC)


class LiveDeps:
    """The agent's Gemini chain, the pinned review search and Steam, built once on first use."""

    def __init__(self, settings: ApiSettings, gemini_key: SecretStr) -> None:
        self.settings = settings
        self.gemini_key = gemini_key
        self._lock = threading.Lock()
        self._built: tuple[ModelRouter, ReviewSearch | None, SteamHttp, Tracer] | None = None

    def _build(self) -> tuple[ModelRouter, ReviewSearch | None, SteamHttp, Tracer]:
        settings = self.settings
        models = load_model_config()
        if models.agent is None:
            raise ValueError("config/models.yaml has no agent section")
        # The free tier's per-day counts, for this replica only; Google's own 429 covers the rest.
        state = Path(tempfile.mkdtemp(prefix="pp-gemini-")) / "gemini-requests.json"
        clients = make_agent_clients(
            models,
            self.gemini_key.get_secret_value(),
            httpx2.Client(),
            state_path=state,
            clock=_now,
            sleep=time.sleep,
        )
        cache = ReplayCache(Path(AGENT_CACHE_FILE), "live")
        router = ModelRouter(models.agent.chain, clients, cache, clock=_now)
        search = None
        if settings.qdrant_url is not None and settings.encoder_dir is not None:
            search = open_pinned_search(
                settings.qdrant_url.get_secret_value(), settings.qdrant_api_key,
                settings.encoder_dir,
            )  # fmt: skip
        # The month's Langfuse units are counted with Ask's limits, in the Qdrant cluster.
        store: LimitStore = MemoryLimitStore()
        if search is not None:
            store = QdrantLimitStore(search.client)
        tracer = make_tracer(
            settings.langfuse_public_key,
            settings.langfuse_secret_key,
            settings.langfuse_host,
            store=store,
            ask_sample=settings.ask_trace_sample,
        )
        return router, search, SteamHttp(httpx2.Client(), limiter=RateLimiter()), tracer

    def __call__(self, bundle: Bundle) -> AgentDeps:
        with self._lock:
            if self._built is None:
                self._built = self._build()
            router, search, steam, tracer = self._built
        return AgentDeps(bundle=bundle, router=router, search=search, steam=steam, tracer=tracer)


def live_deps(settings: ApiSettings) -> Callable[[Bundle], AgentDeps] | None:
    key = settings.gemini_api_key
    return None if key is None else LiveDeps(settings, key)


def live_limits(settings: ApiSettings) -> AskLimits | None:
    """Counters in the Qdrant cluster when there is one, else in memory (local runs)."""
    if settings.gemini_api_key is None:
        return None
    from patchpulse.retrieval.qdrant import make_client

    key = settings.gemini_api_key.get_secret_value().encode()
    secret = hashlib.sha256(b"patchpulse-ask-visitors|" + key).digest()
    store: LimitStore = MemoryLimitStore()
    if settings.qdrant_url is not None:
        client = make_client(settings.qdrant_url.get_secret_value(), settings.qdrant_api_key)
        store = QdrantLimitStore(client)
    return AskLimits(
        store,
        secret=secret,
        per_visitor_hour=settings.ask_per_visitor_hour,
        daily_cap=settings.ask_daily_cap,
    )
