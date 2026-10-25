"""FastAPI entry point (`uvicorn patchpulse.api.app:app`).

- `/health`: liveness only; it never touches SQL or the agent's data.
- `/reports/{patch_id}` (Phase 5, design D9): the public copy of the report for the patch's bundle,
  from the night's published data; 404 without one, 503 until the data has loaded.
- `POST /ask` ({appid, question}): a visitor's question about one tracked game, answered by the
  agent on the loaded bundle (one bundle for the whole run). Refusals come back in plain words as
  `refusal`; an answer as the report's public copy. Answers are cached by data version, game and
  question; only questions that run the agent count against Ask's limits (429 with the reason).

Interactive docs stay off: every request wakes a replica. Browsers may read the API from the
site's origin only (PP_SITE_ORIGIN; none set, none allowed).
"""

from __future__ import annotations

import os
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware

from patchpulse.agent.bundle import Bundle
from patchpulse.agent.graph import (
    UNAVAILABLE as MODELS_DOWN,
)
from patchpulse.agent.graph import (
    AgentDeps,
    Refusal,
    ReportRequest,
    question_precheck,
    run_report,
)
from patchpulse.agent.report import public_copy
from patchpulse.api.ask import (
    UNAVAILABLE,
    AnswerCache,
    AskRequest,
    live_deps,
    live_limits,
    normalise,
    visitor_address,
)
from patchpulse.api.bundle_source import BundleSource, http_fetch
from patchpulse.api.limits import AskLimits, Refused
from patchpulse.api.settings import ApiSettings

NOT_LOADED = "data not available yet"
PATCH_ID = r"^[0-9A-Za-z_-]{1,40}$"


def create_app(
    settings: ApiSettings | None = None,
    source: BundleSource | None = None,
    *,
    limits: AskLimits | None = None,
    deps_for: Callable[[Bundle], AgentDeps] | None = None,
) -> FastAPI:
    """The app; `source`, `limits` and `deps_for` replace the live ones in tests."""
    settings = settings or ApiSettings()
    data = source or BundleSource(http_fetch(settings.agent_data_url))
    agent = deps_for or live_deps(settings)
    counters = limits or live_limits(settings)
    answers = AnswerCache()
    one_run = threading.Lock()  # one question at a time: the free tier's per-minute limits

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Load the data while the replica starts, so the first visitor doesn't wait for it.
        threading.Thread(target=data.current, daemon=True).start()
        yield

    app = FastAPI(
        title="PatchPulse API", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )
    if settings.site_origin:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[settings.site_origin],
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness check used by the platform probe, the deploy smoke test and uptime.yml.

        It never touches SQL or the agent's data: a paused free-offer database or an unpublished
        bundle must not make the API look down. `APP_VERSION` is baked into the image at build
        time (the git SHA).
        """
        return {"status": "ok", "version": os.environ.get("APP_VERSION", "dev")}

    @app.get("/reports/{patch_id}")
    def report(
        patch_id: Annotated[str, PathParam(pattern=PATCH_ID)],
    ) -> dict[str, Any]:
        loaded = data.current()
        if loaded is None:
            raise HTTPException(503, NOT_LOADED)
        bundle_id = loaded.patches.get(patch_id)
        public = loaded.reports.get(bundle_id) if bundle_id else None
        if public is None:
            raise HTTPException(404, "no report for this patch yet")
        return {"data_version": loaded.data_version, "report": public}

    @app.post("/ask")
    def ask(body: AskRequest, request: Request) -> dict[str, Any]:
        loaded = data.current()
        if loaded is None:
            raise HTTPException(503, NOT_LOADED)
        if agent is None or counters is None:
            raise HTTPException(503, UNAVAILABLE)
        version = loaded.data_version
        early = question_precheck(loaded.bundle, body.question, body.appid)
        if early is not None:
            return {"data_version": version, "refusal": early.reason}
        key = (version, body.appid, normalise(body.question))
        cached = answers.get(key)
        if cached is not None:
            return cached
        verdict = counters.check(visitor_address(request))
        if isinstance(verdict, Refused):
            headers = {"Retry-After": str(verdict.retry_after)}
            raise HTTPException(429, verdict.reason, headers=headers)
        with one_run:
            cached = answers.get(key)
            if cached is not None:
                return cached
            question = ReportRequest.for_question(body.question, appid=body.appid)
            result = run_report(question, agent(loaded.bundle))
        if isinstance(result, Refusal):
            answer: dict[str, Any] = {"data_version": version, "refusal": result.reason}
            if result.reason != MODELS_DOWN:
                answers.put(key, answer)
            return answer
        answer = {"data_version": version, "report": public_copy(result)}
        if not result.basic:  # a basic answer is tried again with the models next time
            answers.put(key, answer)
        return answer

    return app


app = create_app()
