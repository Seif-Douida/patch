"""The agent's Langfuse traces (Phase 5 plan Task 18, design D13). The real SDK, with its spans
kept in memory and its HTTP calls (the scores) recorded instead of sent: every byte a trace would
send leaves through one of the two."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from agent_fixtures import GAME, make_bundle
from agent_world import CHAIN, CRASHES, Scripted, World, report_of
from fake_clock import FakeClock
from langfuse import Langfuse
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from patchpulse.agent.cli import _live_deps
from patchpulse.agent.graph import ReportRequest
from patchpulse.agent.tracing import (
    ASK_SAMPLE,
    MONTH_UNITS_CAP,
    LangfuseTracer,
    MonthUnits,
    NoTracer,
    make_tracer,
)
from patchpulse.api.ask import LiveDeps
from patchpulse.api.limits import MemoryLimitStore
from patchpulse.api.settings import ApiSettings
from patchpulse.config import LabSettings, Settings
from patchpulse.pipeline.reports import _live_agent

STEPS = ["scope", "core_evidence", "rules", "gather", "draft", "verify", "summarize", "finalize"]
QUESTION = "Why did Helldivers 2 reviews drop after patch 1.1.0?"
OCTOBER = datetime(2026, 10, 11, 3, 0, tzinfo=UTC)


class Capture:
    """A Langfuse client whose spans stay in memory and whose HTTP requests are recorded."""

    def __init__(self) -> None:
        self.exporter = InMemorySpanExporter()
        self.requests: list[httpx.Request] = []

        def answer(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(200, json={})

        # The SDK keeps one set of resources per public key: a new key per test.
        self.client = Langfuse(
            public_key=f"pk-lf-test-{uuid.uuid4().hex}",
            secret_key="sk-lf-test",
            base_url="https://langfuse.test",
            span_exporter=self.exporter,
            httpx_client=httpx.Client(transport=httpx.MockTransport(answer)),
        )

    def spans(self) -> list[ReadableSpan]:
        self.client.flush()
        return list(self.exporter.get_finished_spans())

    def scores(self) -> list[dict[str, Any]]:
        self.client.flush()
        found: list[dict[str, Any]] = []
        for request in self.requests:
            if request.url.path == "/api/public/scores":
                body = json.loads(request.content)
                found += body if isinstance(body, list) else [body]
        return found

    def payloads(self) -> str:
        """Everything the trace sent: every span attribute and every request body."""
        texts = [json.dumps(dict(s.attributes or {}), default=str) for s in self.spans()]
        texts += [s.name for s in self.spans()]
        texts += [r.content.decode("utf-8", "replace") for r in self.requests]
        return "\n".join(texts)


@pytest.fixture
def capture() -> Iterator[Capture]:
    found = Capture()
    yield found
    found.client.shutdown()


def tracer(capture: Capture, **kwargs: Any) -> LangfuseTracer:
    units = kwargs.pop("units", None) or MonthUnits(MemoryLimitStore(), clock=FakeClock(OCTOBER))
    return LangfuseTracer(capture.client, units=units, **kwargs)


def attribute(span: ReadableSpan, name: str) -> Any:
    """A span attribute; the SDK sends inputs, outputs and metadata values as JSON."""
    value = (span.attributes or {}).get(f"langfuse.{name}")
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def test_a_report_trace_has_every_step(tmp_path: Path, capture: Capture) -> None:
    units = MonthUnits(MemoryLimitStore(), clock=FakeClock(OCTOBER))
    traced = tracer(capture, units=units)

    report = report_of(
        World(tmp_path, Scripted(), tracer=traced).run(ReportRequest.for_bundle("p1", 7))
    )

    spans = capture.spans()
    assert len({s.context.trace_id for s in spans}) == 1
    (root,) = [s for s in spans if s.parent is None]
    assert root.name == "report p1-7d"
    assert attribute(root, "observation.type") == "agent"
    assert attribute(root, "trace.name") == "report"
    assert attribute(root, "trace.tags") == ("report",)
    assert attribute(root, "observation.input") == {"bundle_id": "p1", "horizon": 7}
    output = attribute(root, "observation.output")
    assert (output["report_id"], output["confidence"], output["basic"]) == (
        "p1-7d",
        "medium",
        False,
    )
    assert (output["claims"], output["withheld"]) == (3, 1)

    def children(parent: ReadableSpan) -> list[ReadableSpan]:
        found = [s for s in spans if s.parent and s.parent.span_id == parent.context.span_id]
        return sorted(found, key=lambda s: s.start_time or 0)

    steps = children(root)
    assert [s.name for s in steps] == STEPS
    by_step = {s.name: children(s) for s in steps}
    core = [s.name for s in by_step["core_evidence"]]
    assert {"sentiment_timeseries", "aspect_breakdown", "estimate_patch_effect"} <= set(core)
    assert all(attribute(s, "observation.type") == "tool" for s in by_step["core_evidence"])
    gathered = [
        s.name for s in by_step["gather"] if attribute(s, "observation.type") != "generation"
    ]
    assert gathered == ["get_patch_notes"] + ["search_reviews"] * 7
    # A generation is named after its prompt, under the step that asked.
    generations = [s for s in spans if attribute(s, "observation.type") == "generation"]
    assert [(g.name, g.parent and g.parent.span_id) for g in generations] == [
        (prompt, next(s for s in steps if s.name == step).context.span_id)
        for prompt, step in (("gather", "gather"), ("draft", "draft"), ("summary", "summarize"))
    ]
    for generation in generations:
        assert attribute(generation, "observation.model.name") == CHAIN[0]
        assert attribute(generation, "observation.usage_details") == {"input": 900, "output": 120}
        assert attribute(generation, "observation.metadata.cached") is False
        assert (generation.end_time or 0) >= (generation.start_time or 0)
    verify = next(s for s in steps if s.name == "verify")
    assert attribute(verify, "observation.type") == "evaluator"
    assert attribute(verify, "observation.output")["pass_rates"] == pytest.approx(
        {k: v for k, v in report.rule_pass_rates.items() if k != "V6"}
    )
    scores = {s["name"]: s["value"] for s in capture.scores()}
    assert scores == pytest.approx(
        {f"{rule} pass rate": v for rule, v in report.rule_pass_rates.items()}
    )
    # The month's units, as Langfuse counts them: the trace, its observations and its scores.
    assert units.used() == 1 + len(spans) + len(scores)


def test_traces_never_carry_review_text(tmp_path: Path, capture: Capture) -> None:
    world = World(tmp_path, Scripted(), tracer=tracer(capture, ask_sample=1.0))

    world.run(ReportRequest.for_bundle("p1", 7))
    world.run(ReportRequest.for_question(QUESTION, appid=GAME))

    sent = capture.payloads()
    retrieved = [CRASHES, "Servers are stable now."]
    for text in [*retrieved, *(t[:24] for t in retrieved), "crashes on load", "Fixed crashes"]:
        assert text not in sent, text
    assert QUESTION not in sent  # the visitor's words stay out too: only their length goes
    # What does go: the hits' ids, scores and lengths.
    searches = [s for s in capture.spans() if s.name == "search_reviews"]
    assert attribute(searches[0], "observation.output")["hits"] == [
        {"review_id": 101, "score": 0.9, "chars": len(CRASHES)},
        {"review_id": 102, "score": 0.4, "chars": len("Servers are stable now.")},
    ]
    assert attribute(searches[0], "observation.input")["query"] == "crashes 0"


def test_ask_traces_are_sampled_and_capped(tmp_path: Path, capture: Capture) -> None:
    assert (ASK_SAMPLE, MONTH_UNITS_CAP) == (0.25, 45_000)
    clock = FakeClock(OCTOBER)
    store = MemoryLimitStore()
    units = MonthUnits(store, clock=clock)
    draws = iter([0.1, 0.3, 0.24, 0.9, 0.0, 0.0])
    world = World(
        tmp_path, Scripted(), tracer=tracer(capture, units=units, rng=lambda: next(draws))
    )

    def traces() -> int:
        return len({s.context.trace_id for s in capture.spans()})

    for _ in range(4):
        world.run(ReportRequest.for_question(QUESTION, appid=GAME))
    assert traces() == 2  # the draws under 0.25

    store.put(units.key(), {"month": "2026-10", "units": MONTH_UNITS_CAP})
    world.run(ReportRequest.for_question(QUESTION, appid=GAME))
    assert traces() == 2  # the month's units are spent: Ask isn't traced
    world.run(ReportRequest.for_bundle("p1", 7))
    assert traces() == 3  # nightly reports always are, and still count
    assert units.used() > MONTH_UNITS_CAP

    clock.now += timedelta(days=21)  # November
    world.run(ReportRequest.for_question(QUESTION, appid=GAME))
    assert traces() == 4


class Broken:
    """A Langfuse client that fails on every call."""

    def __getattr__(self, name: str) -> Any:
        def fail(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("langfuse is down")

        return fail


class DownStore:
    def get(self, key: str) -> dict[str, Any] | None:
        raise ConnectionError("qdrant is down")

    def put(self, key: str, payload: dict[str, Any]) -> None:
        raise ConnectionError("qdrant is down")


def test_a_failing_trace_never_fails_the_report(tmp_path: Path, capture: Capture) -> None:
    untraced = report_of(World(tmp_path, Scripted()).run(ReportRequest.for_bundle("p1", 7)))
    broken = LangfuseTracer(cast(Langfuse, Broken()), units=MonthUnits(MemoryLimitStore()))
    no_counts = tracer(capture, units=MonthUnits(DownStore()), ask_sample=1.0)

    for traced in (broken, no_counts):
        world = World(tmp_path, Scripted(), tracer=traced)
        assert report_of(world.run(ReportRequest.for_bundle("p1", 7))) == untraced
        report_of(world.run(ReportRequest.for_question(QUESTION, appid=GAME)))
        traced.close()

    # Without the month's count, Ask isn't traced; the nightly report still is.
    roots = [s.name for s in capture.spans() if s.parent is None]
    assert roots == ["report p1-7d"]


def test_no_keys_means_no_tracer() -> None:
    assert isinstance(make_tracer(None, None, None, store=MemoryLimitStore()), NoTracer)


def test_a_traced_run_knows_its_trace_id(capture: Capture) -> None:
    """The acceptance run prints each run's trace (plan Task 21)."""
    request = ReportRequest.for_bundle("p1", 7)
    traced = tracer(capture).start(request, data_version=52)
    untraced = NoTracer().start(request, data_version=52)

    traced.finish(None)

    (root,) = capture.spans()
    assert traced.trace_id == format(root.context.trace_id, "032x")
    assert untraced.trace_id is None


def test_the_live_agents_trace_with_the_langfuse_keys(tmp_path: Path) -> None:
    # The nightly stage and the backfill trace every report; the API traces Ask at its sample.
    keys: dict[str, Any] = {
        "langfuse_public_key": f"pk-lf-test-{uuid.uuid4().hex}",
        "langfuse_secret_key": SecretStr("hunter2-hunter2"),
        "langfuse_host": "https://langfuse.test",
        "gemini_api_key": SecretStr("hunter2-gemini"),
    }
    db: dict[str, Any] = {"db_host": "h", "db_name": "n", "db_user": "u", "db_password": "p"}
    bundle = make_bundle()
    with ExitStack() as stack:
        for traced in (True, False):
            langfuse = keys if traced else {"gemini_api_key": keys["gemini_api_key"]}
            nightly = _live_agent(stack, Settings(**db, **langfuse), bundle, steam=NO_STEAM)
            backfill = _live_deps(stack, LabSettings(**langfuse), tmp_path)(bundle)
            api = LiveDeps(ApiSettings(ask_trace_sample=0.5, **langfuse), SecretStr("k"))(bundle)
            tracers = [cast(Any, nightly).keywords["deps"].tracer, backfill.tracer, api.tracer]
            kind = LangfuseTracer if traced else NoTracer
            assert all(isinstance(t, kind) for t in tracers), tracers
            if traced:
                assert [t.ask_sample for t in tracers] == [ASK_SAMPLE, ASK_SAMPLE, 0.5]
                api.tracer.close()


NO_STEAM: Any = None  # the agents are built, never run
