"""Phase 7 Task 13: App Insights on pp-api (design D8) - requests and dependencies, scrubbed,
rate-limited, and never a question or a review."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest
from api_fixtures import Published
from ask_fixtures import REVIEW_ID, Ask, Scripted
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import Decision
from opentelemetry.trace import SpanKind
from pydantic import SecretStr

from patchpulse.api import telemetry
from patchpulse.api.app import create_app
from patchpulse.api.bundle_source import BundleSource
from patchpulse.api.settings import ApiSettings
from patchpulse.api.telemetry import (
    TRACES_PER_SECOND,
    RateLimitedSampler,
    ScrubbingExporter,
    configure_telemetry,
)

CONNECTION = SecretStr(
    "InstrumentationKey=00000000-0000-0000-0000-000000000000;"
    "IngestionEndpoint=https://telemetry.invalid/"
)
CLUSTER = "https://c0ffee00-cluster.eu-central-1-0.aws.cloud.qdrant.io:6333"
QUESTION = "SENTINEL-QUESTION why did my frobnicator stop working after the patch?"
REVIEW = "SENTINEL-REVIEW the patch broke my saves and the support ignored me"


class HttpQdrant:
    """The review search's Qdrant, answering over HTTP (httpx2, a mock transport) with a review
    and a query string, as the real cluster would."""

    def __init__(self) -> None:
        def answer(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(200, json={"text": REVIEW})

        self.http = httpx2.Client(transport=httpx2.MockTransport(answer))

    def query_points(self, collection: str, **kwargs: Any) -> Any:
        body = self.http.get(f"{CLUSTER}/collections/{collection}/points?api-key=k&q=1").json()
        # Langfuse's SDK traces through the global provider: none of it may reach App Insights.
        with trace.get_tracer("langfuse-like").start_as_current_span(f"search {REVIEW}") as span:
            span.set_attribute("input", REVIEW)
        return SimpleNamespace(
            points=[SimpleNamespace(id=REVIEW_ID, score=0.9, payload={"text": body["text"]})]
        )


@pytest.fixture
def exported(tmp_path: Path) -> Iterator[tuple[Ask, InMemorySpanExporter]]:
    memory = InMemorySpanExporter()
    ask = Ask(tmp_path, Scripted(), qdrant=HttpQdrant())
    app: FastAPI = ask.client.app  # type: ignore[assignment]
    assert configure_telemetry(
        app, CONNECTION, exporter=memory, qdrant_url=SecretStr(CLUSTER), per_second=1000.0
    )
    yield ask, memory
    telemetry.shutdown(app)


def everything(spans: Sequence[ReadableSpan]) -> str:
    """Every name, attribute and event an exported span carries, as one string."""
    parts = []
    for span in spans:
        parts.append(span.name)
        parts.append(json.dumps({k: str(v) for k, v in (span.attributes or {}).items()}))
        for event in span.events:
            parts.append(event.name)
            parts.append(json.dumps({k: str(v) for k, v in (event.attributes or {}).items()}))
    return "\n".join(parts)


def test_without_a_connection_string_nothing_is_configured() -> None:
    app = FastAPI()

    assert configure_telemetry(app, None) is False
    assert not getattr(app, "_is_instrumented_by_opentelemetry", False)


def test_requests_and_dependencies_are_exported(
    exported: tuple[Ask, InMemorySpanExporter],
) -> None:
    ask, memory = exported

    assert ask.ask(QUESTION).status_code == 200

    spans = memory.get_finished_spans()
    # The request: under uvicorn a SERVER span; here the test client's own call (httpx2, traced
    # like any dependency) is its parent, so the instrumentation marks it INTERNAL.
    (request,) = [s for s in spans if (s.attributes or {}).get("http.route") == "/ask"]
    assert request.name == "POST /ask"
    assert (request.attributes or {}).get("http.status_code") == 200
    searches = [
        s
        for s in spans
        if s.kind is SpanKind.CLIENT and "qdrant" in str((s.attributes or {}).get("url.full"))
    ]
    assert len(searches) == 1, "the review search's HTTP call is a dependency"
    assert searches[0].parent is not None
    assert searches[0].context.trace_id == request.context.trace_id


def test_exported_telemetry_holds_no_text(exported: tuple[Ask, InMemorySpanExporter]) -> None:
    ask, memory = exported

    ask.ask(QUESTION)
    ask.client.get("/reports/p1", params={"q": QUESTION})

    sent = everything(memory.get_finished_spans())
    assert "/ask" in sent
    assert "SENTINEL-QUESTION" not in sent
    assert "SENTINEL-REVIEW" not in sent
    assert "frobnicator" not in sent


def test_query_strings_headers_and_the_cluster_host_are_scrubbed(
    exported: tuple[Ask, InMemorySpanExporter],
) -> None:
    ask, memory = exported

    ask.ask(QUESTION)

    sent = everything(memory.get_finished_spans())
    assert "c0ffee00-cluster" not in sent
    assert "qdrant" in sent
    assert "api-key" not in sent
    assert "?" not in "".join(
        str(v)
        for s in memory.get_finished_spans()
        for k, v in (s.attributes or {}).items()
        if "url" in k
    )
    assert "header" not in sent
    assert "X-Forwarded-For" not in sent
    assert "203.0.113.7" not in sent  # the visitor's address


def test_an_exception_keeps_its_type_and_loses_its_message() -> None:
    memory = InMemorySpanExporter()
    exporter = ScrubbingExporter(memory, qdrant_host=None)
    provider = telemetry.provider_for(exporter, per_second=1000.0)
    with provider.get_tracer("t").start_as_current_span("POST /ask") as span:
        span.record_exception(ValueError(f"bad question: {QUESTION}"))
    provider.shutdown()

    (event,) = memory.get_finished_spans()[0].events
    assert event.attributes is not None
    assert event.attributes["exception.type"] == "ValueError"
    assert "exception.message" not in event.attributes
    assert "SENTINEL" not in everything(memory.get_finished_spans())


class Ticks:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_the_sampler_caps_traces_per_second() -> None:
    clock = Ticks()
    sampler = RateLimitedSampler(2.0, clock=clock)

    def root() -> bool:
        decision = sampler.should_sample(None, 1, "GET /reports")
        return decision.decision is Decision.RECORD_AND_SAMPLE

    burst = [root() for _ in range(100)]
    clock.now += 1.0
    next_second = [root() for _ in range(100)]
    clock.now += 60.0
    after_a_quiet_minute = [root() for _ in range(100)]

    assert sum(burst) == 2
    assert sum(next_second) == 2
    assert sum(after_a_quiet_minute) == 2  # the bucket holds a second's worth, no more


def test_children_follow_their_parents_decision() -> None:
    memory = InMemorySpanExporter()
    clock = Ticks()
    provider = telemetry.provider_for(memory, per_second=1.0, clock=clock)
    tracer = provider.get_tracer("t")
    for _ in range(3):
        with tracer.start_as_current_span("root"), tracer.start_as_current_span("child"):
            pass
    provider.shutdown()

    assert sorted(s.name for s in memory.get_finished_spans()) == ["child", "root"]


def test_the_rate_keeps_a_flood_under_a_third_of_the_daily_cap(
    exported: tuple[Ask, InMemorySpanExporter],
) -> None:
    # Design D8: a flood at the sampler's limit for ~5 hours (until cost-guard stops pp-api) must
    # stay under 0.05 GB, a third of the workspace's 0.15 GB a day. An answered question is the
    # largest trace pp-api makes: the request and its dependencies.
    ask, memory = exported
    ask.ask(QUESTION)
    trace_bytes = sum(len(span.to_json()) for span in memory.get_finished_spans())

    assert trace_bytes <= telemetry.TRACE_BYTES
    assert TRACES_PER_SECOND * 5 * 3600 * telemetry.TRACE_BYTES <= 0.05e9


def test_the_azure_exporter_sends_no_resource_record(monkeypatch: pytest.MonkeyPatch) -> None:
    # pp-api, 19 Nov: an export logged "Failed to derive Resource from Tracer Provider" (it
    # reads the global provider, and ours isn't). The record is the exporter's own, like Statsbeat.
    from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter

    name = "APPLICATIONINSIGHTS_OPENTELEMETRY_RESOURCE_METRIC_DISABLED"
    monkeypatch.setenv(name, "placeholder")
    monkeypatch.delenv(name)  # unset now, and unset again after the test
    app = FastAPI()
    assert configure_telemetry(app, CONNECTION)
    try:
        exporter = AzureMonitorTraceExporter(
            connection_string=CONNECTION.get_secret_value(), disable_offline_storage=True
        )
        collects = exporter._should_collect_otel_resource_metric()  # type: ignore[no-untyped-call]
        assert collects is False
    finally:
        telemetry.shutdown(app)


def test_the_app_is_traced_when_azure_sets_the_connection_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Container Apps sets the variable without PP_ (infra/modules/containerapps.bicep).
    monkeypatch.setenv("APPLICATIONINSIGHTS_CONNECTION_STRING", CONNECTION.get_secret_value())
    settings = ApiSettings()
    assert settings.applicationinsights_connection_string == CONNECTION

    app = create_app(settings, BundleSource(Published()))
    try:
        assert getattr(app, "_is_instrumented_by_opentelemetry", False)
    finally:
        telemetry.shutdown(app)
