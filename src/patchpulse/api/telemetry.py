"""App Insights on pp-api (Phase 7 design D8): requests, dependencies and exceptions, through
OpenTelemetry's FastAPI and httpx instrumentation and the Azure Monitor exporter (not the distro,
which instruments every library it finds and would slow the cold start Ask's visitors wait for).

- **Own provider, never the global one.** Langfuse's SDK traces the agent through OpenTelemetry's
  global provider and adds its processor to whichever is registered: on the global provider, the
  agent's spans (which hold prompts) would reach App Insights, and every request would reach
  Langfuse. Ours is passed to each instrumentation explicitly.
- **What leaves:** route, status, duration; each dependency's method, scrubbed URL and status. No
  bodies, so no question or review. `ScrubbingExporter` drops, before export: query strings,
  every header, the client's address and user agent, exception messages and stack traces (the
  type stays), status descriptions, and links; the Qdrant cluster's host becomes `qdrant`.
- **Sampling:** `RateLimitedSampler`, a token bucket for new traces (children follow their parent).
  A flood at its limit for ~5 hours, until cost-guard stops pp-api, stays under 0.05 GB: a third of
  the workspace's 0.15 GB daily cap, which the nightly's console logs share.
- `/health` is not traced: the platform probe calls it every 30 s.

Gemini and GitHub are called through httpx2, which OpenTelemetry doesn't instrument: `Client.send`
gets a small client span here. Qdrant's client uses httpx (the httpx instrumentation).
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx2
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult
from opentelemetry.trace import Link, SpanKind, Status, StatusCode
from opentelemetry.util.types import Attributes
from pydantic import SecretStr

# The largest trace pp-api makes (an answered question: the request, its ASGI receive/send spans
# and the review search), as OpenTelemetry JSON: 6 spans, 5,722 bytes indented (4,410 compact),
# measured 18 Nov 2026 (tests/api/test_telemetry.py's fixture). 10,000 leaves ~1.75x for the
# columns App Insights adds to each stored row; the ingested size is checked after the deploy.
TRACE_BYTES = 10_000
FLOOD_SECONDS = 5 * 3600  # GitHub's scheduled cost-guard runs have had ~5-hour gaps
FLOOD_BYTES = 0.05e9  # a third of the 0.15 GB daily cap
# 0.05e9 / (18,000 s x 10,000 B) = 0.28 a second; one trace every 4 seconds.
TRACES_PER_SECOND = 0.25

SERVICE = "pp-api"
QDRANT = "qdrant"
_URL_KEYS = frozenset({"url.full", "http.url", "http.target", "url.path"})
_DROPPED = frozenset(
    {
        "url.query",
        "exception.message",
        "exception.stacktrace",
        "client.address",
        "client.port",
        "net.peer.ip",
        "net.sock.peer.addr",
        "net.sock.peer.port",
        "http.client_ip",
        "user_agent.original",
        "http.user_agent",
    }
)
_HEADERS = ("http.request.header.", "http.response.header.")
_STATE: dict[str, Any] = {}  # the live provider and tracer, for httpx2's span and for shutdown


class RateLimitedSampler(Sampler):
    """At most `per_second` new traces a second (a bucket of one second's worth, at least one);
    a span with a parent is sampled exactly when its parent was."""

    def __init__(self, per_second: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.per_second = per_second
        self.capacity = max(1.0, per_second)
        self._tokens = self.capacity
        self._clock = clock
        self._last = clock()
        self._lock = threading.Lock()

    def _take(self) -> bool:
        with self._lock:
            now = self._clock()
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.per_second)
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def should_sample(
        self,
        parent_context: Context | None,
        trace_id: int,
        name: str,
        kind: SpanKind | None = None,
        attributes: Attributes = None,
        links: Sequence[Link] | None = None,
        trace_state: Any = None,
    ) -> SamplingResult:
        parent = trace.get_current_span(parent_context).get_span_context()
        if parent.is_valid:
            sampled = parent.trace_flags.sampled
            state = parent.trace_state
        else:
            sampled, state = self._take(), None
        if sampled:
            return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes, state)
        return SamplingResult(Decision.DROP, None, state)

    def get_description(self) -> str:
        return f"RateLimitedSampler({self.per_second}/s)"


def _strip_query(value: str) -> str:
    return value.split("?", 1)[0].split("#", 1)[0]


class ScrubbingExporter(SpanExporter):
    """Copies each span without what must not leave (see the module docstring), then exports."""

    def __init__(self, inner: SpanExporter, *, qdrant_host: str | None) -> None:
        self.inner = inner
        self.qdrant_host = qdrant_host

    def _text(self, value: str) -> str:
        return value.replace(self.qdrant_host, QDRANT) if self.qdrant_host else value

    def _attributes(self, attributes: Mapping[str, Any] | None) -> dict[str, Any]:
        clean: dict[str, Any] = {}
        for key, value in (attributes or {}).items():
            if key in _DROPPED or key.startswith(_HEADERS):
                continue
            if isinstance(value, str):
                value = self._text(_strip_query(value) if key in _URL_KEYS else value)
            clean[key] = value
        return clean

    def _scrub(self, span: ReadableSpan) -> ReadableSpan:
        events = [
            Event(event.name, self._attributes(event.attributes), event.timestamp)
            for event in span.events
        ]
        return ReadableSpan(
            name=self._text(_strip_query(span.name)),
            context=span.context,
            parent=span.parent,
            resource=span.resource,
            attributes=self._attributes(span.attributes),
            events=events,
            links=(),
            kind=span.kind,
            status=Status(span.status.status_code),
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=span.instrumentation_scope,
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self.inner.export([self._scrub(span) for span in spans])

    def shutdown(self) -> None:
        self.inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self.inner.force_flush(timeout_millis)


def provider_for(
    exporter: SpanExporter,
    *,
    per_second: float = TRACES_PER_SECOND,
    clock: Callable[[], float] = time.monotonic,
    batch: bool = False,
) -> TracerProvider:
    provider = TracerProvider(
        sampler=RateLimitedSampler(per_second, clock=clock),
        resource=Resource.create({"service.name": SERVICE}),
    )
    processor = BatchSpanProcessor(exporter) if batch else SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    return provider


_ORIGINAL_SEND = httpx2.Client.send


def _traced_send(self: httpx2.Client, request: httpx2.Request, **kwargs: Any) -> httpx2.Response:
    tracer = _STATE.get("tracer")
    if tracer is None:
        return _ORIGINAL_SEND(self, request, **kwargs)
    attributes = {
        "http.request.method": request.method,
        "url.full": _strip_query(str(request.url)),
        "server.address": request.url.host,
    }
    with tracer.start_as_current_span(
        request.method, kind=SpanKind.CLIENT, attributes=attributes
    ) as span:
        response = _ORIGINAL_SEND(self, request, **kwargs)
        span.set_attribute("http.response.status_code", response.status_code)
        if response.status_code >= 400:
            span.set_status(Status(StatusCode.ERROR))
        return response


def configure_telemetry(
    app: FastAPI,
    connection_string: SecretStr | None,
    *,
    qdrant_url: SecretStr | None = None,
    exporter: SpanExporter | None = None,
    per_second: float = TRACES_PER_SECOND,
) -> bool:
    """Trace `app` to App Insights; nothing without a connection string. `exporter` replaces the
    Azure one in tests."""
    if connection_string is None:
        return False
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    if exporter is None:
        # Statsbeat is the exporter's own usage telemetry to Microsoft; nothing of ours needs it.
        os.environ.setdefault("APPLICATIONINSIGHTS_STATSBEAT_DISABLED_ALL", "true")
        from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter

        inner: SpanExporter = AzureMonitorTraceExporter(
            connection_string=connection_string.get_secret_value(),
            disable_offline_storage=True,  # nothing written to the replica's disk
        )
        batch = True
    else:
        inner, batch = exporter, False
    host = urlsplit(qdrant_url.get_secret_value()).hostname if qdrant_url else None
    provider = provider_for(
        ScrubbingExporter(inner, qdrant_host=host), per_second=per_second, batch=batch
    )
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider, excluded_urls="health")
    HTTPXClientInstrumentor().instrument(tracer_provider=provider)
    _STATE.update(provider=provider, tracer=provider.get_tracer("patchpulse.api.http"))
    httpx2.Client.send = _traced_send  # type: ignore[method-assign]
    return True


def shutdown(app: FastAPI) -> None:
    """Flush and undo `configure_telemetry` (tests, and a clean stop)."""
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    FastAPIInstrumentor.uninstrument_app(app)
    HTTPXClientInstrumentor().uninstrument()
    httpx2.Client.send = _ORIGINAL_SEND  # type: ignore[method-assign]
    provider = _STATE.pop("provider", None)
    _STATE.pop("tracer", None)
    if provider is not None:
        provider.shutdown()
