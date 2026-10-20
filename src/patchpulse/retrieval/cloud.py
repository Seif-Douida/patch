"""The free cluster's rows for the eval report (plan Task 17, design D7 and D9).

`patchpulse-ml retrieval report --cloud` measures the uploaded collection from the laptop and adds
one section to reports/retrieval/phase-3-eval.md, just before its limitations; the rest of the
report, and the judgments it rests on, are left as they are. A later run replaces the section.
- ANN recall@10 of the pinned encoder: dense search through the cluster's HNSW index against
  exact search, base filters, every frozen query, through the alias the job and the API use.
- Each system's latency (BM25, dense, hybrid; base filters): one search request's round trip
  from the laptop, every query 3 times. The query is embedded beforehand: its cost is the
  encoder's, measured on pp-api's CPU in reports/retrieval/embedder-speed.json. Beside the p95 of
  every request, the p95 of each query's fastest try: on 20 Oct the laptop's requests carrying an
  8 KB vector stalled about half a second now and then (BM25's never did), which the second
  number leaves out.
- Memory and data, from the cluster's telemetry: Qdrant's resident memory against the node's RAM,
  and the collection's stored vectors and payloads against its disk.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx2
import numpy as np
from numpy.typing import NDArray
from pydantic import SecretStr

from patchpulse.models.text import clean_text
from patchpulse.retrieval.evaluate import ann_recall
from patchpulse.retrieval.queries import Query
from patchpulse.retrieval.search import QueryEmbedder, search

SECTION = "## On the free cluster"
LIMITATIONS = "## Limitations"
SYSTEMS = (("BM25", "bm25"), ("Dense", "dense"), ("Hybrid", "hybrid"))
REPEATS = 3
_KIB = 1024
_MIB = 2**20


@dataclass(frozen=True)
class Telemetry:
    resident_bytes: int  # Qdrant's resident memory
    ram_bytes: int  # the node's
    stored_bytes: int  # the collection's vectors and payloads
    disk_bytes: int  # the node's


@dataclass(frozen=True)
class Latency:
    system: str
    p50_ms: float
    p95_ms: float
    p95_fastest_ms: float  # over queries, each one's fastest try
    requests: int


@dataclass(frozen=True)
class CloudRows:
    measured_on: str
    version: str
    collection: str
    alias: str
    points: int
    embedder: str
    registry_version: int
    queries: int
    ann_recall: float
    latency: list[Latency]
    telemetry: Telemetry


class _Embedded:
    """The queries' vectors, computed once, so a timed search is only its request."""

    def __init__(self, embedder: QueryEmbedder, queries: Sequence[Query]) -> None:
        texts = sorted({clean_text(query.text) for query in queries} - {""})
        vectors = embedder.embed_queries(texts)
        self._vectors = dict(zip(texts, vectors, strict=True))

    def embed_queries(self, texts: list[str], *, batch: int = 32) -> NDArray[np.float32]:
        return np.stack([self._vectors[text] for text in texts])


def read_telemetry(url: str, api_key: SecretStr | None, collection: str) -> Telemetry:
    """The cluster's own figures (details level 3: per-shard sizes). qdrant-client's models can't
    parse that level, so it's read as JSON."""
    headers = {"api-key": api_key.get_secret_value()} if api_key is not None else {}
    response = httpx2.get(
        url.rstrip("/") + "/telemetry", params={"details_level": "3"}, headers=headers, timeout=60
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()["result"]
    system = result["app"]["system"]
    (found,) = [c for c in result["collections"]["collections"] if c["id"] == collection]
    stored = sum(
        int(shard["local"].get("vectors_size_bytes", 0))
        + int(shard["local"].get("payloads_size_bytes", 0))
        for shard in found["shards"]
        if shard.get("local")
    )
    return Telemetry(
        resident_bytes=int(result["memory"]["resident_bytes"]),
        ram_bytes=int(system["ram_size"]) * _KIB,  # sysinfo reports KiB
        stored_bytes=stored,
        disk_bytes=int(system["disk_size"]) * _KIB,
    )


def measure_latency(
    client: Any,
    collection: str,
    embedder: QueryEmbedder,
    queries: Sequence[Query],
    *,
    repeats: int = REPEATS,
) -> list[Latency]:
    """Each system's search round trip, every query `repeats` times."""
    embedded = _Embedded(embedder, queries)
    found = []
    for label, mode in SYSTEMS:
        seconds: list[float] = []
        fastest = [float("inf")] * len(queries)
        for _ in range(repeats):
            for i, query in enumerate(queries):
                started = time.perf_counter()
                search(
                    client,
                    collection,
                    embedded,
                    query.text,
                    appid=query.appid,
                    language=query.language,
                    filters=query.base_filters,
                    mode=mode,
                    k=10,
                )
                seconds.append(time.perf_counter() - started)
                fastest[i] = min(fastest[i], seconds[-1])
        p50, p95 = np.percentile(np.array(seconds) * 1000, [50, 95])
        best = np.percentile(np.array(fastest) * 1000, 95)
        found.append(
            Latency(
                label,
                round(float(p50), 1),
                round(float(p95), 1),
                round(float(best), 1),
                len(seconds),
            )
        )
    return found


def measure_cloud(
    client: Any,
    *,
    collection: str,
    alias: str,
    embedder: QueryEmbedder,
    embedder_name: str,
    registry_version: int,
    queries: Sequence[Query],
    telemetry: Telemetry,
    repeats: int = REPEATS,
) -> CloudRows:
    return CloudRows(
        measured_on=datetime.now(UTC).date().isoformat(),
        version=str(client.info().version),
        collection=collection,
        alias=alias,
        points=int(client.count(alias, exact=True).count),
        embedder=embedder_name,
        registry_version=registry_version,
        queries=len(queries),
        ann_recall=ann_recall(client, alias, embedder, queries),
        latency=measure_latency(client, alias, embedder, queries, repeats=repeats),
        telemetry=telemetry,
    )


def _mib(value: int) -> str:
    return f"{value / _MIB:,.0f} MiB"


def render_cloud_section(rows: CloudRows) -> str:
    t = rows.telemetry
    lines = [
        SECTION,
        "",
        f"Measured on {rows.measured_on} on Qdrant Cloud's free cluster (Qdrant {rows.version}): "
        f"`{rows.collection}` behind the alias `{rows.alias}`, {rows.points:,} points, encoder "
        f"{rows.embedder} (`review-encoder` version {rows.registry_version}).",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| ANN recall@10 ({rows.embedder}, base filters, {rows.queries} queries) | "
        f"{rows.ann_recall:.3f} |",
        f"| Qdrant's resident memory | {_mib(t.resident_bytes)} of {_mib(t.ram_bytes)} "
        f"({t.resident_bytes / t.ram_bytes:.0%}) |",
        f"| Stored vectors and payloads | {_mib(t.stored_bytes)} of {_mib(t.disk_bytes)} disk |",
        "",
        f"One search's round trip from the laptop, base filters, every query {REPEATS} times (the "
        "query's embedding is excluded: the speed report measures it on pp-api's CPU). The last "
        "column takes each query's fastest try, leaving out the laptop's occasional stalls of "
        "about half a second on requests that carry the query's vector:",
        "",
        f"| System | p50 ms | p95 ms | p95 ms, fastest of {REPEATS} | Requests |",
        "|---|---|---|---|---|",
        *(
            f"| {latency.system} | {latency.p50_ms:.1f} | {latency.p95_ms:.1f} | "
            f"{latency.p95_fastest_ms:.1f} | {latency.requests} |"
            for latency in rows.latency
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def extend_report(text: str, section: str) -> str:
    """The report with `section` just before its limitations, replacing an earlier one."""
    if SECTION in text:
        start = text.index(SECTION)
        end = text.find("\n## ", start + len(SECTION))
        return text[:start] + section + (text[end + 1 :] if end != -1 else "")
    if LIMITATIONS not in text:
        raise ValueError(f"the report has no {LIMITATIONS!r} section to add the cloud rows before")
    at = text.index(LIMITATIONS)
    return text[:at] + section + text[at:]
