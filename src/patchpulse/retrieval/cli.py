"""`patchpulse-ml retrieval ...`: Phase 3's local commands, run on Seif's machine.

    retrieval benchmark  Time each candidate embedder where it will run: a night's reviews in a
                         1-CPU, 2 GiB Docker container, single queries in a half-CPU, 1 GiB one
                         (reports/retrieval/embedder-speed.json). Builds missing int8 embedders
                         under build/encoders/ (the gpu group: `uv run --group gpu ...`).
    retrieval embed      Embed the pulled index set with one candidate (`--model e5-small`) on all
                         CPU threads, into data/local/embeddings/. Hours for the larger model;
                         a stop resumes at the next 10,000.
    retrieval load       Load each candidate's collection (`eval_<name>`) on the local Qdrant
                         (`--target local`) from the embedding cache; resumable and idempotent.
    retrieval queries check   Check data/retrieval/queries.csv and count its queries.
    retrieval queries freeze  Check it and record its SHA-256 (queries.sha256); runs and the report
                              refuse queries that aren't frozen or changed since.

Exit codes as `patchpulse-ml`: 0 success, 1 the command failed, 2 bad configuration or a refused
action.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from patchpulse.config import LabSettings
from patchpulse.models.snapshot import load_snapshot
from patchpulse.retrieval.benchmark import (
    BENCH_QUERIES,
    INDEX_BUDGET_S,
    NIGHTLY_REVIEWS,
    QUERY_P95_MS,
    benchmark_embedders,
    newest_index_texts,
    passes_floor,
)
from patchpulse.retrieval.embed import EmbeddingCache, embed_index_set
from patchpulse.retrieval.embedders import CANDIDATES, OnnxEmbedder, build_embedder
from patchpulse.retrieval.load import build_local_collections, eval_collection
from patchpulse.retrieval.qdrant import make_client
from patchpulse.retrieval.queries import (
    QueriesNotFrozenError,
    QueryFileError,
    freeze_queries,
    load_queries,
)

log = logging.getLogger("patchpulse.retrieval")

REPORTS_DIR = Path("reports/retrieval")
SPEED_PATH = REPORTS_DIR / "embedder-speed.json"
ENCODER_BUILDS = Path("build/encoders")
BENCH_TEXTS = 1_000
QUERIES_PATH = Path("data/retrieval/queries.csv")


def embedder_dir(name: str) -> Path:
    """A candidate's int8 build, made on first use."""
    return build_embedder(CANDIDATES[name], ENCODER_BUILDS / f"{name}-int8")


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    retrieval = commands.add_parser(
        "retrieval", help="Phase 3: the review index and its eval"
    ).add_subparsers(dest="retrieval_command", required=True)
    benchmark = retrieval.add_parser(
        "benchmark", help="time each candidate embedder in 1-CPU containers (needs Docker)"
    )
    benchmark.add_argument(
        "--models", default=",".join(CANDIDATES), help="comma-separated candidate names"
    )
    embed = retrieval.add_parser(
        "embed", help="embed the index set with one candidate (hours; resumable)"
    )
    embed.add_argument("--model", required=True, choices=list(CANDIDATES))
    load = retrieval.add_parser("load", help="load the candidates' collections on Qdrant")
    load.add_argument("--target", required=True, choices=["local"])
    load.add_argument(
        "--models", default=",".join(CANDIDATES), help="comma-separated candidate names"
    )
    queries = retrieval.add_parser("queries", help="the eval's queries").add_subparsers(
        dest="queries_command", required=True
    )
    queries.add_parser("check", help="check data/retrieval/queries.csv and count its queries")
    queries.add_parser("freeze", help="check the queries and record their SHA-256")


def run(args: argparse.Namespace, settings: LabSettings) -> int:
    if args.retrieval_command == "queries":
        return _queries(freeze=args.queries_command == "freeze")
    if args.retrieval_command == "embed":
        return _embed(settings, args.model)
    names = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = [name for name in names if name not in CANDIDATES]
    if unknown:
        print(
            f"patchpulse-ml: unknown embedder {', '.join(unknown)}; "
            f"the candidates are {', '.join(CANDIDATES)}",
            file=sys.stderr,
        )
        return 2
    if args.retrieval_command == "load":
        return _load(settings, names)
    return _benchmark(settings, names)


def _benchmark(settings: LabSettings, names: list[str]) -> int:
    texts = newest_index_texts(load_snapshot(settings.data_dir), BENCH_TEXTS)
    dirs = [embedder_dir(name) for name in names]
    speeds = benchmark_embedders(dirs, texts, BENCH_QUERIES)
    report = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "texts": len(texts),
        "queries": len(BENCH_QUERIES),
        "floor": {
            "reviews_in_index_budget": NIGHTLY_REVIEWS,
            "budget_s": INDEX_BUDGET_S,
            "query_p95_ms": QUERY_P95_MS,
        },
        "embedders": [{**asdict(speed), "passes_floor": passes_floor(speed)} for speed in speeds],
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SPEED_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for speed in speeds:
        log.info(
            "%s: %.1f passages/s on 1 CPU, query p50 %.0f ms p95 %.0f ms on 0.5 CPU, %.0f MB: %s",
            speed.name,
            speed.passages_per_s,
            speed.query_p50_ms,
            speed.query_p95_ms,
            speed.size_mb,
            "passes the floor" if passes_floor(speed) else "under the floor",
        )
    return 0


def _embed(settings: LabSettings, name: str) -> int:
    snapshot = load_snapshot(settings.data_dir)
    embedder = OnnxEmbedder(embedder_dir(name))  # every CPU thread
    cache = EmbeddingCache.for_embedder(settings.data_dir, embedder)
    stats = embed_index_set(snapshot, embedder, cache)
    cached = set(cache.load()[0].tolist())
    held = sum(int(rid) in cached for rid in snapshot.index["review_id"])
    log.info(
        "%s: %d vectors for the %d reviews of the index set (%d embedded, %d reused, %.0f s) in %s",
        name,
        held,
        len(snapshot.index),
        stats.embedded,
        stats.reused,
        stats.seconds,
        cache.directory.as_posix(),
    )
    return 0 if held == len(snapshot.index) else 1


def _load(settings: LabSettings, names: list[str]) -> int:
    snapshot = load_snapshot(settings.data_dir)
    client = make_client(settings.qdrant_local_url, None)
    try:
        dirs = {name: embedder_dir(name) for name in names}
        build_local_collections(client, snapshot, dirs, data_dir=settings.data_dir, names=names)
        failed = 0
        for name in names:
            points = client.count(eval_collection(name), exact=True).count
            if points != len(snapshot.index):
                log.error(
                    "%s holds %d points for an index set of %d: delete it and load again",
                    eval_collection(name),
                    points,
                    len(snapshot.index),
                )
                failed += 1
    finally:
        client.close()
    return 1 if failed else 0


def _queries(*, freeze: bool) -> int:
    try:
        queries = load_queries(QUERIES_PATH)
        digest = freeze_queries(QUERIES_PATH) if freeze else None
    except (QueryFileError, QueriesNotFrozenError, FileNotFoundError) as error:
        print(f"patchpulse-ml: {error}", file=sys.stderr)
        return 1
    by = Counter((q.language, q.kind) for q in queries)
    games = Counter(q.appid for q in queries)
    print(
        f"{len(queries)} queries: "
        + ", ".join(f"{n} {language} {kind}" for (language, kind), n in sorted(by.items()))
        + f"; {sum(q.has_extra_filters for q in queries)} with extra filters; "
        + f"{len(games)} games, {min(games.values())} to {max(games.values())} each"
    )
    if digest is not None:
        print(f"frozen: {QUERIES_PATH.as_posix()} sha256 {digest}")
    return 0
