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
    retrieval run        Run every system (BM25, each candidate's dense and hybrid search; base
                         and filtered) over the frozen queries on the local collections, into
                         data/retrieval/runs/ (ids and scores, committed), and size the pool.
    retrieval export-batches  Write the pool's pairs as judging batches to
                              data/local/retrieval_batches/ (review text: never committed).

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

import pandas as pd

from patchpulse.config import LabSettings
from patchpulse.models.snapshot import REVIEWS_FILE, load_snapshot
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

# Commands that talk to Qdrant import its client when they run: `patchpulse-ml` must start with
# only the groups a step installs (deploy.yml's champion download has no `retrieval` group).

log = logging.getLogger("patchpulse.retrieval")

REPORTS_DIR = Path("reports/retrieval")
SPEED_PATH = REPORTS_DIR / "embedder-speed.json"
ENCODER_BUILDS = Path("build/encoders")
BENCH_TEXTS = 1_000
QUERIES_PATH = Path("data/retrieval/queries.csv")
RUNS_DIR = Path("data/retrieval/runs")
BATCHES_DIR = "retrieval_batches"  # under the data dir (data/local)
REJUDGE_DIR = "retrieval_rejudge"  # judged again after a guidelines revision (D7, data/local)
PILOT_DIR = "retrieval_pilot"  # the relevance pilot, judged and shown to Seif (data/local)


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
    retrieval.add_parser("run", help="run every system over the frozen queries (local Qdrant)")
    batches = retrieval.add_parser(
        "export-batches", help="write the pool as judging batches (data/local)"
    )
    batches.add_argument(
        "--rejudge-grades",
        help="only the pairs judged with these grades, e.g. 0,1, to data/local/retrieval_rejudge "
        "(after a guidelines revision, design D7)",
    )
    judged = retrieval.add_parser(
        "import-judgments", help="check a judge's grades for one batch, then merge them in"
    )
    judged.add_argument("--batch", required=True, type=Path, help="the batch-NNN.jsonl judged")
    judged.add_argument("--answers", required=True, type=Path, help="the judge's CSV")
    judged.add_argument("--guidelines", required=True, help="the guidelines version, e.g. v1")
    judged.add_argument(
        "--judge", required=True, help="the model that graded it, e.g. claude-opus-5-5"
    )
    retrieval.add_parser("audit-sample", help="draw Seif's 300 audit pairs from the pool, once")
    retrieval.add_parser("agreement", help="Seif's audit against the judges: kappa and the rule")


def run(args: argparse.Namespace, settings: LabSettings) -> int:
    if args.retrieval_command == "queries":
        return _queries(freeze=args.queries_command == "freeze")
    if args.retrieval_command == "run":
        return _run(settings)
    if args.retrieval_command == "export-batches":
        return _export_batches(settings, args.rejudge_grades)
    if args.retrieval_command == "import-judgments":
        return _import_judgments(args.batch, args.answers, args.guidelines, args.judge)
    if args.retrieval_command == "audit-sample":
        return _audit_sample(settings)
    if args.retrieval_command == "agreement":
        return _agreement(settings)
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
    from patchpulse.retrieval.load import build_local_collections, eval_collection
    from patchpulse.retrieval.qdrant import make_client

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


def _run(settings: LabSettings) -> int:
    from patchpulse.retrieval import runs
    from patchpulse.retrieval.qdrant import make_client
    from patchpulse.retrieval.queries import QueriesNotFrozenError

    embedders = {name: OnnxEmbedder(embedder_dir(name)) for name in CANDIDATES}
    client = make_client(settings.qdrant_local_url, None)
    try:
        found = runs.run_systems(client, QUERIES_PATH, embedders)
    except QueriesNotFrozenError as error:
        print(f"patchpulse-ml: {error}", file=sys.stderr)
        return 1
    finally:
        client.close()
    runs.write_runs(found, RUNS_DIR)
    pairs = runs.pool(found)
    sizes = pairs.groupby("query_id").size()
    print(
        f"{len(found)} runs in {RUNS_DIR.as_posix()}; pool: {len(pairs)} pairs for "
        f"{len(sizes)} queries, {sizes.min()} to {sizes.max()} a query "
        f"(median {sizes.median():.0f})"
    )
    return 0


def _export_batches(settings: LabSettings, rejudge: str | None = None) -> int:
    from patchpulse.retrieval import runs
    from patchpulse.retrieval.judging import (
        JUDGING_SEED,
        JUDGMENTS_PATH,
        LONG_TEXT,
        MANIFEST_FILE,
        export_judging_batches,
    )
    from patchpulse.retrieval.queries import QueriesNotFrozenError, load_queries, require_frozen

    try:
        require_frozen(QUERIES_PATH)
    except QueriesNotFrozenError as error:
        print(f"patchpulse-ml: {error}", file=sys.stderr)
        return 1
    found = runs.read_runs(RUNS_DIR)
    missing = [
        runs.run_file(system, variant)
        for system in runs.SYSTEMS
        for variant in runs.VARIANTS
        if (system, variant) not in found
    ]
    if missing:
        print(
            f"patchpulse-ml: missing runs ({', '.join(missing)}): run "
            "`patchpulse-ml retrieval run` first",
            file=sys.stderr,
        )
        return 1
    pairs, folder = runs.pool(found), BATCHES_DIR
    if rejudge is not None:
        grades = {int(grade) for grade in rejudge.split(",") if grade.strip()}
        judged = pd.read_csv(JUDGMENTS_PATH, dtype={"query_id": str})
        merged = pairs.merge(judged[["query_id", "review_id", "grade"]], how="left")
        unjudged = int(merged["grade"].isna().sum())
        if unjudged:
            print(
                f"patchpulse-ml: {unjudged} pool pair{' has' if unjudged == 1 else 's have'} no "
                "judgment: import every batch before re-judging",
                file=sys.stderr,
            )
            return 1
        pairs = merged[merged["grade"].isin(grades)][["query_id", "review_id"]]
        folder = REJUDGE_DIR
    batches = export_judging_batches(
        pairs,
        load_queries(QUERIES_PATH),
        load_snapshot(settings.data_dir),
        settings.data_dir / folder,
        seed=JUDGING_SEED,
    )
    out = (settings.data_dir / folder).as_posix()
    total = sum(b.pairs for b in batches)
    print(
        f"{len(batches)} batch{'es' if len(batches) != 1 else ''}, "
        f"{total} pair{'s' if total != 1 else ''} in {out}; "
        f"{sum(len(b.long_pairs) for b in batches)} longer than {LONG_TEXT:,} characters "
        f"(listed per batch in {MANIFEST_FILE})"
    )
    return 0


def _import_judgments(batch: Path, answers: Path, guidelines: str, judge: str) -> int:
    from patchpulse.retrieval.judging import import_judgments

    try:
        graded = import_judgments(batch, answers, guidelines_version=guidelines, judge=judge)
    except (ValueError, FileNotFoundError) as error:
        print(f"patchpulse-ml: {answers.name}: {error}", file=sys.stderr)
        return 1
    grades = pd.read_csv(answers, dtype=str)["grade"].str.strip().value_counts()
    print(
        f"{batch.name}: {graded} pairs graded "
        f"({', '.join(f'{g}: {grades.get(g, 0)}' for g in ('0', '1', '2'))}) under {guidelines} "
        f"by {judge}"
    )
    return 0


def _audit_sample(settings: LabSettings) -> int:
    from patchpulse.retrieval import runs
    from patchpulse.retrieval.audit import AUDIT_PATH, audit_sample, write_audit

    # Seif read the pilot's pairs with the pilot judge's grades: they can't be audited blind.
    pilot = settings.data_dir / PILOT_DIR / "judgments.csv"
    seen: set[tuple[str, int]] = set()
    if pilot.is_file():
        rows = pd.read_csv(pilot, dtype={"query_id": str})
        seen = set(zip(rows["query_id"], rows["review_id"].astype(int), strict=True))
    pairs = audit_sample(runs.pool(runs.read_runs(RUNS_DIR)), exclude=seen)
    try:
        write_audit(pairs, AUDIT_PATH)
    except ValueError as error:
        print(f"patchpulse-ml: {error}", file=sys.stderr)
        return 2
    print(
        f"{len(pairs)} pairs for Seif's audit in {AUDIT_PATH.as_posix()} "
        f"({len({q for q, _ in pairs})} queries)"
    )
    return 0


def _agreement(settings: LabSettings) -> int:
    from patchpulse.retrieval.agreement import (
        relevance_agreement,
        render_relevance_agreement,
        version_key,
    )
    from patchpulse.retrieval.audit import (
        AUDIT_PATH,
        GRADES_PATH,
        RelevanceAuditSession,
        read_audit,
        unread_pairs,
    )
    from patchpulse.retrieval.judging import JUDGMENTS_PATH

    audit = read_audit(AUDIT_PATH)
    judged = pd.read_csv(JUDGMENTS_PATH, dtype={"query_id": str})
    keys = list(zip(judged["query_id"], judged["review_id"].astype(int), strict=True))
    claude = dict(zip(keys, judged["grade"].astype(int), strict=True))
    judges = dict(zip(keys, judged["judge"].astype(str), strict=True))
    session = RelevanceAuditSession(audit, GRADES_PATH)
    seif = session.grades()
    reviews = pd.read_parquet(settings.data_dir / REVIEWS_FILE, columns=["review_id", "language"])
    languages = dict(zip(reviews["review_id"].astype(int), reviews["language"], strict=True))
    try:
        overall = relevance_agreement(claude, seif, audit, judges=judges)
        unread = unread_pairs(seif, session.unsure(), languages)
        read = [pair for pair in audit if pair not in unread]
        result = relevance_agreement(claude, seif, read, judges=judges)
    except ValueError as error:
        print(f"patchpulse-ml: {error}", file=sys.stderr)
        return 1
    versions = dict(zip(keys, judged["guidelines"].astype(str), strict=True))
    counts = Counter(versions[pair] for pair in read)
    latest = max(counts, key=version_key)
    path = REPORTS_DIR / f"judge-agreement-{latest}.md"  # one per version: the first check stays
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_relevance_agreement(result, overall=overall, guidelines=counts), encoding="utf-8"
    )
    kappa = "n/a" if result.kappa is None else f"{result.kappa:.3f}"
    every = "n/a" if overall.kappa is None else f"{overall.kappa:.3f}"
    print(
        f"weighted kappa {kappa} on the {result.n} pairs Seif read ({len(unread)} unread left "
        f"out): {'passed' if result.passed else 'failed'} (rule: 0.70); over all {overall.n}: "
        f"{every}; report in {path.as_posix()}"
    )
    return 0 if result.passed else 1


def _queries(*, freeze: bool) -> int:
    from patchpulse.retrieval.queries import (
        QueriesNotFrozenError,
        QueryFileError,
        freeze_queries,
        load_queries,
    )

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
