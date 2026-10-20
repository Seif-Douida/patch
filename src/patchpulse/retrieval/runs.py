"""The eval's runs (design D7): every system's top 20 for every frozen query, and their pool.

A system is BM25, or one candidate's dense or hybrid search. Each runs twice per query: `base`
(the game only) and `filtered` (plus the query's window, thumbs and aspect; the same as base for a
query without any). A cross-lingual query searches French reviews in both. BM25 doesn't depend on
the embedder, so it runs once, on the first candidate's collection (every collection holds the same
reviews and BM25 settings).

The runs are written to `data/retrieval/runs/` and committed: ids, ranks and scores only. The pool,
the union of every run's top 20 per query, is what gets judged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from patchpulse.retrieval.embedders import CANDIDATES
from patchpulse.retrieval.load import eval_collection
from patchpulse.retrieval.queries import load_queries, require_frozen
from patchpulse.retrieval.search import QueryEmbedder, search

VARIANTS = ("base", "filtered")
DEPTH = 20
RUN_COLUMNS = ["query_id", "rank", "review_id", "score"]
POOL_COLUMNS = ["query_id", "review_id"]

# (system, variant) -> query_id, rank (from 1), review_id, score.
Runs = dict[tuple[str, str], pd.DataFrame]
# query_id, review_id: one row per pair to judge.
Pool = pd.DataFrame


def systems(names: Sequence[str]) -> list[str]:
    """BM25, then each candidate's dense and hybrid search."""
    return ["bm25", *(f"{mode}:{name}" for name in names for mode in ("dense", "hybrid"))]


SYSTEMS = tuple(systems(list(CANDIDATES)))


def run_systems(
    client: Any,
    queries_path: Path,
    embedders: Mapping[str, QueryEmbedder],
    *,
    collection: Callable[[str], str] = eval_collection,
    depth: int = DEPTH,
    games: set[int] | None = None,
) -> Runs:
    """Every system's top `depth` for every query of the frozen file. The file is checked against
    its frozen hash before it's read, so a run can't use queries that changed."""
    require_frozen(queries_path)
    queries = load_queries(queries_path, games=games)
    names = list(embedders)
    rows: dict[tuple[str, str], list[tuple[str, int, int, float]]] = {
        (system, variant): [] for system in systems(names) for variant in VARIANTS
    }
    for query in queries:
        for system in systems(names):
            mode, _, name = system.partition(":")
            name = name or names[0]
            found: dict[str, list[tuple[str, int, int, float]]] = {}
            for variant in VARIANTS:
                if variant == "filtered" and not query.has_extra_filters:
                    found[variant] = found["base"]
                    continue
                hits = search(
                    client,
                    collection(name),
                    embedders[name],
                    query.text,
                    appid=query.appid,
                    language=query.language,
                    filters=query.filters if variant == "filtered" else query.base_filters,
                    mode=mode,
                    k=depth,
                )
                found[variant] = [
                    (query.query_id, rank, hit.review_id, hit.score)
                    for rank, hit in enumerate(hits, start=1)
                ]
            for variant in VARIANTS:
                rows[(system, variant)].extend(found[variant])
    return {key: pd.DataFrame(value, columns=RUN_COLUMNS) for key, value in rows.items()}


def run_file(system: str, variant: str) -> str:
    """`dense:e5-small`, `base` -> `dense_e5-small-base.csv`: no colon (Windows file names)."""
    return f"{system.replace(':', '_')}-{variant}.csv"


def write_runs(runs: Runs, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for (system, variant), frame in runs.items():
        path = out_dir / run_file(system, variant)
        frame[RUN_COLUMNS].sort_values(["query_id", "rank"]).to_csv(
            path, index=False, lineterminator="\n", float_format="%.6f"
        )
        paths.append(path)
    return paths


def read_runs(out_dir: Path, system_names: Sequence[str] = SYSTEMS) -> Runs:
    return {
        (system, variant): pd.read_csv(
            out_dir / run_file(system, variant), dtype={"query_id": str}
        )[RUN_COLUMNS]
        for system in system_names
        for variant in VARIANTS
        if (out_dir / run_file(system, variant)).is_file()
    }


def pool(runs: Runs, depth: int = DEPTH) -> Pool:
    """The union of every run's top `depth` per query, one row per pair, in id order."""
    frames = [frame.loc[frame["rank"] <= depth, POOL_COLUMNS] for frame in runs.values()]
    if not frames:
        return pd.DataFrame(columns=POOL_COLUMNS)
    pairs = pd.concat(frames).drop_duplicates()
    pairs["review_id"] = pairs["review_id"].astype("int64")
    return pairs.sort_values(POOL_COLUMNS).reset_index(drop=True)
