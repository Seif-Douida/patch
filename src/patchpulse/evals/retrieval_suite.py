"""The retrieval suite (Phase 6 design D11): Phase 3's queries on a small pool collection, in CI.

The pool is the reviews Phase 3's judges graded (its 5,930 pairs): their `core.review_index` rows,
cleaned text and pinned-encoder vectors, exported once on the laptop into the eval bundle. CI loads
it into a Qdrant container and runs hybrid search with each query's base filters (the production
system), k=20. The queries' vectors are recorded with it, so CI needs no encoder. These numbers
guard changes; on a pool of judged reviews only, they aren't Phase 3's full-index numbers.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from patchpulse.models.snapshot import Snapshot
from patchpulse.models.text import clean_text
from patchpulse.retrieval.embed import EmbeddingCache, version_seconds
from patchpulse.retrieval.metrics import Grades, mean_over_queries, ndcg_at, recall_at
from patchpulse.retrieval.points import to_point
from patchpulse.retrieval.qdrant import ensure_collection
from patchpulse.retrieval.queries import Query
from patchpulse.retrieval.rules import INDEX_COLUMNS
from patchpulse.retrieval.search import QueryEmbedder, search

log = logging.getLogger("patchpulse.evals")

POOL_COLLECTION = "eval_pool"
QUERIES_PATH = Path("data") / "retrieval" / "queries.csv"  # Phase 3's frozen queries
POOL_BATCH = 256
DEPTH = 20  # hits per query, as Phase 3's runs
POOL_COLUMNS = (*INDEX_COLUMNS, "text", "vector")


def grades_of(judgments: pd.DataFrame) -> dict[str, dict[int, int]]:
    """Query id -> review id -> grade, from Phase 3's judgments."""
    return {
        str(query_id): dict(
            zip(rows["review_id"].astype(int), rows["grade"].astype(int), strict=True)
        )
        for query_id, rows in judgments.groupby("query_id")
    }


def export_pool(snapshot: Snapshot, judgments: pd.DataFrame, cache: EmbeddingCache) -> pd.DataFrame:
    """The judged reviews' index rows, cleaned text and cached vector, by review id. A review
    the snapshot's index set no longer holds, or whose cached vector is of another version, is
    skipped; how many is in `attrs["skipped"]`."""
    judged = {int(rid) for rid in judgments["review_id"]}
    rows = snapshot.index[snapshot.index["review_id"].astype(int).isin(judged)]
    rows = rows.sort_values("review_id").reset_index(drop=True)
    cached_ids, cached_versions, vectors = cache.load()
    position = {int(rid): i for i, rid in enumerate(cached_ids.tolist())}
    versions = version_seconds(rows["version"])
    texts = snapshot.reviews.set_index("review_id")["text"]
    kept: list[dict[str, Any]] = []
    for i, row in enumerate(cast(list[dict[str, Any]], rows.to_dict("records"))):
        rid = int(row["review_id"])
        at = position.get(rid)
        if at is None or int(cached_versions[at]) != int(versions[i]) or rid not in texts.index:
            continue
        vector = [float(x) for x in vectors[at]]
        kept.append({**row, "text": clean_text(str(texts[rid])), "vector": vector})
    pool = pd.DataFrame(kept, columns=list(POOL_COLUMNS))
    pool.attrs["skipped"] = len(judged) - len(kept)
    log.info("pool: %d judged reviews, %d skipped (not in the index set or no vector at their "
             "version)", len(kept), pool.attrs["skipped"])  # fmt: skip
    return pool


def export_query_vectors(
    queries: Sequence[Query], embedder: QueryEmbedder
) -> dict[str, list[float]]:
    """Each query's vector, keyed by its cleaned text: what `search` hands the embedder."""
    texts = list(dict.fromkeys(clean_text(query.text) for query in queries))
    vectors = embedder.embed_queries(texts)
    return {text: [float(x) for x in vector] for text, vector in zip(texts, vectors, strict=True)}


class RecordedQueryEmbedder:
    """A QueryEmbedder that answers from recorded vectors only (CI has no encoder)."""

    def __init__(self, vectors: Mapping[str, Sequence[float]]) -> None:
        self._vectors = {text: np.asarray(v, dtype=np.float32) for text, v in vectors.items()}

    def embed_queries(self, texts: list[str], *, batch: int = 32) -> NDArray[np.float32]:
        missing = sum(1 for text in texts if text not in self._vectors)
        if missing:
            raise ValueError(
                f"no recorded vector for {missing} query text(s): rebuild the eval bundle"
            )
        return np.stack([self._vectors[text] for text in texts])


def load_pool(client: Any, pool: pd.DataFrame, *, collection: str = POOL_COLLECTION) -> None:
    """The pool into `collection`, created with the design's settings, in confirmed batches."""
    if pool.empty:
        raise ValueError("the pool is empty: nothing to search")
    ensure_collection(client, collection, len(pool["vector"].iloc[0]))
    records = cast(list[dict[str, Any]], pool.to_dict("records"))
    for start in range(0, len(records), POOL_BATCH):
        points = [
            to_point(row, str(row["text"]), np.asarray(row["vector"], dtype=np.float32))
            for row in records[start : start + POOL_BATCH]
        ]
        client.upsert(collection, points, wait=True)


@dataclass(frozen=True)
class RetrievalScores:
    ndcg_at_10: float | None
    recall_at_20: float | None
    per_query: dict[str, tuple[float | None, float | None]]  # (NDCG@10, Recall@20)


def run_retrieval(
    client: Any,
    queries: Sequence[Query],
    grades: Mapping[str, Grades],
    embedder: QueryEmbedder,
    *,
    collection: str = POOL_COLLECTION,
) -> RetrievalScores:
    """Hybrid search with each query's base filters, k=20, scored as Phase 3 scored it; a query
    with no relevant judged review is left out of the means. Repeated loads of the real pool
    vary by about 0.002 NDCG@10 (ties in the fused ranking), a tenth of the gate's tolerance."""
    per_query: dict[str, tuple[float | None, float | None]] = {}
    for query in queries:
        hits = search(
            client,
            collection,
            embedder,
            query.text,
            appid=query.appid,
            language=query.language,
            filters=query.base_filters,
            mode="hybrid",
            k=DEPTH,
        )
        ranked = [hit.review_id for hit in hits]
        judged = grades.get(query.query_id, {})
        per_query[query.query_id] = (ndcg_at(ranked, judged, 10), recall_at(ranked, judged, 20))
    ndcg, _, _ = mean_over_queries([values[0] for values in per_query.values()])
    recall, _, _ = mean_over_queries([values[1] for values in per_query.values()])
    return RetrievalScores(ndcg, recall, per_query)
