"""Search over a review collection (design D6): hybrid, dense or BM25, always within one game.

The one module that searches: the eval, the nightly health check and Phase 5's `search_reviews`
all call `search`. Hybrid runs Qdrant's Query API: a dense and a BM25 prefetch under the same
filter, fused by reciprocal rank (RRF). The query is cleaned and cut like a review; its own
language picks the BM25 options, so matching across languages is the dense vector's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from qdrant_client import models

from patchpulse.models.text import clean_text
from patchpulse.retrieval.qdrant import BM25_MODEL, DENSE, SPARSE, bm25_options

MODES = ("hybrid", "dense", "bm25")
PREFETCH = 50  # each side of a hybrid search, at least


class QueryEmbedder(Protocol):
    def embed_queries(self, texts: list[str], *, batch: int = 32) -> NDArray[np.float32]: ...


@dataclass(frozen=True)
class SearchFilters:
    """Optional filters beside the game: a window of days (yyyymmdd, inclusive), the thumbs, an
    aspect, the reviews' language (Steam's code), the data version (reviews first seen up to a
    pipeline run) and playtime buckets at review (any of them)."""

    start_day: int | None = None
    end_day: int | None = None
    voted_up: bool | None = None
    aspect: str | None = None
    review_language: str | None = None
    as_of_run_id: int | None = None
    playtime_buckets: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Hit:
    review_id: int
    score: float
    text: str


def query_filter(appid: int, filters: SearchFilters) -> models.Filter:
    must: list[models.Condition] = [
        models.FieldCondition(key="appid", match=models.MatchValue(value=appid))
    ]
    if filters.start_day is not None or filters.end_day is not None:
        must.append(
            models.FieldCondition(
                key="day", range=models.Range(gte=filters.start_day, lte=filters.end_day)
            )
        )
    if filters.voted_up is not None:
        must.append(
            models.FieldCondition(key="voted_up", match=models.MatchValue(value=filters.voted_up))
        )
    if filters.aspect is not None:
        must.append(
            models.FieldCondition(key="aspects", match=models.MatchValue(value=filters.aspect))
        )
    if filters.review_language is not None:
        must.append(
            models.FieldCondition(
                key="language", match=models.MatchValue(value=filters.review_language)
            )
        )
    if filters.as_of_run_id is not None:
        must.append(
            models.FieldCondition(
                key="first_seen_run_id", range=models.Range(lte=filters.as_of_run_id)
            )
        )
    if filters.playtime_buckets is not None:
        must.append(
            models.FieldCondition(
                key="playtime_bucket", match=models.MatchAny(any=list(filters.playtime_buckets))
            )
        )
    return models.Filter(must=must)


def search(
    client: Any,
    collection: str,
    embedder: QueryEmbedder,
    query: str,
    *,
    appid: int,
    language: str,
    filters: SearchFilters | None = None,
    mode: str = "hybrid",
    k: int = 20,
    exact: bool = False,
) -> list[Hit]:
    """The `k` best reviews of one game for `query`. An empty query returns no hits without
    asking Qdrant; `exact=True` searches the dense vectors without the HNSW index."""
    if mode not in MODES:
        raise ValueError(f"unknown search mode {mode!r}; the modes are {', '.join(MODES)}")
    text = clean_text(query)
    if not text:
        return []
    where = query_filter(appid, filters or SearchFilters())
    sparse = models.Document(text=text, model=BM25_MODEL, options=bm25_options(language))
    params = models.SearchParams(
        exact=exact, quantization=models.QuantizationSearchParams(rescore=True)
    )
    if mode == "bm25":
        response = client.query_points(
            collection, query=sparse, using=SPARSE, query_filter=where, limit=k,
            with_payload=["text"],
        )  # fmt: skip
    else:
        dense = [float(x) for x in embedder.embed_queries([text])[0]]
        if mode == "dense":
            response = client.query_points(
                collection, query=dense, using=DENSE, query_filter=where, limit=k,
                search_params=params, with_payload=["text"],
            )  # fmt: skip
        else:
            depth = max(k, PREFETCH)
            response = client.query_points(
                collection,
                prefetch=[
                    models.Prefetch(
                        query=dense, using=DENSE, filter=where, limit=depth, params=params
                    ),
                    models.Prefetch(query=sparse, using=SPARSE, filter=where, limit=depth),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k,
                with_payload=["text"],
            )
    return [
        Hit(review_id=int(point.id), score=float(point.score), text=str(point.payload["text"]))
        for point in response.points
    ]
