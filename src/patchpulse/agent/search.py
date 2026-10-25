"""search_reviews, the model's review search (design D3): Phase 3's hybrid search on Qdrant
(ADR-022), always within one tracked game, a window of days and the bundle's data version.

The model sees review ids, scores and snippets of at most SNIPPET_CHARS characters, cut from the
start of the text so a quote from a snippet is a quote from the review; the full text stays on
the run's trace for V3 and V7. Minimum playtime filters on the indexed playtime bucket, so it
must be one of the buckets' lower bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.retrieval.search import Hit, QueryEmbedder, SearchFilters, search

SNIPPET_CHARS = 400
DEFAULT_HITS = 8
MAX_HITS = 10
QUERY_LANGUAGE = "english"  # the model writes its queries in English (reports are in English)
# Hours at review -> the playtime buckets (fact_review.playtime_bucket) at or above it.
PLAYTIME_FLOORS: dict[int, tuple[str, ...]] = {
    2: ("2h_10h", "10h_50h", "50h_200h", "200h_plus"),
    10: ("10h_50h", "50h_200h", "200h_plus"),
    50: ("50h_200h", "200h_plus"),
    200: ("200h_plus",),
}


@dataclass(frozen=True)
class ReviewSearch:
    """Where search_reviews looks: a Qdrant client, the collection (or its alias) and the pinned
    query encoder."""

    client: Any
    collection: str
    embedder: QueryEmbedder


def open_pinned_search(
    url: str, api_key: SecretStr | None, encoder_dir: Path, *, pin_path: Path | None = None
) -> ReviewSearch:
    """The collection config/retrieval.yaml pins, queried with the encoder in `encoder_dir`, which
    must be the one the pin names (the index stage's check). The caller closes `client`."""
    # Imported here: the encoder needs onnxruntime, which the agent's other users don't.
    from patchpulse.models.artifact import sha256_file
    from patchpulse.retrieval.embedders import MANIFEST_FILE, OnnxEmbedder
    from patchpulse.retrieval.pin import PIN_PATH, read_pin
    from patchpulse.retrieval.qdrant import make_client

    path = pin_path or PIN_PATH
    pin = read_pin(path)
    if sha256_file(encoder_dir / MANIFEST_FILE) != pin.embedder_json_sha256:
        raise ValueError(f"the encoder in {encoder_dir} isn't the one {path} pins")
    embedder = OnnxEmbedder(encoder_dir)
    return ReviewSearch(make_client(url, api_key), pin.collection, embedder)


def snippet(text: str) -> str:
    """The start of `text`, at most SNIPPET_CHARS long, cut at a space when one falls in the
    second half."""
    if len(text) <= SNIPPET_CHARS:
        return text
    cut = text[:SNIPPET_CHARS]
    boundary = max(cut.rfind(" "), cut.rfind("\n"))
    return cut[:boundary].rstrip() if boundary >= SNIPPET_CHARS // 2 else cut


def _day_key(day: date) -> int:
    return day.year * 10_000 + day.month * 100 + day.day


def find_reviews(
    source: ReviewSearch,
    *,
    appid: int,
    query: str,
    window: tuple[date, date],
    as_of_run_id: int,
    voted_up: bool | None = None,
    aspect: str | None = None,
    language: str | None = None,
    min_playtime: float | None = None,
    k: int = DEFAULT_HITS,
) -> list[Hit]:
    """The best `k` reviews (capped at MAX_HITS) for `query`; ValueError names a filter the index
    can't apply."""
    if aspect is not None and aspect not in ASPECTS:
        raise ValueError(f"unknown aspect {aspect!r}")
    buckets = None
    if min_playtime is not None:
        if min_playtime not in PLAYTIME_FLOORS:
            floors = ", ".join(str(h) for h in list(PLAYTIME_FLOORS)[:-1])
            raise ValueError(
                f"min_playtime must be {floors} or {list(PLAYTIME_FLOORS)[-1]} hours: "
                "the index stores playtime buckets"
            )
        buckets = PLAYTIME_FLOORS[int(min_playtime)]
    filters = SearchFilters(
        start_day=_day_key(window[0]),
        end_day=_day_key(window[1]),
        voted_up=voted_up,
        aspect=aspect,
        review_language=language,
        as_of_run_id=as_of_run_id,
        playtime_buckets=buckets,
    )
    return search(
        source.client,
        source.collection,
        source.embedder,
        query,
        appid=appid,
        language=QUERY_LANGUAGE,
        filters=filters,
        k=max(1, min(k, MAX_HITS)),
    )
