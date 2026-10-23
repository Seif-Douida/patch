"""Loading a collection from the laptop (plan Task 7, design D4): the eval's local collections, and
the upload of the pinned encoder's collection to the free cluster (plan Task 17), which then
points the alias at it.

Points go up in id order, in batches Qdrant confirms (`wait=True`). A progress file records the
last confirmed id for this index set (a hash of its ids and versions), so a rerun after a stop
resumes there and a new pull starts over; the same ids make a full rerun safe.
Vectors come from the embedding cache and must be of each review's current version.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from patchpulse.models.snapshot import Snapshot
from patchpulse.models.text import clean_text
from patchpulse.retrieval.embed import EmbeddingCache, version_seconds
from patchpulse.retrieval.embedders import MODEL_FILE, read_manifest
from patchpulse.retrieval.points import to_point
from patchpulse.retrieval.qdrant import ensure_collection, point_alias

log = logging.getLogger("patchpulse.retrieval")

BATCH = 256


class Upserter(Protocol):
    def upsert(self, collection_name: str, points: Any, *, wait: bool) -> Any: ...


@dataclass(frozen=True)
class LoadStats:
    upserted: int
    skipped: int  # confirmed by an earlier, interrupted run
    seconds: float


def _confirmed(progress: Path | None, resume: dict[str, str]) -> int:
    """The last review id an earlier run of this upload confirmed: same collection, index set and
    encoder (its cache), or none."""
    if progress is None or not progress.is_file():
        return -1
    state = json.loads(progress.read_text(encoding="utf-8"))
    same = all(state.get(key) == value for key, value in resume.items())
    return int(state["confirmed_through"]) if same else -1


def load_collection(
    client: Upserter,
    collection: str,
    rows: pd.DataFrame,
    texts: Mapping[int, str],
    cache: EmbeddingCache,
    *,
    batch: int = BATCH,
    progress: Path | None = None,
) -> LoadStats:
    """Upsert `rows` (`core.review_index` columns) with their texts (cleaned here) and cached
    vectors. Refuses a review whose cached vector is of another version."""
    started = time.perf_counter()
    ordered = rows.sort_values("review_id").reset_index(drop=True)
    cached_ids, cached_versions, vectors = cache.load()
    position = {int(rid): i for i, rid in enumerate(cached_ids.tolist())}
    versions = version_seconds(ordered["version"])
    digest = hashlib.sha256(np.asarray(ordered["review_id"], dtype=np.int64).tobytes())
    digest.update(versions.tobytes())
    resume = {
        "collection": collection,
        "index_set": digest.hexdigest()[:16],
        "encoder": cache.directory.name,
    }
    after = _confirmed(progress, resume)
    records = cast(list[dict[str, Any]], ordered.to_dict("records"))
    todo = [i for i, row in enumerate(records) if int(row["review_id"]) > after]
    for start in range(0, len(todo), batch):
        points = []
        for i in todo[start : start + batch]:
            rid = int(records[i]["review_id"])
            at = position.get(rid)
            if at is None or int(cached_versions[at]) != int(versions[i]):
                raise ValueError(
                    f"no vector for review {rid} at its current version: run `retrieval embed`"
                )
            points.append(to_point(records[i], clean_text(str(texts[rid])), vectors[at]))
        client.upsert(collection, points, wait=True)
        if progress is not None:
            last = int(records[todo[min(start + batch, len(todo)) - 1]]["review_id"])
            state = {**resume, "confirmed_through": last}
            progress.write_text(json.dumps(state) + "\n", encoding="utf-8")
        if (start // batch) % 100 == 99:
            log.info("%s: %d of %d points", collection, start + len(points), len(todo))
    return LoadStats(
        upserted=len(todo),
        skipped=len(records) - len(todo),
        seconds=round(time.perf_counter() - started, 1),
    )


def _cache(model_dir: Path, data_dir: Path) -> tuple[EmbeddingCache, int]:
    """The embedding cache of this build, and its dimensions."""
    manifest = read_manifest(model_dir)
    spec = manifest["spec"]
    cache = EmbeddingCache(
        data_dir,
        name=spec["name"],
        model_sha256=manifest["files"][MODEL_FILE],
        cleaning_version=manifest["cleaning_version"],
        dims=spec["dims"],
    )
    return cache, int(spec["dims"])


def upload_collection(
    client: Any,
    snapshot: Snapshot,
    model_dir: Path,
    *,
    data_dir: Path,
    collection: str,
    batch: int = BATCH,
) -> LoadStats:
    """The index set into `collection` (created with the design's settings if missing), from the
    cache of this build, resuming after the last confirmed batch."""
    cache, dims = _cache(model_dir, data_dir)
    ensure_collection(client, collection, dims)
    texts = dict(zip(snapshot.reviews["review_id"], snapshot.reviews["text"], strict=True))
    stats = load_collection(
        client,
        collection,
        snapshot.index,
        texts,
        cache,
        batch=batch,
        progress=data_dir / f"load-cloud-{collection}.json",
    )
    log.info(
        "%s: %d points uploaded, %d already there, in %.0f s",
        collection,
        stats.upserted,
        stats.skipped,
        stats.seconds,
    )
    return stats


def publish_alias(client: Any, alias: str, collection: str) -> None:
    """Point the alias the job and the API search through at the collection."""
    point_alias(client, alias, collection)
    log.info("alias %s now points at %s", alias, collection)


def eval_collection(name: str) -> str:
    return f"eval_{name}"


def build_local_collections(
    client: Any,
    snapshot: Snapshot,
    embedder_dirs: Mapping[str, Path],
    *,
    data_dir: Path,
    names: Sequence[str],
) -> dict[str, LoadStats]:
    """One collection per candidate (`eval_<name>`) with identical settings, from the cache."""
    texts = dict(zip(snapshot.reviews["review_id"], snapshot.reviews["text"], strict=True))
    stats = {}
    for name in names:
        cache, dims = _cache(embedder_dirs[name], data_dir)
        ensure_collection(client, eval_collection(name), dims)
        stats[name] = load_collection(
            client,
            eval_collection(name),
            snapshot.index,
            texts,
            cache,
            progress=data_dir / f"load-{eval_collection(name)}.json",
        )
        log.info(
            "%s: %d points loaded, %d already there, in %.0f s",
            eval_collection(name),
            stats[name].upserted,
            stats[name].skipped,
            stats[name].seconds,
        )
    return stats
