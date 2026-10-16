"""Loading a collection from the laptop (plan Task 7, design D4): the eval's local collections, and
later the upload to the free cluster.

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
from patchpulse.retrieval.qdrant import ensure_collection

log = logging.getLogger("patchpulse.retrieval")

BATCH = 256


class Upserter(Protocol):
    def upsert(self, collection_name: str, points: Any, *, wait: bool) -> Any: ...


@dataclass(frozen=True)
class LoadStats:
    upserted: int
    skipped: int  # confirmed by an earlier, interrupted run
    seconds: float


def _confirmed(progress: Path | None, collection: str, index_set: str) -> int:
    if progress is None or not progress.is_file():
        return -1
    state = json.loads(progress.read_text(encoding="utf-8"))
    same = state.get("collection") == collection and state.get("index_set") == index_set
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
    index_set = digest.hexdigest()[:16]
    after = _confirmed(progress, collection, index_set)
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
            state = {"collection": collection, "index_set": index_set, "confirmed_through": last}
            progress.write_text(json.dumps(state) + "\n", encoding="utf-8")
        if (start // batch) % 100 == 99:
            log.info("%s: %d of %d points", collection, start + len(points), len(todo))
    return LoadStats(
        upserted=len(todo),
        skipped=len(records) - len(todo),
        seconds=round(time.perf_counter() - started, 1),
    )


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
        manifest = read_manifest(embedder_dirs[name])
        spec = manifest["spec"]
        cache = EmbeddingCache(
            data_dir,
            name=spec["name"],
            model_sha256=manifest["files"][MODEL_FILE],
            cleaning_version=manifest["cleaning_version"],
            dims=spec["dims"],
        )
        ensure_collection(client, eval_collection(name), int(spec["dims"]))
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
