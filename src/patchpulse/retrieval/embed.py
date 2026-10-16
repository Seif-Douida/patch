"""Embedding the index set on Seif's machine (plan Task 6, design D4): once per candidate for the
eval, with the int8 file the nightly job and the queries use.

Vectors are cached in `data/local/embeddings/<name>-<model sha8>-<cleaning>/`, keyed by review id
and version: an edited review is embedded again, the rest reused. Another model file or another
text cleaning gets another directory, so vectors are never mixed. The cache grows in chunks, each
written whole under a temporary name and then renamed, so a stop resumes at the next chunk.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from patchpulse.models.snapshot import Snapshot
from patchpulse.models.text import clean_text
from patchpulse.retrieval.embedders import EmbedderSpec

log = logging.getLogger("patchpulse.retrieval")

CHUNK_SIZE = 10_000


class PassageEmbedder(Protocol):
    spec: EmbedderSpec
    model_sha256: str
    cleaning_version: str

    def embed_passages(self, texts: list[str], *, batch: int = 32) -> NDArray[np.float32]: ...


@dataclass(frozen=True)
class EmbedStats:
    embedded: int
    reused: int
    seconds: float


def version_seconds(versions: pd.Series) -> NDArray[np.int64]:
    """Review versions (timestamp_updated) as whole seconds since 1970."""
    stamps = pd.to_datetime(versions).astype("datetime64[s]")
    return np.asarray(stamps.astype("int64"), dtype=np.int64)


class EmbeddingCache:
    def __init__(
        self, data_dir: Path, *, name: str, model_sha256: str, cleaning_version: str, dims: int
    ) -> None:
        self.directory = data_dir / "embeddings" / f"{name}-{model_sha256[:8]}-{cleaning_version}"
        self.dims = dims

    @classmethod
    def for_embedder(cls, data_dir: Path, embedder: PassageEmbedder) -> EmbeddingCache:
        return cls(
            data_dir,
            name=embedder.spec.name,
            model_sha256=embedder.model_sha256,
            cleaning_version=embedder.cleaning_version,
            dims=embedder.spec.dims,
        )

    def _chunks(self) -> list[Path]:
        return sorted(self.directory.glob("chunk-*.npz")) if self.directory.is_dir() else []

    def load(self) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]]:
        """(ids, versions, vectors), sorted by id. A review in several chunks (an edit) keeps the
        vector of the latest chunk."""
        if not self._chunks():
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty.copy(), np.zeros((0, self.dims), dtype=np.float32)
        ids, versions, vectors = [], [], []
        for path in self._chunks():
            with np.load(path) as chunk:
                ids.append(chunk["ids"])
                versions.append(chunk["versions"])
                vectors.append(chunk["vectors"])
        all_ids = np.concatenate(ids)
        # The last occurrence of each id: unique over the reversed array, then back to positions.
        _, last = np.unique(all_ids[::-1], return_index=True)
        keep = len(all_ids) - 1 - last
        return (
            np.asarray(all_ids[keep], dtype=np.int64),
            np.asarray(np.concatenate(versions)[keep], dtype=np.int64),
            np.asarray(np.concatenate(vectors)[keep], dtype=np.float32),
        )

    def append(
        self, ids: NDArray[np.int64], versions: NDArray[np.int64], vectors: NDArray[np.float32]
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"chunk-{len(self._chunks()):05d}.npz"
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as file:
            np.savez(file, ids=ids, versions=versions, vectors=vectors)
        os.replace(temporary, path)


def embed_index_set(
    snapshot: Snapshot,
    embedder: PassageEmbedder,
    cache: EmbeddingCache,
    *,
    chunk_size: int = CHUNK_SIZE,
) -> EmbedStats:
    """Embed every index-set review the cache doesn't hold at its current version, in id order,
    one chunk at a time. Texts are cleaned as everywhere (`clean_text`, cut at 2,000 characters);
    each chunk is batched by length inside the embedder and stored in id order."""
    if snapshot.index.empty:
        raise ValueError("the snapshot has no index set: pull core.review_index first")
    started = time.perf_counter()
    index = snapshot.index.sort_values("review_id")
    ids = np.asarray(index["review_id"], dtype=np.int64)
    versions = version_seconds(index["version"])
    cached_ids, cached_versions, _ = cache.load()
    cached = dict(zip(cached_ids.tolist(), cached_versions.tolist(), strict=True))
    todo = np.array(
        [
            cached.get(rid) != version
            for rid, version in zip(ids.tolist(), versions.tolist(), strict=True)
        ]
    )
    texts = snapshot.reviews.set_index("review_id")["text"]
    todo_ids, todo_versions = ids[todo], versions[todo]
    for start in range(0, len(todo_ids), chunk_size):
        chunk_ids = todo_ids[start : start + chunk_size]
        chunk_texts = [clean_text(str(texts[rid])) for rid in chunk_ids.tolist()]
        vectors = embedder.embed_passages(chunk_texts)
        cache.append(chunk_ids, todo_versions[start : start + chunk_size], vectors)
        done = start + len(chunk_ids)
        elapsed = time.perf_counter() - started
        log.info(
            "%s: embedded %d of %d (%.1f a second)",
            embedder.spec.name,
            done,
            len(todo_ids),
            done / elapsed,
        )
    return EmbedStats(
        embedded=int(todo.sum()),
        reused=int(len(ids) - todo.sum()),
        seconds=round(time.perf_counter() - started, 1),
    )
