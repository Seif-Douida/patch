"""Embedding the index set once on the laptop (plan Task 6, design D4): a cache of vectors by review
id and version, written chunk by chunk so a stop resumes, and never shared across models or text
cleanings."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from tiny_encoder import write_tiny_embedder

from patchpulse.models.snapshot import Snapshot
from patchpulse.models.text import clean_text
from patchpulse.retrieval.embed import EmbeddingCache, embed_index_set
from patchpulse.retrieval.embedders import EmbedderSpec, OnnxEmbedder

TINY = EmbedderSpec(
    name="tiny",
    model_id="test/tiny",
    revision="0" * 40,
    onnx_file="onnx/model.onnx",
    tokenizer_file="tokenizer.json",
    pooling="mean",
    query_prefix="query: ",
    passage_prefix="passage: ",
    max_tokens=16,
    dims=4,
    licence="MIT",
)
WORDS = ("good", "bad", "game", "fps", "drops", "crashes", "great", "story", "server", "down")
VERSION = pd.Timestamp("2026-10-01 12:00:00")


def text_for(rid: int) -> str:
    # Lengths vary with the id, so length-sorted batches don't follow id order.
    return "[b]" + " ".join(WORDS[(rid + i) % len(WORDS)] for i in range(1 + rid * 7 % 13)) + "[/b]"


def snapshot(ids: Sequence[int], *, versions: dict[int, pd.Timestamp] | None = None) -> Snapshot:
    versions = versions or {}
    reviews = pd.DataFrame({"review_id": list(ids), "text": [text_for(rid) for rid in ids]})
    index = pd.DataFrame(
        {"review_id": list(ids), "version": [versions.get(rid, VERSION) for rid in ids]}
    )
    frame = pd.DataFrame({"appid": [553850]})
    return Snapshot(reviews=reviews, games=frame, patches=frame, index=index)


@pytest.fixture
def embedder(tmp_path: Path) -> OnnxEmbedder:
    write_tiny_embedder(tmp_path / "tiny", TINY)
    return OnnxEmbedder(tmp_path / "tiny")


class StopsAfter:
    """The embedder, interrupted (like a closed laptop) on its `calls`-th chunk."""

    def __init__(self, embedder: OnnxEmbedder, calls: int) -> None:
        self.embedder, self.calls = embedder, calls
        self.spec, self.model_sha256 = embedder.spec, embedder.model_sha256
        self.cleaning_version = embedder.cleaning_version

    def embed_passages(self, texts: Sequence[str], *, batch: int = 32) -> NDArray[np.float32]:
        self.calls -= 1
        if self.calls == 0:
            raise KeyboardInterrupt
        return self.embedder.embed_passages(texts, batch=batch)


def test_cache_resumes_after_a_stop(tmp_path: Path, embedder: OnnxEmbedder) -> None:
    ids = list(range(1, 11))
    cache = EmbeddingCache.for_embedder(tmp_path / "local", embedder)

    with pytest.raises(KeyboardInterrupt):
        embed_index_set(snapshot(ids), StopsAfter(embedder, 3), cache, chunk_size=3)

    assert len(cache.load()[0]) == 6  # two whole chunks; nothing half-written
    assert not list(cache.directory.glob("*.tmp*"))
    stats = embed_index_set(snapshot(ids), embedder, cache, chunk_size=3)
    assert (stats.embedded, stats.reused) == (4, 6)
    assert list(cache.load()[0]) == ids


def test_a_new_version_is_reembedded_and_others_reused(
    tmp_path: Path, embedder: OnnxEmbedder
) -> None:
    ids = [3, 1, 2]
    cache = EmbeddingCache.for_embedder(tmp_path / "local", embedder)
    embed_index_set(snapshot(ids), embedder, cache)
    edited = snapshot(ids, versions={2: VERSION + pd.Timedelta(days=1)})
    edited.reviews.loc[edited.reviews["review_id"] == 2, "text"] = "great story"

    stats = embed_index_set(edited, embedder, cache)

    assert (stats.embedded, stats.reused) == (1, 2)
    loaded_ids, versions, vectors = cache.load()
    assert list(loaded_ids) == [1, 2, 3]
    assert versions[1] == int((VERSION + pd.Timedelta(days=1)).timestamp())
    np.testing.assert_allclose(vectors[1], embedder.embed_passages(["great story"])[0], atol=1e-6)


def test_another_model_or_cleaning_never_reuses_vectors(
    tmp_path: Path, embedder: OnnxEmbedder
) -> None:
    ids = [1, 2, 3]
    embed_index_set(snapshot(ids), embedder, EmbeddingCache.for_embedder(tmp_path, embedder))
    other_spec = EmbedderSpec(**{**TINY.__dict__, "name": "tiny-two"})
    write_tiny_embedder(tmp_path / "two", other_spec)
    other = OnnxEmbedder(tmp_path / "two")
    other_cleaning = EmbeddingCache(
        tmp_path, name="tiny", model_sha256=embedder.model_sha256, cleaning_version="c2", dims=4
    )

    for model, cache in (
        (other, EmbeddingCache.for_embedder(tmp_path, other)),
        (embedder, other_cleaning),
    ):
        assert embed_index_set(snapshot(ids), model, cache).embedded == 3
    assert len({p.name for p in (tmp_path / "embeddings").iterdir()}) == 3


def test_output_order_is_the_id_order_whatever_the_batching(
    tmp_path: Path, embedder: OnnxEmbedder
) -> None:
    ids = [9, 2, 7, 4, 1, 8, 3]
    cache = EmbeddingCache.for_embedder(tmp_path, embedder)

    embed_index_set(snapshot(ids), embedder, cache, chunk_size=4)

    loaded_ids, versions, vectors = cache.load()
    assert list(loaded_ids) == sorted(ids)
    assert set(versions) == {int(VERSION.timestamp())}
    for rid, vector in zip(loaded_ids, vectors, strict=True):
        alone = embedder.embed_passages([clean_text(text_for(int(rid)))])[0]
        np.testing.assert_allclose(vector, alone, atol=1e-6)


def test_embedding_needs_the_index_set(tmp_path: Path, embedder: OnnxEmbedder) -> None:
    frame = pd.DataFrame({"appid": [553850]})
    empty = Snapshot(
        reviews=pd.DataFrame({"review_id": [1], "text": ["x"]}), games=frame, patches=frame
    )

    with pytest.raises(ValueError, match="index set"):
        embed_index_set(empty, embedder, EmbeddingCache.for_embedder(tmp_path, embedder))
