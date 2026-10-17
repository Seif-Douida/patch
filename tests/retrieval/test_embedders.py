"""The retrieval embedders (plan Task 4, design D3): pooling, prefixes, Matryoshka truncation and
the checks on a build's files, on tiny fakes whose outputs can be worked out by hand."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from tiny_encoder import DIM, embedder_ids, write_tiny_embedder

from patchpulse.retrieval.embedders import (
    CANDIDATES,
    MANIFEST_FILE,
    MODEL_FILE,
    TOKENIZER_FILE,
    EmbedderIntegrityError,
    EmbedderSpec,
    OnnxEmbedder,
)

MEAN = EmbedderSpec(
    name="tiny-mean",
    model_id="test/tiny-mean",
    revision="0" * 40,
    onnx_file="onnx/model.onnx",
    tokenizer_file="tokenizer.json",
    pooling="mean",
    query_prefix="query: ",
    passage_prefix="passage: ",
    max_tokens=16,
    dims=DIM,
    licence="MIT",
)
# Like arctic-embed: the first token's vector, a prefix for queries only.
CLS = replace(MEAN, name="tiny-cls", pooling="cls", passage_prefix="")


def unit(row: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.asarray(row / np.linalg.norm(row), dtype=np.float32)


def by_hand_mean(
    table: NDArray[np.float32], text: str, prefix: str, max_tokens: int = 16
) -> NDArray[np.float32]:
    return unit(table[embedder_ids(text, prefix=prefix, max_tokens=max_tokens)].mean(axis=0))


def test_candidates_are_e5_small_and_minilm() -> None:
    # e5-base, under the speed floor, was dropped on 17 Oct before its backlog was embedded.
    assert {name: (s.pooling, s.max_tokens, s.dims) for name, s in CANDIDATES.items()} == {
        "e5-small": ("mean", 512, 384),
        "minilm-l12": ("mean", 128, 384),
    }
    assert all(len(s.revision) == 40 for s in CANDIDATES.values())  # pinned commits


def test_mean_and_cls_pooling(tmp_path: Path) -> None:
    table = write_tiny_embedder(tmp_path / "mean", MEAN)
    mixed = write_tiny_embedder(tmp_path / "cls", CLS, mix=True)
    texts = ["fps drops", "the game crashes and the server is down again"]  # padding in a batch

    assert OnnxEmbedder(tmp_path / "mean").quantization == "fp32"  # recorded in embedder.json
    mean = OnnxEmbedder(tmp_path / "mean").embed_passages(texts)
    cls = OnnxEmbedder(tmp_path / "cls").embed_passages(texts)

    for row, text in zip(mean, texts, strict=True):
        np.testing.assert_allclose(row, by_hand_mean(table, text, "passage: "), atol=1e-5)
    for row, text in zip(cls, texts, strict=True):
        ids = embedder_ids(text, prefix="", max_tokens=16)
        # The first token's own row plus the masked sum of the text: never the padding.
        np.testing.assert_allclose(row, unit(mixed[ids[0]] + mixed[ids].sum(axis=0)), atol=1e-5)


def test_query_and_passage_prefixes_by_spec(tmp_path: Path) -> None:
    table = write_tiny_embedder(tmp_path / "mean", MEAN)
    mixed = write_tiny_embedder(tmp_path / "cls", CLS, mix=True)
    mean, cls = OnnxEmbedder(tmp_path / "mean"), OnnxEmbedder(tmp_path / "cls")

    query, passage = mean.embed_queries(["game"])[0], mean.embed_passages(["game"])[0]

    np.testing.assert_allclose(query, by_hand_mean(table, "game", "query: "), atol=1e-5)
    np.testing.assert_allclose(passage, by_hand_mean(table, "game", "passage: "), atol=1e-5)
    assert not np.allclose(query, passage)
    # CLS-style spec: queries get "query: ", passages nothing.
    for prefix, row in (
        ("query: ", cls.embed_queries(["game"])[0]),
        ("", cls.embed_passages(["game"])[0]),
    ):
        ids = embedder_ids("game", prefix=prefix, max_tokens=16)
        np.testing.assert_allclose(row, unit(mixed[ids[0]] + mixed[ids].sum(axis=0)), atol=1e-5)
    assert cls.spec == CLS


def test_matryoshka_truncation_is_renormalised(tmp_path: Path) -> None:
    short = replace(MEAN, name="tiny-short", dims=2)
    table = write_tiny_embedder(tmp_path / "short", short)

    rows = OnnxEmbedder(tmp_path / "short").embed_passages(["great story", "bad game"])

    assert rows.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(rows, axis=1), 1.0, atol=1e-6)
    pooled = table[embedder_ids("great story", prefix="passage: ", max_tokens=16)].mean(axis=0)
    np.testing.assert_allclose(rows[0], unit(pooled[:2]), atol=1e-5)


def test_embedders_accept_odd_texts(tmp_path: Path) -> None:
    write_tiny_embedder(tmp_path / "mean", MEAN)
    write_tiny_embedder(tmp_path / "cls", CLS, mix=True)
    odd = [
        "",
        "\U0001f600\U0001f600\U0001f600",
        "lag " * 2_000,  # 8,000 characters, cut to max_tokens
        "游戏更新后经常崩溃",
        "fps 崩溃 drops фпс",
    ]

    for directory in ("mean", "cls"):
        embedder = OnnxEmbedder(tmp_path / directory)
        for rows in (embedder.embed_passages(odd), embedder.embed_queries(odd)):
            assert rows.shape == (len(odd), DIM)
            assert np.isfinite(rows).all()
            np.testing.assert_allclose(np.linalg.norm(rows, axis=1), 1.0, atol=1e-5)
    assert OnnxEmbedder(tmp_path / "mean").embed_passages([]).shape == (0, DIM)


@pytest.mark.parametrize("damage", ["model", "tokenizer", "manifest", "spec"])
def test_altered_embedder_files_are_refused(tmp_path: Path, damage: str) -> None:
    write_tiny_embedder(tmp_path, MEAN)
    OnnxEmbedder(tmp_path)  # intact: loads
    if damage == "model":
        (tmp_path / MODEL_FILE).write_bytes((tmp_path / MODEL_FILE).read_bytes() + b"\0")
        expected = MODEL_FILE
    elif damage == "tokenizer":
        (tmp_path / TOKENIZER_FILE).write_text("{}", encoding="utf-8")
        expected = TOKENIZER_FILE
    elif damage == "manifest":
        (tmp_path / MANIFEST_FILE).unlink()
        expected = MANIFEST_FILE
    else:  # a manifest whose spec has an unknown field
        manifest = (tmp_path / MANIFEST_FILE).read_text(encoding="utf-8")
        (tmp_path / MANIFEST_FILE).write_text(
            manifest.replace('"pooling"', '"extra": 1, "pooling"'), encoding="utf-8"
        )
        expected = MANIFEST_FILE

    with pytest.raises(EmbedderIntegrityError, match=expected):
        OnnxEmbedder(tmp_path)
