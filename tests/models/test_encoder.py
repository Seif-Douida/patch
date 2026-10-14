"""The e5 encoder on ONNX Runtime (plan Task 12): a "query: " prefix, mean pooling over real
tokens only, unit length. Checked on a tiny fake whose answers can be worked out by hand."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tiny_encoder import DIM, expected_embedding, write_tiny_encoder

from patchpulse.models.encoder import OnnxEncoder, l2_normalise, mean_pool


def test_mean_pool_counts_only_real_tokens() -> None:
    hidden = np.array([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]])

    assert mean_pool(hidden, mask).tolist() == [[2.0, 2.0]]


def test_l2_normalise_gives_unit_rows_and_leaves_zero_rows() -> None:
    rows = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

    np.testing.assert_allclose(l2_normalise(rows), [[0.6, 0.8], [0.0, 0.0]], atol=1e-7)


def test_pooling_and_normalisation_on_a_tiny_fake(tmp_path: Path) -> None:
    table = write_tiny_encoder(tmp_path)
    encoder = OnnxEncoder(tmp_path)
    texts = ["good game", "bad", "great story fps drops", "", "unknown words here"]

    together = encoder.embed(texts, batch=4)
    one_by_one = np.vstack([encoder.embed([text], batch=1) for text in texts])

    assert together.shape == (len(texts), DIM)
    assert together.dtype == np.float32
    np.testing.assert_allclose(together, one_by_one, atol=1e-6)  # padding never leaks in
    expected = np.vstack([expected_embedding(text, table) for text in texts])
    np.testing.assert_allclose(together, expected, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(together, axis=1), 1.0, atol=1e-6)


def test_texts_are_cut_at_max_tokens(tmp_path: Path) -> None:
    table = write_tiny_encoder(tmp_path)
    long = "good " * 50

    embedded = OnnxEncoder(tmp_path).embed([long], max_tokens=5)

    np.testing.assert_allclose(
        embedded[0], expected_embedding(long, table, max_tokens=5), atol=1e-6
    )


def test_no_texts_give_an_empty_matrix(tmp_path: Path) -> None:
    write_tiny_encoder(tmp_path)

    assert OnnxEncoder(tmp_path).embed([]).shape == (0, DIM)


def test_a_missing_encoder_is_named(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="encoder"):
        OnnxEncoder(tmp_path)
