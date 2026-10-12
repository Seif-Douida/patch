"""S1a (plan Task 12, design D5): frozen e5 embeddings and a logistic head per aspect. The head is
saved as plain arrays (.npz), never a pickle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tiny_encoder import write_tiny_encoder

from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.artifact import load_scorer
from patchpulse.models.encoder import ENCODER_FILE, TOKENIZER_FILE, OnnxEncoder
from patchpulse.models.s1a import (
    HEAD_FILE,
    S1aParams,
    S1aScorer,
    load_head,
    save_head,
    save_s1a,
    train_s1a,
)
from patchpulse.models.text import clean_texts


def toy_embeddings(n: int, *, seed: int) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Unit-length vectors that point towards the directions of their aspects."""
    rng = np.random.default_rng(seed)
    directions = np.random.default_rng(99).normal(size=(len(ASPECTS), 16))
    labels = rng.random((n, len(ASPECTS))) < 0.25
    vectors = labels @ directions + 0.3 * rng.normal(size=(n, 16))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors.astype(np.float32), labels


def test_s1a_learns_a_toy_multilabel_set() -> None:
    x, y = toy_embeddings(600, seed=1)
    unseen, truth = toy_embeddings(200, seed=2)

    probs = train_s1a(x, y).predict_proba(unseen)

    assert probs.shape == (200, len(ASPECTS))
    assert ((probs >= 0.5) == truth).mean() > 0.95


def test_the_head_is_the_same_model_as_scikit_learns() -> None:
    """Standardising is folded into the saved weights; probabilities must not change."""
    x, y = toy_embeddings(300, seed=3)
    params = S1aParams()

    head = train_s1a(x, y, params=params)

    for aspect in range(len(ASPECTS)):
        reference = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=params.c, class_weight="balanced", max_iter=params.max_iter),
        ).fit(x, y[:, aspect])
        np.testing.assert_allclose(
            head.predict_proba(x)[:, aspect], reference.predict_proba(x)[:, 1], atol=1e-5
        )


def test_an_aspect_without_both_classes_is_refused() -> None:
    x, y = toy_embeddings(50, seed=4)
    y[:, 3] = False

    with pytest.raises(ValueError, match=ASPECTS[3]):
        train_s1a(x, y)


def test_head_round_trips_as_npz(tmp_path: Path) -> None:
    x, y = toy_embeddings(200, seed=5)
    head = train_s1a(x, y)

    save_head(head, tmp_path / HEAD_FILE)
    loaded = load_head(tmp_path / HEAD_FILE)

    assert np.array_equal(loaded.predict_proba(x), head.predict_proba(x))
    with np.load(tmp_path / HEAD_FILE, allow_pickle=False) as arrays:
        assert set(arrays.files) == {"coef", "intercept"}


def test_s1a_artifact_loads_through_the_common_scorer(tmp_path: Path) -> None:
    encoder_dir, model_dir = tmp_path / "encoder", tmp_path / "model"
    write_tiny_encoder(encoder_dir)
    encoder = OnnxEncoder(encoder_dir)
    texts = ["good game", "bad fps drops", "great story", "server down crashes"] * 10
    labels = np.random.default_rng(6).random((len(texts), len(ASPECTS))) < 0.5
    head = train_s1a(encoder.embed(clean_texts(texts)), labels)
    cuts = np.linspace(0.3, 0.7, len(ASPECTS))

    artifact = save_s1a(head, encoder_dir, model_dir, thresholds=cuts, version="t")
    scorer = load_scorer(model_dir)
    odd = ["", "[b]great[/b] story https://example.com", "\U0001f525" * 3, "good " * 2_000]
    probs = scorer.predict_proba(odd)

    assert isinstance(scorer, S1aScorer)
    assert artifact.family == "s1a"
    assert set(artifact.files) == {ENCODER_FILE, TOKENIZER_FILE, HEAD_FILE}
    expected = head.predict_proba(encoder.embed(clean_texts(odd)))
    np.testing.assert_allclose(probs, expected, atol=1e-6)
    assert np.array_equal(scorer.labels(probs), probs >= cuts)
