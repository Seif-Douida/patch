"""The classification suite (Phase 6 plan Task 9, design D11): the champion student on gold test,
raw texts through `predict_proba` as the nightly scoring stage does, macro-F1 overall and per
language."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from patchpulse.evals.classification_suite import export_gold_test, score_classification
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.snapshot import Snapshot


class FakeScorer:
    """Reads the aspects off the text ("<language>:<aspects>"); `blind` predicts nothing for
    texts in that language."""

    def __init__(self, blind: str | None = None) -> None:
        self.blind = blind

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]:
        rows = []
        for text in texts:
            language, aspects = text.split(":", 1)
            seen = set() if language == self.blind else set(aspects.split("|"))
            rows.append([float(code in seen) for code in ASPECTS])
        return np.asarray(rows, dtype=np.float64)

    def labels(self, probs: NDArray[np.float64]) -> NDArray[np.bool_]:
        return probs >= 0.5


def gold() -> pd.DataFrame:
    rows: list[tuple[int, str, str, str]] = []
    for language in ("english", "french"):
        for n, code in enumerate(ASPECTS):
            rows.append((len(rows) + 1, f"{language}:{code}", language, code))
            if n == 0:
                rows.append((len(rows) + 1, f"{language}:none", language, "none"))
    return pd.DataFrame(rows, columns=["review_id", "text", "language", "aspects"])


def test_classification_scores_a_fake_scorer() -> None:
    perfect = score_classification(FakeScorer(), gold())
    blind = score_classification(FakeScorer(blind="french"), gold())

    assert (perfect.macro_f1, perfect.english, perfect.french) == (1.0, 1.0, 1.0)
    assert perfect.n == 22
    assert blind.english == 1.0
    assert blind.french == 0.0
    # Every aspect: 1 found in English, 1 missed in French: F1 = 2 / 3.
    assert blind.macro_f1 == pytest.approx(2 / 3)


def test_the_gold_test_export_keeps_the_test_split(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        {"review_id": [1, 2, 3], "language": ["english", "french", "english"],
         "split": ["test", "test", "dev"]}
    ).to_csv(sample, index=False)  # fmt: skip
    pd.DataFrame(
        {"review_id": [1, 2, 3], "aspects": ["content|performance", "none", "story_world"]}
    ).to_csv(labels, index=False)
    reviews = pd.DataFrame({"review_id": [1, 2, 3], "text": ["[b]Raw[/b] one", "deux", "three"]})
    frame = pd.DataFrame({"appid": [1]})

    test = export_gold_test(sample, labels, Snapshot(reviews=reviews, games=frame, patches=frame))

    assert list(test["review_id"]) == [1, 2]
    assert list(test["text"]) == ["[b]Raw[/b] one", "deux"]  # raw, as the scorer reads it
    assert list(test["language"]) == ["english", "french"]
    assert list(test["aspects"]) == ["performance|content", "none"]
