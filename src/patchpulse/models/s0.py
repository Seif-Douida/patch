"""S0, the CPU baseline student (plan Task 11, design D5).

TF-IDF on word 1-2-grams and on character 3-5-grams within word boundaries (`char_wb`, which
carries languages without spaces, like Chinese), then one logistic regression per aspect with
balanced class weights. Texts are cleaned with the shared `clean_text` before both training and
scoring, so the two always see the same input.

`min_df` keeps only n-grams found in at least that many different reviews: the saved vocabulary
holds shared phrases and fragments, never a review's text.

scikit-learn, joblib and NumPy only (the `scoring` group), so the jobs image can score with it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.artifact import ModelArtifact, write_artifact
from patchpulse.models.text import clean_text

S0_FILE = "s0.joblib"


@dataclass(frozen=True)
class S0Params:
    word_ngrams: tuple[int, int] = (1, 2)
    char_ngrams: tuple[int, int] = (3, 5)
    min_df: int = 3
    max_word_features: int = 200_000
    max_char_features: int = 300_000
    c: float = 4.0  # inverse regularisation strength of each logistic regression
    max_iter: int = 1_000


def _clean(texts: Sequence[str]) -> list[str]:
    return [clean_text(text) for text in texts]


@dataclass(frozen=True)
class S0Model:
    pipeline: Any  # scikit-learn has no type information
    params: S0Params = field(default_factory=S0Params)

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]:
        if not texts:
            return np.zeros((0, len(ASPECTS)))
        return np.asarray(self.pipeline.predict_proba(_clean(texts)), dtype=np.float64)


def _pipeline(params: S0Params) -> Pipeline:
    shared: dict[str, Any] = {
        "min_df": params.min_df,
        "sublinear_tf": True,
        "dtype": np.float32,
    }
    features = FeatureUnion(
        [
            (
                "words",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=params.word_ngrams,
                    max_features=params.max_word_features,
                    **shared,
                ),
            ),
            (
                "chars",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=params.char_ngrams,
                    max_features=params.max_char_features,
                    **shared,
                ),
            ),
        ]
    )
    head = OneVsRestClassifier(
        LogisticRegression(
            C=params.c, class_weight="balanced", solver="liblinear", max_iter=params.max_iter
        )
    )
    return Pipeline([("features", features), ("head", head)])


def train_s0(texts: Sequence[str], labels: ArrayLike, *, params: S0Params | None = None) -> S0Model:
    """Fit on raw review texts and a (reviews, aspects) boolean label matrix."""
    params = params or S0Params()
    y = np.asarray(labels, dtype=bool)
    if y.ndim != 2 or y.shape[1] != len(ASPECTS):
        raise ValueError(f"labels need one column for each of the {len(ASPECTS)} aspects")
    if y.shape[0] != len(texts):
        raise ValueError("one label row per text")
    pipeline = _pipeline(params).fit(_clean(texts), y.astype(np.int8))
    return S0Model(pipeline=pipeline, params=params)


def save_s0(
    model: S0Model, model_dir: Path, *, thresholds: ArrayLike, version: str
) -> ModelArtifact:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.pipeline, model_dir / S0_FILE, compress=3)
    return write_artifact(
        model_dir, family="s0", version=version, thresholds=thresholds, files=[S0_FILE]
    )


class S0Scorer:
    """S0 with its tuned thresholds: the `Scorer` the scoring stage uses."""

    def __init__(self, model: S0Model, artifact: ModelArtifact) -> None:
        self.model = model
        self.artifact = artifact
        self._cuts = artifact.cuts()

    @classmethod
    def load(cls, model_dir: Path, artifact: ModelArtifact) -> S0Scorer:
        """Only after `read_artifact` has checked every checksum (see `load_scorer`)."""
        return cls(S0Model(pipeline=joblib.load(model_dir / S0_FILE)), artifact)

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]:
        return self.model.predict_proba(texts)

    def labels(self, probs: NDArray[np.float64]) -> NDArray[np.bool_]:
        return np.asarray(probs, dtype=np.float64) >= self._cuts
