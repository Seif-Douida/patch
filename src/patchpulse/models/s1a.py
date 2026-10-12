"""S1a: frozen e5 embeddings and one logistic regression per aspect (plan Task 12, design D5).

Training standardises the embeddings and fits a balanced logistic regression per aspect with
scikit-learn; the standardisation is then folded into the weights, so the saved head is just
`coef` (aspects x 384) and `intercept` (aspects), as plain arrays in an .npz: no pickle to trust.
Scoring is `sigmoid(embedding @ coef.T + intercept)`.

The model directory holds the int8 encoder, its tokenizer and the head. Texts are cleaned with
`clean_texts` before the encoder sees them, in training and in Azure alike.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression

from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.artifact import ModelArtifact, write_artifact
from patchpulse.models.encoder import ENCODER_FILE, TOKENIZER_FILE, OnnxEncoder
from patchpulse.models.text import clean_texts

HEAD_FILE = "head.npz"


@dataclass(frozen=True)
class S1aParams:
    c: float = 1.0  # on standardised embeddings
    max_iter: int = 2_000


@dataclass(frozen=True)
class S1aModel:
    coef: NDArray[np.float64]  # (aspects, dim)
    intercept: NDArray[np.float64]  # (aspects,)

    def predict_proba(self, embeddings: ArrayLike) -> NDArray[np.float64]:
        logits = np.asarray(embeddings, dtype=np.float64) @ self.coef.T + self.intercept
        return np.asarray(1.0 / (1.0 + np.exp(-logits)), dtype=np.float64)


def train_s1a(
    embeddings: ArrayLike, labels: ArrayLike, *, params: S1aParams | None = None
) -> S1aModel:
    params = params or S1aParams()
    x = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    if y.ndim != 2 or y.shape[1] != len(ASPECTS) or y.shape[0] != x.shape[0]:
        raise ValueError(f"need one label row per embedding and {len(ASPECTS)} aspect columns")
    one_class = [
        code for code, column in zip(ASPECTS, y.T, strict=True) if column.all() or not column.any()
    ]
    if one_class:
        raise ValueError(f"aspects with only one class in training: {', '.join(one_class)}")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    coef = np.zeros((len(ASPECTS), x.shape[1]))
    intercept = np.zeros(len(ASPECTS))
    for index in range(len(ASPECTS)):
        head = LogisticRegression(
            C=params.c, class_weight="balanced", max_iter=params.max_iter
        ).fit(z, y[:, index])
        coef[index] = head.coef_[0] / scale
        intercept[index] = head.intercept_[0] - coef[index] @ mean
    return S1aModel(coef=coef, intercept=intercept)


def save_head(model: S1aModel, path: Path) -> None:
    with path.open("wb") as file:
        np.savez(file, coef=model.coef, intercept=model.intercept)


def load_head(path: Path) -> S1aModel:
    with np.load(path, allow_pickle=False) as arrays:
        return S1aModel(coef=arrays["coef"], intercept=arrays["intercept"])


def save_s1a(
    model: S1aModel,
    encoder_dir: Path,
    model_dir: Path,
    *,
    thresholds: ArrayLike,
    version: str,
) -> ModelArtifact:
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in (ENCODER_FILE, TOKENIZER_FILE):
        shutil.copyfile(encoder_dir / name, model_dir / name)
    save_head(model, model_dir / HEAD_FILE)
    return write_artifact(
        model_dir,
        family="s1a",
        version=version,
        thresholds=thresholds,
        files=[ENCODER_FILE, TOKENIZER_FILE, HEAD_FILE],
    )


class S1aScorer:
    """The encoder and the head with their tuned thresholds: a `Scorer`."""

    def __init__(self, encoder: OnnxEncoder, model: S1aModel, artifact: ModelArtifact) -> None:
        self.encoder = encoder
        self.model = model
        self.artifact = artifact
        self._cuts = artifact.cuts()

    @classmethod
    def load(cls, model_dir: Path, artifact: ModelArtifact) -> S1aScorer:
        """Only after `read_artifact` has checked every checksum (see `load_scorer`)."""
        return cls(OnnxEncoder(model_dir), load_head(model_dir / HEAD_FILE), artifact)

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]:
        return self.model.predict_proba(self.encoder.embed(clean_texts(texts)))

    def labels(self, probs: NDArray[np.float64]) -> NDArray[np.bool_]:
        return np.asarray(probs, dtype=np.float64) >= self._cuts
