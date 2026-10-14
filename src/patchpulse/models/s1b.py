"""S1b: e5-small fine-tuned with a 10-aspect head (plan Task 13, design D5).

Training runs in PyTorch on the GTX 1650 (`s1b_train`, the local `gpu` group). The export puts the
mean pooling and the head inside the ONNX graph, so scoring is: tokenize, run the int8 graph,
sigmoid. This module holds the scorer and the training helpers that need no PyTorch; the jobs image
loads it with the `scoring` group only.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.artifact import ModelArtifact, write_artifact
from patchpulse.models.encoder import (
    E5_MODEL_ID,
    E5_REVISION,
    MAX_TOKENS,
    TOKENIZER_FILE,
    OnnxClassifier,
)
from patchpulse.models.text import clean_texts

S1B_FILE = "s1b.onnx"
CHECKPOINT_FILE = "s1b.pt"  # the fine-tuned weights (a PyTorch state dict)
RECORD_FILE = "s1b.json"  # what they were trained from, and how the training went


@dataclass(frozen=True)
class S1bParams:
    epochs: int = 3
    batch: int = 16
    max_len: int = MAX_TOKENS
    lr: float = 5e-5  # the pretrained encoder
    head_lr: float = 1e-3  # the new head starts from random weights
    weight_decay: float = 0.01
    warmup: float = 0.06  # share of the steps spent warming the learning rate up
    patience: int = 1  # epochs without a better holdout loss before stopping
    seed: int = 20261012
    # The 250k-token embedding table stays as pretrained: languages the silver set barely covers
    # keep their vectors, and the optimizer state fits the GTX 1650's 4 GB.
    freeze_word_embeddings: bool = True


@dataclass(frozen=True)
class S1bTraining:
    checkpoint: Path  # the directory holding the best epoch's weights
    epochs_run: int
    best_epoch: int  # 1-based
    holdout_losses: tuple[float, ...]  # one per epoch run
    seconds: float
    params: S1bParams = field(default_factory=S1bParams)


def write_training(training: S1bTraining) -> None:
    """`s1b.json` next to the weights, so a finished fine-tune can be exported again later."""
    record = {
        "base": E5_MODEL_ID,
        "revision": E5_REVISION,
        "aspects": list(ASPECTS),
        "params": asdict(training.params),
        "training": {
            "epochs_run": training.epochs_run,
            "best_epoch": training.best_epoch,
            "holdout_losses": list(training.holdout_losses),
            "seconds": training.seconds,
        },
    }
    (training.checkpoint / RECORD_FILE).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def read_training(checkpoint: Path) -> S1bTraining:
    path = checkpoint / RECORD_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no {RECORD_FILE} in {checkpoint}: not a finished S1b fine-tune")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("revision") != E5_REVISION or tuple(record.get("aspects", ())) != ASPECTS:
        raise ValueError(f"{path} was trained from another e5 revision or aspect list")
    done = record["training"]
    return S1bTraining(
        checkpoint=checkpoint,
        epochs_run=int(done["epochs_run"]),
        best_epoch=int(done["best_epoch"]),
        holdout_losses=tuple(float(loss) for loss in done["holdout_losses"]),
        seconds=float(done["seconds"]),
        params=S1bParams(**record["params"]),
    )


def pos_weights(labels: ArrayLike) -> NDArray[np.float64]:
    """Negatives over positives per aspect: BCE's weight on positives, so rare aspects count as
    much as common ones (as `class_weight="balanced"` does for S0 and S1a)."""
    y = np.asarray(labels, dtype=bool)
    positives = y.sum(axis=0)
    missing = [code for code, count in zip(ASPECTS, positives, strict=True) if count == 0]
    if missing:
        raise ValueError(f"aspects without a positive example: {', '.join(missing)}")
    return np.asarray((y.shape[0] - positives) / positives, dtype=np.float64)


def length_grouped_batches(
    lengths: ArrayLike, *, batch: int, seed: int, epoch: int, group: int = 50
) -> list[list[int]]:
    """Shuffled batches of reviews of similar length, so little time goes on padding.

    Each epoch shuffles the reviews, sorts every run of `batch * group` of them by length, cuts
    batches, and shuffles the batches.
    """
    sizes = np.asarray(lengths)
    rng = np.random.default_rng([seed, epoch])
    order = rng.permutation(sizes.size)
    batches: list[list[int]] = []
    for start in range(0, order.size, batch * group):
        chunk = order[start : start + batch * group]
        chunk = chunk[np.argsort(sizes[chunk], kind="stable")]
        batches += [chunk[i : i + batch].tolist() for i in range(0, chunk.size, batch)]
    return [batches[i] for i in rng.permutation(len(batches))]


def early_stop(losses: Sequence[float], *, patience: int) -> tuple[int, bool]:
    """The best epoch so far (1-based, lowest holdout loss) and whether to stop: `patience`
    epochs have passed without beating it."""
    best = int(np.argmin(losses)) + 1
    return best, len(losses) - best >= patience


def sigmoid(logits: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64))))


def save_s1b(
    onnx_dir: Path, model_dir: Path, *, thresholds: ArrayLike, version: str
) -> ModelArtifact:
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in (S1B_FILE, TOKENIZER_FILE):
        shutil.copyfile(onnx_dir / name, model_dir / name)
    return write_artifact(
        model_dir,
        family="s1b",
        version=version,
        thresholds=thresholds,
        files=[S1B_FILE, TOKENIZER_FILE],
    )


class S1bScorer:
    """The exported graph with its tuned thresholds: a `Scorer`."""

    def __init__(self, classifier: OnnxClassifier, artifact: ModelArtifact) -> None:
        self.classifier = classifier
        self.artifact = artifact
        self._cuts = artifact.cuts()

    @classmethod
    def load(cls, model_dir: Path, artifact: ModelArtifact) -> S1bScorer:
        """Only after `read_artifact` has checked every checksum (see `load_scorer`)."""
        return cls(OnnxClassifier(model_dir, S1B_FILE), artifact)

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]:
        return sigmoid(self.classifier.logits(clean_texts(texts)))

    def labels(self, probs: NDArray[np.float64]) -> NDArray[np.bool_]:
        return np.asarray(probs, dtype=np.float64) >= self._cuts
