"""S1b (plan Task 13, design D5): e5-small fine-tuned with a 10-aspect head, exported to ONNX and
quantized to int8. The training helpers and the scorer run anywhere; training and the ONNX parity
check need the `gpu` group, CUDA and the e5 download (Seif's machine)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from tiny_encoder import expected_logits, write_tiny_classifier

from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.artifact import load_scorer
from patchpulse.models.encoder import TOKENIZER_FILE, OnnxClassifier
from patchpulse.models.s1b import (
    CHECKPOINT_FILE,
    RECORD_FILE,
    S1B_FILE,
    S1bParams,
    S1bScorer,
    S1bTraining,
    early_stop,
    length_grouped_batches,
    pos_weights,
    read_training,
    save_s1b,
    write_training,
)


def sigmoid(logits: NDArray[np.float32]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))


def test_pos_weights_balance_each_aspect() -> None:
    labels = np.zeros((10, len(ASPECTS)), dtype=bool)
    labels[:2, 0] = True  # 2 positives, 8 negatives
    labels[:5, 1:] = True  # 5 and 5

    assert pos_weights(labels).tolist() == [4.0] + [1.0] * (len(ASPECTS) - 1)


def test_an_aspect_without_positives_is_refused() -> None:
    labels = np.ones((10, len(ASPECTS)), dtype=bool)
    labels[:, 3] = False

    with pytest.raises(ValueError, match=ASPECTS[3]):
        pos_weights(labels)


def test_length_grouped_batches_cover_every_review_once() -> None:
    lengths = np.random.default_rng(1).integers(1, 500, size=103)

    batches = length_grouped_batches(lengths, batch=16, seed=1, epoch=0)

    assert sorted(i for b in batches for i in b) == list(range(103))
    assert all(1 <= len(b) <= 16 for b in batches)
    spread = np.mean([lengths[b].max() - lengths[b].min() for b in batches])
    assert spread < 0.25 * (lengths.max() - lengths.min())  # similar lengths: little padding
    assert batches == length_grouped_batches(lengths, batch=16, seed=1, epoch=0)
    assert batches != length_grouped_batches(lengths, batch=16, seed=1, epoch=1)


@pytest.mark.parametrize(
    ("losses", "patience", "expected"),
    [
        ([0.9], 1, (1, False)),
        ([0.9, 0.8], 1, (2, False)),
        ([0.9, 0.8, 0.85], 1, (2, True)),
        ([0.9, 0.95], 2, (1, False)),
        ([0.9, 0.95, 0.97], 2, (1, True)),
    ],
)
def test_early_stopping_keeps_the_best_epoch(
    losses: list[float], patience: int, expected: tuple[int, bool]
) -> None:
    assert early_stop(losses, patience=patience) == expected


def test_s1b_artifact_loads_through_the_common_scorer(tmp_path: Path) -> None:
    onnx_dir, model_dir = tmp_path / "onnx", tmp_path / "model"
    table = write_tiny_classifier(onnx_dir, file_name=S1B_FILE)
    cuts = np.linspace(0.3, 0.7, len(ASPECTS))

    artifact = save_s1b(onnx_dir, model_dir, thresholds=cuts, version="t")
    scorer = load_scorer(model_dir)
    probs = scorer.predict_proba(["good game", "[b]bad[/b] fps drops"])  # cleaned: "bad fps drops"

    assert isinstance(scorer, S1bScorer)
    assert artifact.family == "s1b"
    assert set(artifact.files) == {S1B_FILE, TOKENIZER_FILE}
    expected = sigmoid(
        np.vstack([expected_logits(t, table) for t in ("good game", "bad fps drops")])
    )
    np.testing.assert_allclose(probs, expected, atol=1e-6)
    assert np.array_equal(scorer.labels(probs), probs >= cuts)
    odd = scorer.predict_proba(["", "\U0001f525" * 3, "good " * 2_000])
    assert odd.shape == (3, len(ASPECTS))
    assert np.isfinite(odd).all()


# Invented examples (no review text), one aspect each, every aspect covered.
EXAMPLES = [
    "The frame rate drops in every fight.",
    "It crashes to the desktop after loading.",
    "The new rifle is far too strong.",
    "There are only three maps after launch.",
    "The battle pass costs too much.",
    "Servers go down every evening.",
    "The story and its characters are wonderful.",
    "The menus are confusing with a gamepad.",
    "Region locked and the launcher needs an account.",
    "My dog barked at the screen.",
]


def gpu_examples(n: int) -> tuple[list[str], NDArray[np.bool_]]:
    texts = [f"{EXAMPLES[i % 10]} ({i})" for i in range(n)]
    labels = np.zeros((n, len(ASPECTS)), dtype=bool)
    labels[np.arange(n), np.arange(n) % 10] = True
    return texts, labels


@pytest.mark.gpu
def test_one_training_step_runs(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch", reason="needs the gpu group")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from patchpulse.models.s1b_train import train_s1b

    texts, labels = gpu_examples(32)
    result = train_s1b(
        texts, labels, texts[:10], labels[:10], tmp_path, params=S1bParams(epochs=1, batch=16)
    )

    assert (result.epochs_run, result.best_epoch) == (1, 1)
    assert len(result.holdout_losses) == 1
    assert math.isfinite(result.holdout_losses[0])
    assert (result.checkpoint / CHECKPOINT_FILE).is_file()
    assert read_training(result.checkpoint) == result


@pytest.mark.gpu
def test_onnx_logits_match_torch_within_tolerance(tmp_path: Path) -> None:
    """Tolerances set before the first run: ONNX fp32 logits within 1e-3 of PyTorch's, and int8
    probabilities within 0.1."""
    torch = pytest.importorskip("torch", reason="needs the gpu group")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from patchpulse.models.s1b_train import export_s1b, torch_logits, train_s1b

    texts, labels = gpu_examples(32)
    trained = train_s1b(
        texts, labels, texts[:10], labels[:10], tmp_path / "ckpt", params=S1bParams(epochs=1)
    )
    reference = torch_logits(trained.checkpoint, EXAMPLES, device="cpu")
    fp32 = OnnxClassifier(
        export_s1b(trained.checkpoint, tmp_path / "fp32", quantize=False), S1B_FILE
    )
    int8 = OnnxClassifier(export_s1b(trained.checkpoint, tmp_path / "int8"), S1B_FILE)

    assert reference.shape == (len(EXAMPLES), len(ASPECTS))
    np.testing.assert_allclose(fp32.logits(EXAMPLES), reference, atol=1e-3)
    assert np.abs(sigmoid(int8.logits(EXAMPLES)) - sigmoid(reference)).max() <= 0.1


def test_a_finished_fine_tune_can_be_read_back(tmp_path: Path) -> None:
    training = S1bTraining(
        tmp_path,
        epochs_run=2,
        best_epoch=1,
        holdout_losses=(0.4345, 0.4651),
        seconds=2047.0,
        params=S1bParams(epochs=3, lr=3e-5),
    )

    write_training(training)

    assert read_training(tmp_path) == training


def test_a_checkpoint_without_its_record_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=RECORD_FILE):
        read_training(tmp_path)
