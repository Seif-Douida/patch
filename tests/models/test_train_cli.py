"""`patchpulse-ml train s0` (plan Task 11): silver train in, thresholds from gold dev, a loadable
model under build/ and an MLflow run in `students` that holds no review text."""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pytest
from student_inputs import N_DEV, N_SILVER, write_inputs
from tiny_encoder import write_tiny_classifier, write_tiny_encoder

from patchpulse.labeling.taxonomy import ASPECTS, TAXONOMY_VERSION
from patchpulse.models import cli
from patchpulse.models.artifact import MODEL_FILE, load_scorer
from patchpulse.models.cli import main
from patchpulse.models.encoder import (
    E5_MODEL_ID,
    E5_REVISION,
    ENCODER_FILE,
    TOKENIZER_FILE,
    OnnxEncoder,
)
from patchpulse.models.s0 import S0_FILE
from patchpulse.models.s1a import HEAD_FILE
from patchpulse.models.s1b import S1B_FILE, S1bParams, S1bTraining, write_training
from patchpulse.models.text import clean_texts
from patchpulse.models.training import TrainingData


@pytest.fixture
def workdir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for name in list(os.environ):
        if name.startswith(("PP_", "MLFLOW_")):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_train_s0_saves_a_loadable_model_and_logs_its_cv_dev_macro_f1(
    workdir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    texts = write_inputs(workdir)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(workdir / 'mlflow.db').as_posix()}")

    assert main(["train", "s0"]) == 0

    (model_dir,) = (workdir / "build" / "models").glob("s0-*")
    scorer = load_scorer(model_dir)
    assert scorer.predict_proba(["fps drops"]).shape == (1, len(ASPECTS))
    (run,) = mlflow.search_runs(experiment_names=["students"], output_format="list")
    params, metrics = run.data.params, run.data.metrics
    assert params["family"] == "s0"
    assert params["taxonomy_version"] == TAXONOMY_VERSION
    assert params["teacher"] == "openai_compat/gemma-31b"
    assert params["teacher_prompt_hash"] == "abc123"
    assert params["silver_train_n"] == str(N_SILVER - N_SILVER // 10)
    assert params["gold_dev_n"] == str(N_DEV)
    assert 0.5 < metrics["dev_cv_macro_f1"] <= 1.0
    assert {f"dev_cv_f1.{code}" for code in ASPECTS} <= set(metrics)
    assert mlflow.MlflowClient().list_artifacts(run.info.run_id) == []  # model files stay local
    record = json.loads((model_dir / MODEL_FILE).read_text(encoding="utf-8"))
    assert params[f"file_sha256.{S0_FILE}"] == record["files"][S0_FILE]
    assert params["model_dir"] == model_dir.relative_to(workdir).as_posix()
    values = [*params.values(), *run.data.tags.values()]
    assert not any(text in value for text in texts for value in values)
    assert "CV-tuned dev macro-F1" in capsys.readouterr().out


def test_train_s0_needs_mlflow_before_training(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_inputs(workdir)

    assert main(["train", "s0"]) == 2
    assert "MLFLOW_TRACKING_URI" in capsys.readouterr().err
    assert not (workdir / "build").exists()


def test_train_s1a_saves_a_loadable_model_and_logs_its_run(
    workdir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    texts = write_inputs(workdir)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(workdir / 'mlflow.db').as_posix()}")
    encoder_dir = workdir / "tiny-encoder"
    write_tiny_encoder(encoder_dir)
    monkeypatch.setattr(cli, "_encoder_dir", lambda: encoder_dir)
    monkeypatch.setitem(sys.modules, "torch", None)  # S1a needs no PyTorch: no gpu group

    assert main(["train", "s1a"]) == 0

    (model_dir,) = (workdir / "build" / "models").glob("s1a-*")
    assert load_scorer(model_dir).predict_proba(["good game"]).shape == (1, len(ASPECTS))
    (run,) = mlflow.search_runs(experiment_names=["students"], output_format="list")
    params, metrics = run.data.params, run.data.metrics
    assert params["family"] == "s1a"
    assert params["encoder"] == E5_MODEL_ID
    assert params["encoder_revision"] == E5_REVISION
    assert {f"file_sha256.{name}" for name in (ENCODER_FILE, TOKENIZER_FILE, HEAD_FILE)} <= set(
        params
    )
    assert 0.0 <= metrics["dev_cv_macro_f1"] <= 1.0
    assert metrics["cpu_preview_reviews_per_s"] > 0
    assert mlflow.MlflowClient().list_artifacts(run.info.run_id) == []
    # The head learned from the very embeddings the shipped encoder computes (int8 in real use).
    (cached,) = (workdir / "data" / "local" / "embeddings").glob("*.npy")
    silver_train = [texts[i - 1] for i in range(1, N_SILVER + 1) if i % 10 != 0]
    np.testing.assert_allclose(
        np.load(cached), OnnxEncoder(model_dir).embed(clean_texts(silver_train)), atol=1e-6
    )
    assert "CPU preview" in capsys.readouterr().out


def test_train_s1b_needs_the_gpu_group(
    workdir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_inputs(workdir)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(workdir / 'mlflow.db').as_posix()}")
    monkeypatch.setitem(sys.modules, "torch", None)  # as if the gpu group weren't installed

    assert main(["train", "s1b"]) == 2
    assert "--group gpu" in capsys.readouterr().err
    assert not (workdir / "build").exists()


def test_train_s1b_saves_a_loadable_model_and_logs_its_run(
    workdir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fine-tuning and export are faked (they need the GPU); everything after them is real."""
    write_inputs(workdir)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(workdir / 'mlflow.db').as_posix()}")
    seen: dict[str, int] = {}

    def fine_tune(data: TrainingData, checkpoint_dir: Path, params: S1bParams) -> S1bTraining:
        seen.update(train=len(data.train.ids), holdout=len(data.holdout.ids))
        checkpoint_dir.mkdir(parents=True)
        return S1bTraining(
            checkpoint_dir, epochs_run=2, best_epoch=1, holdout_losses=(0.4, 0.5), seconds=12.0
        )

    def export(checkpoint: Path, out_dir: Path) -> Path:
        write_tiny_classifier(out_dir, file_name=S1B_FILE)
        return out_dir

    monkeypatch.setattr(cli, "_require_torch", lambda: None)
    monkeypatch.setattr(cli, "_fine_tune", fine_tune)
    monkeypatch.setattr(cli, "_export", export)
    monkeypatch.setattr(cli, "_torch_dev_probs", lambda _, texts: np.full((len(texts), 10), 0.5))

    assert main(["train", "s1b"]) == 0

    assert seen == {"train": N_SILVER - N_SILVER // 10, "holdout": N_SILVER // 10}
    (model_dir,) = (workdir / "build" / "models").glob("s1b-*")
    assert load_scorer(model_dir).predict_proba(["good game"]).shape == (1, len(ASPECTS))
    (run,) = mlflow.search_runs(experiment_names=["students"], output_format="list")
    params, metrics = run.data.params, run.data.metrics
    assert params["family"] == "s1b"
    assert params["encoder"] == E5_MODEL_ID
    assert params["epochs"] == "3"
    assert {f"file_sha256.{name}" for name in (S1B_FILE, TOKENIZER_FILE)} <= set(params)
    assert (metrics["epochs_run"], metrics["best_epoch"], metrics["train_seconds"]) == (2, 1, 12)
    assert (metrics["holdout_loss.1"], metrics["holdout_loss.2"]) == (0.4, 0.5)
    assert {"dev_cv_macro_f1", "dev_cv_macro_f1_torch", "cpu_preview_reviews_per_s"} <= set(metrics)
    assert mlflow.MlflowClient().list_artifacts(run.info.run_id) == []
    assert "CPU preview" in capsys.readouterr().out


def test_output_the_console_cannot_encode_is_replaced_not_fatal(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On 12 Oct PyTorch's exporter printed an emoji during `train s1b`: Windows' cp1252 console,
    redirected to a file, couldn't encode it and the run died after 35 minutes of training."""
    raw = io.BytesIO()
    console = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", console)

    def chatty(settings: object) -> int:
        print("export done ✅")
        return 0

    monkeypatch.setattr(cli, "_train_s0", chatty)

    assert main(["train", "s0"]) == 0
    console.flush()
    assert raw.getvalue().decode("cp1252").splitlines() == ["export done ?"]


def test_train_s1b_can_resume_from_a_finished_fine_tune(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_inputs(workdir)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(workdir / 'mlflow.db').as_posix()}")
    checkpoint = workdir / "build" / "checkpoints" / "s1b-earlier"
    checkpoint.mkdir(parents=True)
    write_training(S1bTraining(checkpoint, 2, 1, (0.43, 0.46), 2047.0, S1bParams(lr=3e-5)))

    def no_fine_tune(*_: object) -> S1bTraining:
        raise AssertionError("the saved fine-tune must be reused, not trained again")

    def export(saved: Path, out_dir: Path) -> Path:
        assert saved == checkpoint
        write_tiny_classifier(out_dir, file_name=S1B_FILE)
        return out_dir

    monkeypatch.setattr(cli, "_require_torch", lambda: None)
    monkeypatch.setattr(cli, "_fine_tune", no_fine_tune)
    monkeypatch.setattr(cli, "_export", export)
    monkeypatch.setattr(cli, "_torch_dev_probs", lambda _, texts: np.full((len(texts), 10), 0.5))

    assert main(["train", "s1b", "--checkpoint", str(checkpoint)]) == 0

    (run,) = mlflow.search_runs(experiment_names=["students"], output_format="list")
    assert run.data.params["lr"] == "3e-05"  # the saved fine-tune's settings, not today's
    metrics = run.data.metrics
    assert (metrics["epochs_run"], metrics["train_seconds"], metrics["holdout_loss.2"]) == (
        2,
        2047,
        0.46,
    )
