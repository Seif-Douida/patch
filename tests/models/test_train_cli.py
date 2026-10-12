"""`patchpulse-ml train s0` (plan Task 11): silver train in, thresholds from gold dev, a loadable
model under build/ and an MLflow run in `students` that holds no review text."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from tiny_encoder import write_tiny_encoder
from toy_reviews import toy_reviews

from patchpulse.labeling.gold import format_aspects
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
from patchpulse.models.text import clean_texts

N_SILVER, N_DEV = 300, 60


def aspects_of(row: list[bool]) -> str:
    return format_aspects(frozenset(code for code, on in zip(ASPECTS, row, strict=True) if on))


def write_inputs(root: Path) -> list[str]:
    """A snapshot, silver labels and gold dev, all invented; returns every review text."""
    texts, labels = toy_reviews(N_SILVER + N_DEV, seed=9)
    data, gold = root / "data" / "local", root / "data" / "gold"
    (data / "silver").mkdir(parents=True)
    gold.mkdir(parents=True)
    ids = list(range(1, N_SILVER + N_DEV + 1))
    pd.DataFrame(
        {
            "review_id": ids,
            "appid": 10,
            "language": "english",
            "voted_up": True,
            "text": texts,
        }
    ).to_parquet(data / "reviews.parquet", index=False)
    pd.DataFrame({"appid": [10], "name": ["Toy Game"]}).to_parquet(data / "games.parquet")
    pd.DataFrame({"appid": [], "title": []}).to_parquet(data / "patches.parquet")
    silver = ids[:N_SILVER]
    pd.DataFrame(
        {
            "review_id": silver,
            "split": ["holdout" if i % 10 == 0 else "train" for i in silver],
            "aspects": [aspects_of(list(labels[i - 1])) for i in silver],
            "model": "openai_compat/gemma-31b",
            "prompt_hash": "abc123",
        }
    ).to_parquet(data / "silver" / "labels.parquet", index=False)
    dev = ids[N_SILVER:]
    pd.DataFrame({"review_id": [*dev, 9_999], "split": ["dev"] * N_DEV + ["test"]}).to_csv(
        gold / "sample.csv", index=False
    )
    pd.DataFrame(
        {
            "review_id": dev,
            "aspects": [aspects_of(list(labels[i - 1])) for i in dev],
            "annotator": "claude",
            "guidelines_version": "v4",
        }
    ).to_csv(gold / "labels_claude.csv", index=False)
    return texts


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
