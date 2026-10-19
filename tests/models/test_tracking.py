"""MLflow tracking (plan Task 9, design D7): runs, id hashes and the artifacts allowed to leave."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import mlflow
import pytest

from patchpulse.models.tracking import (
    TrackingError,
    log_ids_hash,
    log_model_dir,
    log_report,
    tracking_run,
)


@pytest.fixture
def local_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)  # artifacts and relative report paths stay in tmp
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    return tmp_path


def write(path: Path, text: str = "# report\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_tracking_logs_params_and_metrics_to_a_local_store(local_store: Path) -> None:
    report = write(Path("reports/aspects/teacher-bakeoff.md"))

    with tracking_run(
        "teacher-bakeoff", params={"model": "gemma", "n": 3}, tags={"phase": "2"}
    ) as run:
        mlflow.log_metric("macro_f1", 0.61)
        log_ids_hash("dev", [3, 1, 2, 2])
        log_report(report)

    logged = mlflow.get_run(run.info.run_id)
    assert logged.data.params["model"] == "gemma"
    assert logged.data.params["n"] == "3"
    assert logged.data.params["dev_n"] == "3"
    assert logged.data.params["dev_ids_sha256"] == hashlib.sha256(b"1,2,3").hexdigest()
    assert logged.data.tags["phase"] == "2"
    assert logged.data.metrics["macro_f1"] == 0.61
    artifacts = [a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id)]
    assert artifacts == ["teacher-bakeoff.md"]


@pytest.mark.parametrize(
    "path",
    [
        "data/local/reviews.parquet",
        "data/local/gold_batches/batch-1.jsonl",
        "data/gold/labels_claude.csv",
        "notes.md",
        "reports/../data/local/gold_reasons.csv",
        "build/models/s0-1/s0.joblib",
        "build/models/s0-1/model.json",
        "reports/../build/models/s0-1/s0.joblib",
    ],
)
def test_tracking_refuses_local_data_and_model_files(local_store: Path, path: str) -> None:
    target = write(Path(path))

    with tracking_run("t", params={}, tags={}), pytest.raises(TrackingError, match="refused"):
        log_report(target)


def test_tracking_needs_a_tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here
    for name in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(TrackingError, match="MLFLOW_TRACKING_URI"), tracking_run("t", params={}):
        pass


def test_runs_never_print_mlflows_run_link(
    local_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For a remote store MLflow prints "View run" with an emoji before it closes the run. A
    Windows console redirected to a file (cp1252) can't encode it: `train s0` crashed on 12 Oct and
    left its DagsHub run open. Our commands print the run id themselves."""
    monkeypatch.setenv("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "false")

    with tracking_run("t", params={}):
        assert os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] == "true"


def test_a_model_folder_under_build_models_is_logged_whole(local_store: Path) -> None:
    """For the registry only (ADR-021 as amended 13 Oct): S0 hashes its n-grams, so no model file
    holds review words."""
    write(Path("build/models/s0-1/model.json"), "{}")
    write(Path("build/models/s0-1/s0.joblib"), "weights")

    with tracking_run("t", params={}, tags={}) as run:
        log_model_dir(Path("build/models/s0-1"))

    logged = mlflow.MlflowClient().list_artifacts(run.info.run_id, "model")
    assert {a.path for a in logged} == {"model/model.json", "model/s0.joblib"}


def test_an_encoder_folder_under_build_encoders_is_logged_whole(local_store: Path) -> None:
    """The review encoder (Phase 3, plan Task 13): a public model, quantized here."""
    write(Path("build/encoders/e5-small-int8/embedder.json"), "{}")
    write(Path("build/encoders/e5-small-int8/model.onnx"), "graph")

    with tracking_run("t", params={}, tags={}) as run:
        log_model_dir(Path("build/encoders/e5-small-int8"))

    logged = mlflow.MlflowClient().list_artifacts(run.info.run_id, "model")
    assert {a.path for a in logged} == {"model/embedder.json", "model/model.onnx"}


@pytest.mark.parametrize("folder", ["data/local/model", "reports/model", "build/other/model"])
def test_other_folders_are_not_models_to_log(local_store: Path, folder: str) -> None:
    write(Path(folder) / "model.json", "{}")

    with tracking_run("t", params={}, tags={}), pytest.raises(TrackingError, match="refused"):
        log_model_dir(Path(folder))
