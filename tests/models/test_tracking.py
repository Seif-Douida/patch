"""MLflow tracking (plan Task 9, design D7): runs, id hashes and the artifacts allowed to leave."""

from __future__ import annotations

import hashlib
from pathlib import Path

import mlflow
import pytest

from patchpulse.models.tracking import TrackingError, log_ids_hash, log_report, tracking_run


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
    ],
)
def test_tracking_refuses_local_data_artifacts(local_store: Path, path: str) -> None:
    target = write(Path(path))

    with tracking_run("t", params={}, tags={}), pytest.raises(TrackingError, match="refused"):
        log_report(target)


def test_model_build_folder_is_allowed(local_store: Path) -> None:
    target = write(Path("build/s0/model.json"), "{}")

    with tracking_run("t", params={}, tags={}) as run:
        log_report(target)

    assert [a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id)] == ["model.json"]


def test_tracking_needs_a_tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here
    for name in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(TrackingError, match="MLFLOW_TRACKING_URI"), tracking_run("t", params={}):
        pass
