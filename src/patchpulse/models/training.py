"""What every student trains and is judged on, and how its run is recorded (plan Tasks 11-13).

- **Silver train** (the teacher's labels) is what a student learns from.
- **Gold dev** (Claude's labels) tunes its thresholds and compares students: out of fold, so the
  comparison isn't flattered by its own tuning (design D6, rule 2).
- **Gold test** isn't read here: it's used once, for the eval report (Task 14).

MLflow gets parameters, metrics, id hashes and each model file's SHA-256: never review text (C4),
and never the model files, which stay in build/models/ on this machine (Seif, 12 Oct).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from patchpulse.labeling.gold import CLAUDE_LABELS_FILE, SAMPLE_FILE, read_labels
from patchpulse.labeling.silver import LABELS_FILE, SILVER_DIR, load_silver
from patchpulse.labeling.taxonomy import ASPECTS, TAXONOMY_VERSION
from patchpulse.models.artifact import read_artifact
from patchpulse.models.metrics import micro_f1, prf
from patchpulse.models.snapshot import load_snapshot
from patchpulse.models.text import CLEANING_VERSION
from patchpulse.models.thresholds import cv_predictions, tune_thresholds
from patchpulse.models.tracking import log_ids_hash, tracking_run

EXPERIMENT = "students"
MODELS_DIR = Path("build/models")
CV_FOLDS = 5
CV_SEED = 20261012


@dataclass(frozen=True)
class Split:
    ids: list[int]
    texts: list[str]
    labels: NDArray[np.bool_]  # (reviews, aspects)


@dataclass(frozen=True)
class TrainingData:
    train: Split
    dev: Split
    holdout: Split  # silver holdout: S1b's early stopping (fidelity is measured in Task 14)
    teacher: str  # the model that labeled silver, as cached
    teacher_prompt_hash: str


def label_matrix(labels: Mapping[int, frozenset[str]], ids: Sequence[int]) -> NDArray[np.bool_]:
    missing = [i for i in ids if i not in labels]
    if missing:
        raise ValueError(f"{len(missing)} reviews have no label, e.g. {missing[:3]}")
    return np.array([[code in labels[i] for code in ASPECTS] for i in ids], dtype=bool)


def _one(values: pd.Series, what: str) -> str:
    unique = sorted({str(v) for v in values})
    if len(unique) != 1:
        raise ValueError(f"silver labels must come from one {what}, found {unique}")
    return unique[0]


def load_training_data(data_dir: Path, gold_dir: Path) -> TrainingData:
    silver = load_silver(data_dir)
    source = pd.read_parquet(data_dir / SILVER_DIR / LABELS_FILE, columns=["model", "prompt_hash"])
    texts = load_snapshot(data_dir).reviews.set_index("review_id")["text"]
    sample = pd.read_csv(gold_dir / SAMPLE_FILE)
    dev_ids = sorted(int(i) for i in sample.loc[sample["split"] == "dev", "review_id"])
    train_ids = [int(i) for i in silver.train["review_id"]]
    holdout_ids = [int(i) for i in silver.holdout["review_id"]]
    return TrainingData(
        train=Split(
            ids=train_ids,
            texts=[str(texts[i]) for i in train_ids],
            labels=silver.train[list(ASPECTS)].to_numpy(dtype=bool),
        ),
        dev=Split(
            ids=dev_ids,
            texts=[str(texts[i]) for i in dev_ids],
            labels=label_matrix(read_labels(gold_dir / CLAUDE_LABELS_FILE), dev_ids),
        ),
        holdout=Split(
            ids=holdout_ids,
            texts=[str(texts[i]) for i in holdout_ids],
            labels=silver.holdout[list(ASPECTS)].to_numpy(dtype=bool),
        ),
        teacher=_one(source["model"], "teacher"),
        teacher_prompt_hash=_one(source["prompt_hash"], "prompt"),
    )


@dataclass(frozen=True)
class DevScores:
    thresholds: NDArray[np.float64]  # tuned on all of dev: the ones the model ships with
    cv_macro_f1: float  # out of fold: the number students are compared on
    cv_micro_f1: float
    cv_f1: dict[str, float]


def score_on_dev(dev: Split, probs: NDArray[np.float64]) -> DevScores:
    predicted = cv_predictions(dev.labels, probs, folds=CV_FOLDS, seed=CV_SEED)
    per_aspect = prf(dev.labels, predicted).f1
    return DevScores(
        thresholds=tune_thresholds(dev.labels, probs),
        cv_macro_f1=float(per_aspect.mean()),
        cv_micro_f1=micro_f1(dev.labels, predicted),
        cv_f1={code: float(f1) for code, f1 in zip(ASPECTS, per_aspect, strict=True)},
    )


def new_version(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")


def log_student(
    *,
    family: str,
    version: str,
    params: Mapping[str, object],
    data: TrainingData,
    scores: DevScores,
    model_dir: Path,
    metrics: Mapping[str, float] | None = None,
) -> str:
    """One run in `students`; returns its id. The model is identified by its files' checksums."""
    files = read_artifact(model_dir).files
    record = {
        "family": family,
        "version": version,
        "taxonomy_version": TAXONOMY_VERSION,
        "cleaning_version": CLEANING_VERSION,
        "teacher": data.teacher,
        "teacher_prompt_hash": data.teacher_prompt_hash,
        "cv_folds": CV_FOLDS,
        "cv_seed": CV_SEED,
        "model_dir": model_dir.as_posix(),
        **{f"file_sha256.{name}": digest for name, digest in files.items()},
        **params,
    }
    cuts = {f"threshold.{c}": float(t) for c, t in zip(ASPECTS, scores.thresholds, strict=True)}
    with tracking_run(
        EXPERIMENT, params=record, tags={"family": family}, run_name=f"{family}-{version}"
    ) as run:
        log_ids_hash("silver_train", data.train.ids)
        log_ids_hash("gold_dev", data.dev.ids)
        mlflow.log_metrics(
            {
                "dev_cv_macro_f1": scores.cv_macro_f1,
                "dev_cv_micro_f1": scores.cv_micro_f1,
                **{f"dev_cv_f1.{code}": f1 for code, f1 in scores.cv_f1.items()},
                **cuts,
                **(metrics or {}),
            }
        )
        return str(run.info.run_id)
