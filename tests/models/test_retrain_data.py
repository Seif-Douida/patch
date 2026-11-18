"""Phase 7 Task 9: the silver labels in git and the retrain's data from SQL (design D5)."""

from __future__ import annotations

import csv
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from monitoring_fixtures import SENTINEL_TEXT, add_rows, review_row
from sqlalchemy import Engine

from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.cli import main
from patchpulse.models.retrain_data import (
    RECENT_WEEKS,
    SILVER_COLUMNS,
    load_retrain_data,
    write_silver_csv,
)
from patchpulse.monitoring.store import StoredTeacherLabel, write_teacher_labels
from patchpulse.pipeline.spotcheck import is_holdout

TODAY = date(2026, 12, 2)
CREATED = datetime(2026, 11, 20, 10, 0)


def silver_csv(path: Path, rows: list[tuple[int, str, str]]) -> Path:
    write_silver_csv(
        [
            {
                "review_id": rid,
                "aspects": aspects,
                "split": split,
                "model": "gemma-31b",
                "prompt_hash": "h1",
            }
            for rid, aspects, split in rows
        ],
        path,
    )
    return path


def gold_dir(path: Path, dev: list[int], test: list[int]) -> Path:
    path.mkdir()
    with (path / "sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["review_id", "appid", "language", "voted_up", "length_bucket", "near_patch", "split"]
        )
        writer.writerows([[i, 553850, "english", True, "short", False, "dev"] for i in dev])
        writer.writerows([[i, 553850, "french", True, "short", False, "test"] for i in test])
    with (path / "labels_claude.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["review_id", "aspects", "annotator", "guidelines_version"])
        writer.writerows([[i, "performance", "claude", "v4"] for i in [*dev, *test]])
    return path


def test_the_export_holds_ids_labels_and_splits_only(tmp_path: Path) -> None:
    path = silver_csv(
        tmp_path / "labels.csv", [(5, "performance|content", "train"), (6, "none", "holdout")]
    )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0]) == list(SILVER_COLUMNS)
    assert rows[0] == {
        "review_id": "5",
        "aspects": "performance|content",
        "split": "train",
        "model": "gemma-31b",
        "prompt_hash": "h1",
    }
    assert "\r" not in path.read_text(encoding="utf-8")


def test_the_export_command_writes_the_committed_file_from_the_local_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in list(os.environ):
        if name.startswith(("PP_", "MLFLOW_")):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "data" / "local" / "silver"
    local.mkdir(parents=True)
    pd.DataFrame(
        {
            "review_id": [30, 10, 20],
            "split": ["train", "holdout", "train"],
            "aspects": ["content", "none", "performance|monetization"],
            "model": ["gemma-31b"] * 3,
            "prompt_hash": ["h1"] * 3,
        }
    ).to_parquet(local / "labels.parquet", index=False)

    assert main(["silver", "export-labels"]) == 0

    with (tmp_path / "data" / "silver" / "labels.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["review_id"] for r in rows] == ["10", "20", "30"]
    assert rows[1]["aspects"] == "performance|monetization"
    out = capsys.readouterr().out
    assert "3 labels" in out
    assert "2 train" in out
    assert "1 holdout" in out


def spot(rid: int, created_at: datetime, aspects: frozenset[str]) -> StoredTeacherLabel:
    return StoredTeacherLabel(rid, CREATED, "gemma-4-31b-it", "h2", "spotcheck", aspects)


def seed(engine: Engine, ids: list[int]) -> None:
    add_rows(engine, [review_row(i, CREATED, text=f"{SENTINEL_TEXT} {i}") for i in ids])


@pytest.mark.integration
def test_train_joins_silver_and_spotcheck_training_labels(
    migrated_engine: Engine, tmp_path: Path
) -> None:
    spot_ids = list(range(1000, 1100))
    seed(migrated_engine, [1, 2, 3, 10, 11, *spot_ids])
    with migrated_engine.begin() as connection:
        write_teacher_labels(
            connection, [spot(i, CREATED, frozenset({"content"})) for i in spot_ids], run_id=7
        )
    silver = silver_csv(
        tmp_path / "labels.csv",
        [(1, "performance", "train"), (2, "none", "train"), (3, "content", "holdout")],
    )

    with migrated_engine.connect() as connection:
        data = load_retrain_data(
            connection,
            today=TODAY,
            silver_csv=silver,
            gold_dir=gold_dir(tmp_path / "gold", [10], [11]),
        )

    training_spot = [i for i in spot_ids if not is_holdout(i)]
    assert data.train.ids == [1, 2, *training_spot]
    assert data.train.labels[0, ASPECTS.index("performance")]
    assert not data.train.labels[1].any()
    assert data.train.texts[0].startswith("SENTINEL-REVIEW")
    assert data.teachers == {"gemma-31b": 2, "gemma-4-31b-it": len(training_spot)}
    assert 3 not in data.train.ids  # the silver holdout isn't trained on


@pytest.mark.integration
def test_the_recent_set_is_the_last_eight_weeks_holdout(
    migrated_engine: Engine, tmp_path: Path
) -> None:
    holdout = [i for i in range(2000, 2400) if is_holdout(i)][:6]
    seed(migrated_engine, holdout)
    with migrated_engine.begin() as connection:
        write_teacher_labels(
            connection, [spot(i, CREATED, frozenset({"performance"})) for i in holdout], run_id=7
        )
        old = TODAY - timedelta(weeks=RECENT_WEEKS, days=1)
        connection.exec_driver_sql(
            f"UPDATE raw.teacher_label SET created_at = '{old:%Y-%m-%d} 03:00:00' "
            f"WHERE recommendation_id = {holdout[0]}"
        )
    silver = silver_csv(tmp_path / "labels.csv", [])

    with migrated_engine.connect() as connection:
        data = load_retrain_data(
            connection, today=TODAY, silver_csv=silver, gold_dir=gold_dir(tmp_path / "gold", [], [])
        )

    assert data.recent.ids == holdout[1:]
    assert data.languages["recent"] == ["english"] * 5


@pytest.mark.integration
def test_a_review_missing_from_sql_is_counted_not_fatal(
    migrated_engine: Engine, tmp_path: Path
) -> None:
    seed(migrated_engine, [1])
    silver = silver_csv(
        tmp_path / "labels.csv", [(1, "performance", "train"), (99, "content", "train")]
    )

    with migrated_engine.connect() as connection:
        data = load_retrain_data(
            connection, today=TODAY, silver_csv=silver, gold_dir=gold_dir(tmp_path / "gold", [], [])
        )

    assert data.train.ids == [1]
    assert data.missing == 1


@pytest.mark.integration
def test_gold_dev_and_test_come_from_the_committed_files(
    migrated_engine: Engine, tmp_path: Path
) -> None:
    seed(migrated_engine, [10, 11, 12])
    silver = silver_csv(tmp_path / "labels.csv", [])

    with migrated_engine.connect() as connection:
        data = load_retrain_data(
            connection,
            today=TODAY,
            silver_csv=silver,
            gold_dir=gold_dir(tmp_path / "gold", [10, 12], [11]),
        )

    assert (data.dev.ids, data.test.ids) == ([10, 12], [11])
    assert data.dev.labels[:, ASPECTS.index("performance")].all()
    assert data.languages["test"] == ["english"]  # the language is SQL's, not the sample file's
