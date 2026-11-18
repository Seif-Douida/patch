"""Phase 7 Task 10: `patchpulse-pipeline retrain` (design D5), against SQL Server and a local
MLflow store with a registry."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pytest
from monitoring_fixtures import SENTINEL_TEXT, add_rows, review_row
from numpy.typing import NDArray
from sqlalchemy import Engine
from toy_reviews import toy_reviews

from patchpulse.config import Settings
from patchpulse.db.engine import make_engine
from patchpulse.db.migrate import upgrade_to_head
from patchpulse.labeling.taxonomy import ASPECTS, format_aspects
from patchpulse.models.artifact import MODEL_FILE, sha256_file
from patchpulse.models.champion import REGISTERED_MODEL
from patchpulse.models.promotion import RetrainResult, load_rule
from patchpulse.models.retrain import RETRAIN_EXPERIMENT, run_retrain
from patchpulse.models.retrain_data import write_silver_csv
from patchpulse.models.s0 import save_s0, train_s0
from patchpulse.monitoring.reference import Reference
from patchpulse.monitoring.store import StoredTeacherLabel, read_metrics, write_teacher_labels
from patchpulse.pipeline.cli import main
from patchpulse.pipeline.spotcheck import is_holdout

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
RULE = load_rule(ROOT / "config" / "retrain.yaml")
TRAINED_AT = datetime(2026, 12, 1, 3, 0, tzinfo=UTC)
VERSION = "20261201T030000Z"
CREATED = datetime(2026, 11, 2, 10, 0)
SILVER = range(1, 601)
DEV, TEST = range(2001, 2151), range(3001, 3101)
SPOT = range(5001, 6001)
EXECUTION = "pp-retrain-x7k2p"


def aspects_of(row: NDArray[np.bool_]) -> frozenset[str]:
    return frozenset(code for code, on in zip(ASPECTS, row, strict=True) if on)


@dataclass(frozen=True)
class World:
    settings: Settings
    engine: Engine
    champion_dir: Path
    train_n: int


@pytest.fixture
def world(
    new_database: Callable[[], Settings], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[World]:
    """A database with labeled reviews, the committed label files, a champion, an MLflow store."""
    settings = new_database()
    engine = make_engine(settings)
    upgrade_to_head(engine)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("CONTAINER_APP_JOB_EXECUTION_NAME", EXECUTION)

    ids = [*SILVER, *DEV, *TEST, *SPOT]
    texts, labels = toy_reviews(len(ids), seed=3)
    by_id = dict(zip(ids, zip(texts, labels, strict=True), strict=True))
    add_rows(engine, [review_row(i, CREATED, text=f"{SENTINEL_TEXT} {by_id[i][0]}") for i in ids])

    write_silver_csv(
        [
            {
                "review_id": i,
                "aspects": format_aspects(aspects_of(by_id[i][1])),
                "split": "train",
                "model": "openai_compat/gemma-31b",
                "prompt_hash": "h1",
            }
            for i in SILVER
        ]
    )
    gold = tmp_path / "data" / "gold"
    gold.mkdir(parents=True)
    (gold / "sample.csv").write_text(
        "review_id,appid,language,voted_up,length_bucket,near_patch,split\n"
        + "".join(f"{i},553850,english,True,short,False,dev\n" for i in DEV)
        + "".join(f"{i},553850,english,True,short,False,test\n" for i in TEST),
        encoding="utf-8",
    )
    (gold / "labels_claude.csv").write_text(
        "review_id,aspects,annotator,guidelines_version\n"
        + "".join(
            f"{i},{format_aspects(aspects_of(by_id[i][1]))},claude,v4\n" for i in [*DEV, *TEST]
        ),
        encoding="utf-8",
    )
    with engine.begin() as connection:
        write_teacher_labels(
            connection,
            [
                StoredTeacherLabel(
                    i, CREATED, "gemma-4-31b-it", "h2", "spotcheck", aspects_of(by_id[i][1])
                )
                for i in SPOT
            ],
            run_id=1,
        )

    champion_texts, champion_labels = toy_reviews(120, seed=21)
    champion_dir = tmp_path / "champion"
    save_s0(
        train_s0(champion_texts, champion_labels),
        champion_dir,
        thresholds=np.full(len(ASPECTS), 0.5),
        version="20261013T000000Z",
    )
    train_n = len(SILVER) + sum(not is_holdout(i) for i in SPOT)
    yield World(settings, engine, champion_dir, train_n)
    engine.dispose()


def retrain(world: World) -> RetrainResult:
    return run_retrain(
        world.engine,
        champion_dir=world.champion_dir,
        out_dir=Path("build/models"),
        today=datetime.now(UTC).date(),
        rule=RULE,
        clock=lambda: TRAINED_AT,
    )


def test_retrain_logs_registers_and_tags_the_decision(world: World) -> None:
    result = retrain(world)

    client = mlflow.MlflowClient()
    run = client.get_run(result.mlflow_run_id)
    assert client.get_experiment(run.info.experiment_id).name == RETRAIN_EXPERIMENT
    assert run.data.tags["decision"] == ("promote" if result.decision.promote else "keep")
    assert run.data.tags["execution"] == EXECUTION
    assert run.data.params["train_n"] == str(world.train_n)
    assert run.data.params["challenger_version"] == VERSION
    assert run.data.params["champion_version"] == "20261013T000000Z"
    for model in ("champion", "challenger"):
        for metric in ("recent_macro_f1", "dev_cv_macro_f1", "test_macro_f1"):
            assert f"{model}.{metric}" in run.data.metrics
    assert run.data.metrics["challenger.reviews_per_second"] >= RULE.min_reviews_per_second

    alias = client.get_model_version_by_alias(REGISTERED_MODEL, "challenger")
    assert int(alias.version) == result.registered_version
    model_dir = Path("build/models") / f"s0-{VERSION}"
    assert result.model_json_sha256 == sha256_file(model_dir / MODEL_FILE)
    assert result.version == VERSION

    logged = {item.path for item in client.list_artifacts(result.mlflow_run_id)}
    assert {f"{VERSION}.md", f"{VERSION}-reference.json", f"{VERSION}-result.json"} <= logged
    reference = Reference.model_validate_json(
        Path("reports/retrain", f"{VERSION}-reference.json").read_text(encoding="utf-8")
    )
    assert reference.model_version == VERSION
    assert reference.n == world.train_n
    assert result.table == Path("reports/retrain", f"{VERSION}.md").read_text(encoding="utf-8")
    # The Monitoring page's last decision (Task 15): in SQL, with the recent holdout's gain.
    with world.engine.connect() as connection:
        (decision,) = read_metrics(connection, family="retrain")
    word = "promote" if result.decision.promote else "keep"
    assert decision.metric == f"decision.{word}"
    assert decision.scope == "run"
    recent = result.decision.checks[0]
    assert recent.champion is not None
    assert recent.challenger is not None
    assert decision.value == pytest.approx(recent.challenger - recent.champion)


def test_retrain_params_and_artifacts_hold_no_text(world: World, tmp_path: Path) -> None:
    result = retrain(world)

    client = mlflow.MlflowClient()
    run = client.get_run(result.mlflow_run_id)
    seen = [*run.data.params.values(), *run.data.tags.values(), *run.data.metrics]
    downloaded = Path(client.download_artifacts(result.mlflow_run_id, "", str(tmp_path / "out")))
    for path in downloaded.rglob("*"):
        if path.is_file():
            seen.append(path.read_text(encoding="utf-8"))
    seen.append((Path("build/models") / f"s0-{VERSION}" / MODEL_FILE).read_text(encoding="utf-8"))

    assert len(seen) > 20
    assert not [value for value in seen if "SENTINEL-REVIEW" in value]
    assert (
        json.loads(Path("reports/retrain", f"{VERSION}-result.json").read_text())["version"]
        == VERSION
    )


def use_settings(monkeypatch: pytest.MonkeyPatch, settings: Settings, **extra: str) -> None:
    for name in list(os.environ):
        if name.startswith("PP_") and name != "PP_TEST_SA_PASSWORD":
            monkeypatch.delenv(name)
    monkeypatch.setenv("PP_DB_HOST", settings.db_host)
    monkeypatch.setenv("PP_DB_NAME", settings.db_name)
    monkeypatch.setenv("PP_DB_USER", settings.db_user)
    monkeypatch.setenv("PP_DB_PASSWORD", settings.db_password.get_secret_value())
    monkeypatch.setenv("PP_DB_TRUST_CERT", "true")
    for name, value in extra.items():
        monkeypatch.setenv(name, value)


def test_the_command_exits_0_with_either_decision(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_settings(monkeypatch, world.settings, PP_MODEL_DIR=str(world.champion_dir))
    shutil.copytree(ROOT / "config", Path("config"), dirs_exist_ok=True)

    assert main(["retrain"]) == 0

    out = capsys.readouterr().out
    assert "decision:" in out
    assert "SENTINEL-REVIEW" not in out


def test_the_command_exits_1_when_the_champion_is_missing(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_settings(monkeypatch, world.settings, PP_MODEL_DIR=str(world.champion_dir / "gone"))
    shutil.copytree(ROOT / "config", Path("config"), dirs_exist_ok=True)

    assert main(["retrain"]) == 1
