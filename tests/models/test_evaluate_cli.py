"""`patchpulse-ml evaluate choose` and `evaluate report` (plan Task 14): the champion from dev and
the container benchmark first; then gold test, read once, for the report."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from student_inputs import N_AUDIT, N_DEV, N_SILVER, N_TEST, TEACHER, write_inputs
from tiny_encoder import write_tiny_encoder

from patchpulse.labeling.cache import TEACHER_CACHE_FILE, LabelCache
from patchpulse.labeling.gold import parse_aspects
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.labeling.teacher import prompt_hash
from patchpulse.models import cli
from patchpulse.models.artifact import read_artifact
from patchpulse.models.champion import REGISTERED_MODEL, read_pin
from patchpulse.models.cli import main
from patchpulse.models.evaluate import Choice

CHOICE = Path("reports/aspects/champion-choice.json")
REPORT = Path("reports/aspects/phase-2-eval.md")


def models_yaml(root: Path) -> None:
    (root / "config").mkdir()
    (root / "config" / "models.yaml").write_text(
        "candidates:\n"
        "  server:\n"
        "    provider: openai_compat\n"
        "    id: gemma-31b\n"
        "    json_mode: true\n"
        "    system_instruction: true\n"
        "teacher: server\n",
        encoding="utf-8",
    )


@pytest.fixture
def trained(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """S0 and S1a (on a tiny fake encoder) trained on the invented inputs."""
    for name in list(os.environ):
        if name.startswith(("PP_", "MLFLOW_")):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    texts = write_inputs(tmp_path)
    models_yaml(tmp_path)
    write_tiny_encoder(tmp_path / "tiny-encoder")
    monkeypatch.setattr(cli, "_encoder_dir", lambda: tmp_path / "tiny-encoder")
    assert main(["train", "s0"]) == 0
    assert main(["train", "s1a"]) == 0
    monkeypatch.setattr(cli, "_benchmark", lambda dirs, texts: {"s0": 900.0, "s1a": 15.0})
    return texts


def cache_teacher_labels(root: Path) -> None:
    """The teacher's gold labels, as `teacher label --set gold` leaves them: Claude's, here."""
    gold = pd.read_csv(root / "data" / "gold" / "labels_claude.csv", keep_default_na=False)
    labels = {
        int(i): parse_aspects(a) for i, a in zip(gold["review_id"], gold["aspects"], strict=True)
    }
    LabelCache(root / "data" / "local" / TEACHER_CACHE_FILE).put_many(
        labels, model=TEACHER, prompt_hash=prompt_hash(), created_at=datetime.now(UTC)
    )


def test_evaluate_choose_picks_the_champion_on_dev(
    trained: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["evaluate", "choose"]) == 0

    choice = Choice.from_json(CHOICE.read_text(encoding="utf-8"))
    assert choice.champion == "s0"
    assert choice.eligible == ("s0",)  # S1a: 15 reviews a second
    assert set(choice.versions) == {"s0", "s1a"}
    assert "Champion: s0" in capsys.readouterr().out
    assert not REPORT.exists()


def test_evaluate_report_needs_a_choice_first(
    trained: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["evaluate", "report"]) == 2
    assert "evaluate choose" in capsys.readouterr().err


def test_evaluate_report_needs_the_teachers_gold_labels(
    trained: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["evaluate", "choose"]) == 0

    assert main(["evaluate", "report"]) == 2
    assert "teacher label --set gold" in capsys.readouterr().err


def test_evaluate_report_reads_gold_test_once(
    trained: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_teacher_labels(tmp_path)
    assert main(["evaluate", "choose"]) == 0

    assert main(["evaluate", "report"]) == 0

    report = REPORT.read_text(encoding="utf-8")
    assert "Champion: s0" in report
    assert f"{N_TEST} reviews" in report
    assert f"{N_AUDIT} reviews" in report
    assert not any(text in report for text in trained)
    for model_dir in Path("build/models").glob("*-*"):
        metrics = read_artifact(model_dir).metrics
        assert {"test_macro_f1", "test_macro_f1_low", "test_macro_f1_high"} <= set(metrics)
        # Each aspect's F1 too: the site marks the aspects the model detects less reliably.
        assert {f"test_f1_{code}" for code in ASPECTS} <= set(metrics)
    (run,) = mlflow.search_runs(
        experiment_names=["evaluation"], filter_string="tags.step = 'report'", output_format="list"
    )
    assert run.data.params["gold_test_n"] == str(N_TEST)
    assert run.data.metrics["teacher.test_macro_f1"] == 1.0
    assert [a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id)] == [REPORT.name]
    capsys.readouterr()

    assert main(["evaluate", "report"]) == 2  # gold test is read once
    assert "--force" in capsys.readouterr().err


def test_the_choice_file_holds_numbers_only(trained: list[str]) -> None:
    assert main(["evaluate", "choose"]) == 0

    record = json.loads(CHOICE.read_text(encoding="utf-8"))
    flat = json.dumps(record)
    assert not any(text in flat for text in trained)
    assert len(trained) == N_SILVER + N_DEV + N_TEST


def test_champion_register_pins_the_evaluated_champion(
    trained: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_teacher_labels(tmp_path)
    assert main(["evaluate", "choose"]) == 0
    assert main(["champion", "register"]) == 2  # models are registered with their test scores
    assert "evaluate report" in capsys.readouterr().err
    assert main(["evaluate", "report"]) == 0

    assert main(["champion", "register"]) == 0

    pin = read_pin(Path("models/champion.yaml"))
    assert pin.family == "s0"
    client = mlflow.MlflowClient()
    assert (
        int(client.get_model_version_by_alias(REGISTERED_MODEL, "champion").version) == pin.version
    )
    challenger = client.get_model_version_by_alias(REGISTERED_MODEL, "challenger")
    assert challenger.tags["family"] == "s1a"
