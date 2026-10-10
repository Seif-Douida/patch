"""The teacher bake-off on gold dev (plan Task 9): scoring, the pre-set choice rule, the report."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx2
import mlflow
import pytest

from patchpulse.labeling.bakeoff import (
    BakeoffResult,
    CandidateResult,
    PairedDifference,
    choose_teacher,
    log_bakeoff,
    render_report,
    run_bakeoff,
)
from patchpulse.labeling.cache import LabelCache
from patchpulse.labeling.gemini import GeminiResponse
from patchpulse.labeling.teacher import ReviewIn
from patchpulse.labeling.teachers import ModelConfig
from patchpulse.models.cli import main

CODES = ("performance", "content", "story_world")


def model(model_id: str, *, rpd: int = 14_400, tpm: int = 16_000) -> ModelConfig:
    return ModelConfig(
        id=model_id,
        rpm=30,
        tpm=tpm,
        rpd=rpd,
        json_mode=True,
        system_instruction=True,
        use_fraction=0.5,
    )


class Teacher:
    """Answers from `labels` by review id, read from each review's text ("review <id> ...")."""

    def __init__(self, labels: Mapping[int, frozenset[str]], tokens: int = 3_000) -> None:
        self.labels = labels
        self.tokens = tokens

    def generate(
        self,
        system: str,
        user: str,
        *,
        json_schema: Mapping[str, Any] | None,
        expected_output_tokens: int = 512,
        max_output_tokens: int = 4096,
    ) -> GeminiResponse:
        reviews = json.loads(user[user.index("[") :])
        answer = [
            {"alias": r["alias"], "aspects": sorted(self.labels[int(r["text"].split()[1])])}
            for r in reviews
        ]
        return GeminiResponse(json.dumps({"labels": answer}), self.tokens - 300, 300, "STOP")


def dev_set(n: int = 60) -> tuple[list[ReviewIn], dict[int, frozenset[str]]]:
    reviews = [ReviewIn(i, "Game", True, f"review {i} SECRET-TEXT-{i}") for i in range(1, n + 1)]
    gold = {i: frozenset({CODES[i % 3]}) if i % 4 else frozenset() for i in range(1, n + 1)}
    return reviews, gold


def test_bakeoff_scores_each_candidate_against_gold(tmp_path: Path) -> None:
    reviews, gold = dev_set()
    teachers = {"perfect": Teacher(gold), "silent": Teacher({i: frozenset() for i in gold})}

    result = run_bakeoff(
        {"perfect": model("m-perfect"), "silent": model("m-silent")},
        reviews,
        gold,
        make_client=lambda name, _: teachers[name],
        cache=LabelCache(tmp_path / "cache.sqlite"),
        samples=200,
    )

    perfect, silent = result.candidates
    assert perfect.name == "perfect"
    assert perfect.labeled == 60
    assert perfect.calls == 3  # 60 reviews in batches of 20
    assert perfect.f1["performance"] == 1.0
    assert silent.f1["performance"] == 0.0
    assert perfect.macro_f1 > silent.macro_f1
    (pair,) = result.paired
    assert (pair.a, pair.b) == ("perfect", "silent")
    assert pair.low > 0
    # 3,000 tokens a call against 8,000 a minute: 2 calls a minute, 20 reviews each.
    assert perfect.reviews_per_day == pytest.approx(2 * 20 * 1440)


def candidate(name: str, macro: float, per_day: float) -> CandidateResult:
    return CandidateResult(
        name=name,
        model_id=f"id-{name}",
        f1={},
        support={},
        macro_f1=macro,
        macro_f1_ci=(macro - 0.05, macro + 0.05),
        reviews_per_day=per_day,
        calls=8,
        labeled=150,
        missing=0,
    )


def result(pair: PairedDifference) -> BakeoffResult:
    return BakeoffResult(
        n_reviews=150,
        candidates=(candidate("gemma-31b", 0.62, 40_000), candidate("flash-lite", 0.55, 5_000)),
        paired=(pair,),
        prompt_version="v1",
        prompt_hash="0" * 64,
    )


def test_clear_winner_is_chosen() -> None:
    clear = PairedDifference("gemma-31b", "flash-lite", 0.07, 0.02, 0.12)

    assert choose_teacher(result(clear)) == "gemma-31b"


def test_tie_goes_to_the_faster_teacher() -> None:
    faster_but_worse = result(PairedDifference("gemma-31b", "flash-lite", 0.07, -0.01, 0.15))
    swapped = BakeoffResult(
        n_reviews=150,
        candidates=(candidate("gemma-31b", 0.62, 5_000), candidate("flash-lite", 0.55, 40_000)),
        paired=faster_but_worse.paired,
        prompt_version="v1",
        prompt_hash="0" * 64,
    )

    assert choose_teacher(faster_but_worse) == "gemma-31b"  # tied, and faster
    assert choose_teacher(swapped) == "flash-lite"  # tied, and faster


def test_bakeoff_is_logged_to_mlflow_with_its_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    reviews, gold = dev_set()
    outcome = run_bakeoff(
        {"perfect": model("m-perfect")},
        reviews,
        gold,
        make_client=lambda name, _: Teacher(gold),
        cache=LabelCache(tmp_path / "cache.sqlite"),
        samples=100,
    )
    report = Path("reports/aspects/teacher-bakeoff.md")
    report.parent.mkdir(parents=True)
    report.write_text(render_report(outcome, "perfect"), encoding="utf-8")

    run_id = log_bakeoff(outcome, "perfect", report)

    logged = mlflow.get_run(run_id)
    assert logged.data.tags["winner"] == "perfect"
    assert logged.data.params["dev_n"] == "60"
    assert logged.data.metrics["perfect.macro_f1"] == pytest.approx(outcome.candidates[0].macro_f1)
    assert "perfect.f1.performance" in logged.data.metrics
    artifacts = [a.path for a in mlflow.MlflowClient().list_artifacts(run_id)]
    assert artifacts == ["teacher-bakeoff.md"]


def test_bakeoff_report_contains_no_review_text(tmp_path: Path) -> None:
    reviews, gold = dev_set()

    outcome = run_bakeoff(
        {"perfect": model("m-perfect")},
        reviews,
        gold,
        make_client=lambda name, _: Teacher(gold),
        cache=LabelCache(tmp_path / "cache.sqlite"),
        samples=100,
    )
    report = render_report(outcome, choose_teacher(outcome))

    assert "SECRET-TEXT" not in report
    assert "review 1 " not in report
    assert "**Teacher: perfect**" in report
    assert "| performance |" in report


def no_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[object]:
    """No PP_ or MLflow variables, no .env, and a record of any HTTP request attempted."""
    for name in list(os.environ):
        if name.startswith(("PP_", "MLFLOW_")):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    requests: list[object] = []
    monkeypatch.setattr(httpx2.Client, "send", lambda *args, **kwargs: requests.append(args))
    return requests


def test_bakeoff_command_refuses_without_mlflow_before_any_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    requests = no_settings(monkeypatch, tmp_path)
    monkeypatch.setenv("PP_GEMINI_API_KEY", "AIza-test-key")

    assert main(["teacher", "bakeoff"]) == 2
    assert "MLFLOW_TRACKING_URI" in capsys.readouterr().err
    assert requests == []


def test_bakeoff_command_needs_the_gemini_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    requests = no_settings(monkeypatch, tmp_path)

    assert main(["teacher", "bakeoff"]) == 2
    assert "PP_GEMINI_API_KEY" in capsys.readouterr().err
    assert requests == []
