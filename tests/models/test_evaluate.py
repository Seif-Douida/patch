"""Phase 2's evaluation (plan Task 14, design D6): the champion from dev by rules fixed before any
results, then one report from gold test and Seif's 150."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from toy_reviews import toy_reviews

from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.evaluate import (
    MIN_REVIEWS_PER_S,
    Candidate,
    Choice,
    GoldSet,
    ModelFacts,
    Outputs,
    benchmark_cpu,
    choose_champion,
    evaluate_models,
    render_eval_report,
)
from patchpulse.models.s0 import S0Params, save_s0, train_s0

N = 150


def truth(seed: int = 1) -> NDArray[np.bool_]:
    return np.random.default_rng(seed).random((N, len(ASPECTS))) < 0.25


def noisy(labels: NDArray[np.bool_], flip: float, seed: int) -> NDArray[np.bool_]:
    """The labels with a share `flip` of the (review, aspect) decisions turned around."""
    flips = np.random.default_rng(seed).random(labels.shape) < flip
    return np.asarray(labels ^ flips)


def candidate(family: str, predicted: NDArray[np.bool_], per_s: float = 500.0) -> Candidate:
    return Candidate(family=family, version="v", reviews_per_s=per_s, dev_predicted=predicted)


def test_fast_model_below_the_bar_is_ineligible() -> None:
    dev = truth()
    choice = choose_champion(
        [
            candidate("s0", noisy(dev, 0.15, 2)),
            candidate("s1b", dev.copy(), per_s=MIN_REVIEWS_PER_S - 1),  # perfect, but too slow
        ],
        dev,
        seed=1,
    )

    assert choice.champion == "s0"
    assert choice.eligible == ("s0",)


def test_tie_goes_to_the_cheaper_model() -> None:
    dev = truth()
    better = noisy(dev, 0.10, 3)
    almost = better.copy()
    almost[:2] = ~almost[:2]  # two reviews worse: within the interval's noise
    choice = choose_champion([candidate("s1a", better), candidate("s0", almost)], dev, seed=1)

    assert choice.best == "s1a"
    (comparison,) = choice.comparisons
    assert comparison.low <= 0 <= comparison.high
    assert choice.champion == "s0"


def test_clear_winner_is_champion() -> None:
    dev = truth()
    choice = choose_champion(
        [candidate("s0", noisy(dev, 0.30, 4)), candidate("s1b", noisy(dev, 0.02, 5))], dev, seed=1
    )

    assert choice.champion == "s1b"
    (comparison,) = choice.comparisons
    assert comparison.low > 0


def test_no_eligible_model_means_no_champion() -> None:
    dev = truth()
    choice = choose_champion([candidate("s1a", dev, per_s=10.0)], dev, seed=1)

    assert choice.champion is None
    assert "70" in choice.reason


def test_a_choice_round_trips_as_json(tmp_path: Path) -> None:
    dev = truth()
    choice = choose_champion(
        [candidate("s0", noisy(dev, 0.1, 6)), candidate("s1a", noisy(dev, 0.1, 7), 20.0)],
        dev,
        seed=1,
    )

    assert Choice.from_json(choice.to_json()) == choice


def gold_sets() -> list[GoldSet]:
    test = truth(8)
    languages = ["english"] * 100 + ["french"] * 50
    return [
        GoldSet("test", list(range(N)), test, languages),
        GoldSet("audit", list(range(50)), test[:50], languages[:50]),
    ]


def test_models_are_scored_per_set_language_and_against_the_teacher() -> None:
    sets = gold_sets()
    test, audit = sets
    rng = np.random.default_rng(9)
    probs = np.clip(test.truth * 0.6 + rng.random(test.truth.shape) * 0.4, 0, 1)
    outputs = {
        "teacher": {"test": Outputs(test.truth.copy()), "audit": Outputs(audit.truth.copy())},
        "s0": {
            "test": Outputs(probs >= 0.5, probs),
            "audit": Outputs(probs[:50] >= 0.5, probs[:50]),
        },
    }
    holdout = truth(10)

    result = evaluate_models(
        sets, outputs, holdout_teacher=holdout, holdout_students={"s0": holdout}, seed=1
    )

    teacher = result.score("test", "teacher")
    assert teacher.macro_f1 == 1.0
    assert teacher.ece is None  # no probabilities
    assert set(teacher.by_language) == {"english", "french"}
    s0 = result.score("test", "s0")
    assert s0.ece is not None
    assert s0.macro_f1_ci[0] <= s0.macro_f1 <= s0.macro_f1_ci[1]
    assert result.score("audit", "s0").n == 50
    (pair,) = [p for p in result.paired if p.set_name == "test"]
    assert (pair.a, pair.b) == ("teacher", "s0")
    assert result.agreement["s0"] == 1.0


def facts() -> dict[str, ModelFacts]:
    return {
        "s0": ModelFacts("s0", "20261012T223218Z", 37.6, 400.0, 0.647),
        "s1b": ModelFacts("s1b", "20261012T215556Z", 135.7, 11.0, 0.650),
    }


def test_eval_report_contains_no_review_text() -> None:
    texts, _ = toy_reviews(40, seed=11)
    sets = gold_sets()
    outputs = {
        name: {s.name: Outputs(noisy(s.truth, 0.1, 12)) for s in sets}
        for name in ("teacher", "s0", "s1b")
    }
    result = evaluate_models(
        sets,
        outputs,
        holdout_teacher=truth(13),
        holdout_students={"s0": truth(13), "s1b": truth(14)},
        seed=1,
    )
    dev = truth()
    choice = choose_champion(
        [candidate("s0", noisy(dev, 0.1, 15)), candidate("s1b", dev, per_s=11.0)], dev, seed=1
    )

    report = render_eval_report(result, choice, facts())

    assert "Champion: s0" in report
    assert "gold test" in report.lower()
    assert not any(text in report for text in texts)
    assert not any(ord(ch) > 0x2FFF for ch in report)  # no emoji


class FakeDocker:
    def __init__(self, out: str) -> None:
        self.out = out
        self.command: list[str] = []
        self.texts: list[str] = []

    def __call__(self, command: Sequence[str], **_: Any) -> subprocess.CompletedProcess[str]:
        self.command = list(command)
        work = next(c.split(":/work")[0] for c in command if c.endswith(":/work:ro"))
        self.texts = json.loads((Path(work) / "texts.json").read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(command, 0, stdout=self.out, stderr="")


def test_benchmark_runs_each_model_in_a_one_cpu_container(tmp_path: Path) -> None:
    docker = FakeDocker("noise\ns0\t412.5\ns1b\t11.2\n")
    models = [tmp_path / "s0-1", tmp_path / "s1b-1"]
    for model in models:
        model.mkdir()

    rates = benchmark_cpu(models, ["one", "two"], run=docker, requirements="numpy==2.5.3")

    assert rates == {"s0": 412.5, "s1b": 11.2}
    command = docker.command
    assert command[:2] == ["docker", "run"]
    assert {"--cpus=1", "--cpuset-cpus=0", "--memory=2g", "--rm"} <= set(command)
    assert all(c.endswith(":ro") for c in command if ":/" in c and c != command[-1])
    assert docker.texts == ["one", "two"]


def test_the_container_side_times_each_model(tmp_path: Path) -> None:
    texts, labels = toy_reviews(200, seed=16)
    model_dir = tmp_path / "s0-1"
    save_s0(
        train_s0(texts, labels, params=S0Params()),
        model_dir,
        thresholds=np.full(10, 0.5),
        version="1",
    )
    (tmp_path / "texts.json").write_text(json.dumps(texts[:50]), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "patchpulse.models.benchmark",
            str(tmp_path / "texts.json"),
            str(model_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    family, rate = result.stdout.strip().splitlines()[-1].split("\t")
    assert family == "s0"
    assert float(rate) > 0


def test_a_failed_benchmark_says_what_docker_said(tmp_path: Path) -> None:
    def broken(command: Sequence[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 125, stdout="", stderr="no route to host")

    with pytest.raises(RuntimeError, match="no route to host"):
        benchmark_cpu([tmp_path], ["one"], run=broken, requirements="numpy")
