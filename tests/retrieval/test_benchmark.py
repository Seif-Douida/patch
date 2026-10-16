"""The speed floor (plan Task 5, design D3): each candidate embedder is timed where it will run, in
a one-CPU container for a night's reviews and a half-CPU one for single queries."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from tiny_encoder import write_tiny_embedder

from patchpulse.models.snapshot import Snapshot
from patchpulse.retrieval.benchmark import (
    BENCH_QUERIES,
    EmbedderSpeed,
    benchmark_embedders,
    newest_index_texts,
    passes_floor,
)
from patchpulse.retrieval.embedders import MODEL_FILE, EmbedderSpec

TINY = EmbedderSpec(
    name="tiny",
    model_id="test/tiny",
    revision="0" * 40,
    onnx_file="onnx/model.onnx",
    tokenizer_file="tokenizer.json",
    pooling="mean",
    query_prefix="query: ",
    passage_prefix="passage: ",
    max_tokens=16,
    dims=4,
    licence="MIT",
)


def speed(per_s: float, p95: float) -> EmbedderSpeed:
    return EmbedderSpeed(
        name="m", passages_per_s=per_s, query_p50_ms=p95 / 2, query_p95_ms=p95, size_mb=1.0
    )


def test_floor_rule() -> None:
    # About 650 new reviews a night within the 5-minute index budget on one vCPU, and one query
    # embedded in under 300 ms on pp-api's half vCPU.
    assert passes_floor(speed(2.2, 299.0))
    assert not passes_floor(speed(2.1, 100.0))  # 630 reviews in 300 s
    assert not passes_floor(speed(50.0, 300.0))


class FakeDocker:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.inputs: list[list[str]] = []

    def __call__(self, command: Sequence[str], **_: Any) -> subprocess.CompletedProcess[str]:
        self.commands.append(list(command))
        work = next(c.split(":/work")[0] for c in command if c.endswith(":/work:ro"))
        mode = "passages" if "passages" in command[-1] else "queries"
        self.inputs.append(json.loads((Path(work) / f"{mode}.json").read_text(encoding="utf-8")))
        out = "noise\ntiny\t42.5\n" if mode == "passages" else "tiny\t12.0\t20.5\n"
        return subprocess.CompletedProcess(command, 0, stdout=out, stderr="")


def test_benchmark_command_is_one_cpu_and_read_only(tmp_path: Path) -> None:
    write_tiny_embedder(tmp_path / "tiny", TINY)
    docker = FakeDocker()

    (result,) = benchmark_embedders(
        [tmp_path / "tiny"], ["one", "two"], ["q"], run=docker, requirements="numpy==2.5.3"
    )

    assert result == EmbedderSpeed(
        name="tiny",
        passages_per_s=42.5,
        query_p50_ms=12.0,
        query_p95_ms=20.5,
        size_mb=round((tmp_path / "tiny" / MODEL_FILE).stat().st_size / 1e6, 1),
    )
    passages, queries = docker.commands
    assert {"--rm", "--cpus=1", "--cpuset-cpus=0", "--memory=2g"} <= set(passages)
    assert {"--rm", "--cpus=0.5", "--cpuset-cpus=0", "--memory=1g"} <= set(queries)
    for command in docker.commands:
        assert command[:2] == ["docker", "run"]
        assert all(c.endswith(":ro") for c in command if ":/" in c and c != command[-1])
    assert docker.inputs == [["one", "two"], ["q"]]


def test_a_failed_benchmark_says_what_docker_said(tmp_path: Path) -> None:
    write_tiny_embedder(tmp_path / "tiny", TINY)

    def broken(command: Sequence[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 125, stdout="", stderr="no route to host")

    with pytest.raises(RuntimeError, match="no route to host"):
        benchmark_embedders([tmp_path / "tiny"], ["one"], ["q"], run=broken, requirements="numpy")


@pytest.mark.parametrize("mode", ["passages", "queries"])
def test_the_container_side_times_each_embedder(tmp_path: Path, mode: str) -> None:
    write_tiny_embedder(tmp_path / "tiny", TINY)
    texts = ["fps drops after the patch"] * 40 if mode == "passages" else BENCH_QUERIES[:10]
    (tmp_path / f"{mode}.json").write_text(json.dumps(texts), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "patchpulse.retrieval.benchmark",
            mode,
            str(tmp_path / f"{mode}.json"),
            str(tmp_path / "tiny"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    name, *numbers = result.stdout.strip().splitlines()[-1].split("\t")
    assert name == "tiny"
    assert len(numbers) == (1 if mode == "passages" else 2)
    assert all(float(n) > 0 for n in numbers)


def test_bench_queries_are_query_like_and_not_the_eval() -> None:
    assert len(BENCH_QUERIES) == len(set(BENCH_QUERIES)) == 100
    assert all(2 <= len(q.split()) <= 12 for q in BENCH_QUERIES)


def snapshot(index: pd.DataFrame | None = None) -> Snapshot:
    reviews = pd.DataFrame(
        {
            "review_id": [1, 2, 3, 4, 5],
            "date": ["2026-10-01", "2026-10-03", "2026-10-02", "2026-10-03", "2026-10-04"],
            "language": ["english", "english", "schinese", "english", "english"],
            "text": [
                " ".join(["old"] * 25),
                "[b]" + " ".join(["two"] * 25) + "[/b]",
                "好" * 45,
                "too short",
                " ".join(["new"] * 30),
            ],
        }
    )
    frame = pd.DataFrame({"appid": [553850]})
    if index is None:
        return Snapshot(reviews=reviews, games=frame, patches=frame)
    return Snapshot(reviews=reviews, games=frame, patches=frame, index=index)


def test_newest_index_texts_follow_the_index_set() -> None:
    index = pd.DataFrame({"review_id": [1, 3], "created_ts": [100, 50]})

    assert newest_index_texts(snapshot(index), 5) == [" ".join(["old"] * 25), "好" * 45]


def test_newest_index_texts_apply_the_rules_without_an_index_set() -> None:
    # Before the first pull of core.review_index: the same rules, applied to the snapshot.
    texts = newest_index_texts(snapshot(), 3)

    assert texts == [" ".join(["new"] * 30), " ".join(["two"] * 25), "好" * 45]  # cleaned
