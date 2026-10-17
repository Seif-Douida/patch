"""`patchpulse-ml retrieval ...` (Phase 3's local commands), with Docker and downloads faked."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from qdrant_client import QdrantClient
from tiny_encoder import write_tiny_embedder

from patchpulse.models.cli import main
from patchpulse.retrieval import cli, load, queries
from patchpulse.retrieval.benchmark import EmbedderSpeed
from patchpulse.retrieval.embedders import CANDIDATES

SPEED = Path("reports/retrieval/embedder-speed.json")
SECRET_TEXT = "a review nobody may publish " * 5


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for name in list(os.environ):
        if name.startswith(("PP_", "MLFLOW_")):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "data" / "local"
    local.mkdir(parents=True)
    pd.DataFrame(
        {
            "review_id": [1, 2],
            "date": ["2026-10-01", "2026-10-02"],
            "language": ["english", "english"],
            "text": [SECRET_TEXT, SECRET_TEXT],
        }
    ).to_parquet(local / "reviews.parquet")
    pd.DataFrame({"appid": [553850]}).to_parquet(local / "games.parquet")
    pd.DataFrame({"appid": [553850]}).to_parquet(local / "patches.parquet")
    return tmp_path


def test_benchmark_command_writes_the_report(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[str] = []

    def embedder_dir(name: str) -> Path:
        built.append(name)
        write_tiny_embedder(workspace / name, CANDIDATES[name])
        return workspace / name

    def benchmark(
        dirs: Sequence[Path], texts: Sequence[str], queries: Sequence[str]
    ) -> list[EmbedderSpeed]:
        assert list(texts) == [SECRET_TEXT.strip()] * 2  # the newest eligible, cleaned
        return [
            EmbedderSpeed("e5-small", 40.0, 20.0, 35.0, 120.0),
            EmbedderSpeed("minilm-l12", 1.5, 90.0, 400.0, 280.0),
        ]

    monkeypatch.setattr(cli, "embedder_dir", embedder_dir)
    monkeypatch.setattr(cli, "benchmark_embedders", benchmark)

    assert main(["retrieval", "benchmark", "--models", "e5-small,minilm-l12"]) == 0

    report = json.loads(SPEED.read_text(encoding="utf-8"))
    assert built == ["e5-small", "minilm-l12"]
    assert [(e["name"], e["passes_floor"]) for e in report["embedders"]] == [
        ("e5-small", True),
        ("minilm-l12", False),
    ]
    assert report["texts"] == 2
    assert report["floor"] == {
        "reviews_in_index_budget": 650,
        "budget_s": 300,
        "query_p95_ms": 300.0,
    }
    assert "nobody may publish" not in SPEED.read_text(encoding="utf-8")


def test_benchmark_refuses_an_unknown_model(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["retrieval", "benchmark", "--models", "e5-small,arctic-m-v2"]) == 2
    assert "arctic-m-v2" in capsys.readouterr().err


def test_embed_command_embeds_the_index_set(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pd.DataFrame(
        {"review_id": [1, 2], "version": pd.to_datetime(["2026-10-01", "2026-10-02"])}
    ).to_parquet(workspace / "data" / "local" / "index.parquet")
    # A tiny stand-in under e5-small's name (four dimensions, sixteen tokens).
    write_tiny_embedder(workspace / "tiny", replace(CANDIDATES["e5-small"], dims=4, max_tokens=16))
    monkeypatch.setattr(cli, "embedder_dir", lambda name: workspace / "tiny")
    caplog.set_level("INFO", logger="patchpulse")

    assert main(["retrieval", "embed", "--model", "e5-small"]) == 0
    assert main(["retrieval", "embed", "--model", "e5-small"]) == 0  # a rerun reuses everything

    (cache,) = (workspace / "data" / "local" / "embeddings").iterdir()
    assert cache.name.startswith("e5-small-")
    assert "2 vectors for the 2 reviews of the index set (0 embedded, 2 reused" in caplog.text
    assert "nobody may publish" not in caplog.text


@pytest.mark.qdrant
def test_load_command_builds_the_local_collections(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, qdrant: QdrantClient
) -> None:
    pd.DataFrame(
        {
            "review_id": [1, 2],
            "appid": [553850, 553850],
            "day": [20261001, 20261002],
            "created_ts": [1_790_000_001, 1_790_000_002],
            "language": ["english", "english"],
            "voted_up": [True, False],
            "playtime_bucket": [None, "lt_2h"],
            "first_seen_run_id": [1, 1],
            "version": pd.to_datetime(["2026-10-01", "2026-10-02"]),
            "aspects": ["", "performance"],
            "aspect_model_version": [None, "s0-1"],
            "fingerprint": ["a" * 64, "b" * 64],
            "rules_version": ["r1", "r1"],
        }
    ).to_parquet(workspace / "data" / "local" / "index.parquet")
    write_tiny_embedder(workspace / "tiny", replace(CANDIDATES["e5-small"], dims=4, max_tokens=16))
    monkeypatch.setattr(cli, "embedder_dir", lambda name: workspace / "tiny")
    name = f"pp_test_eval_{workspace.name[-8:]}"
    monkeypatch.setattr(load, "eval_collection", lambda model: name)
    assert main(["retrieval", "embed", "--model", "e5-small"]) == 0

    try:
        assert main(["retrieval", "load", "--target", "local", "--models", "e5-small"]) == 0
        assert qdrant.count(name, exact=True).count == 2
        assert main(["retrieval", "load", "--target", "local", "--models", "e5-small"]) == 0
    finally:
        qdrant.delete_collection(name)


def test_queries_check_then_freeze(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = workspace / "data" / "retrieval"
    folder.mkdir(parents=True)
    (folder / "queries.csv").write_text(
        "query_id,appid,language,kind,text,start_day,end_day,voted_up,aspect\n"
        "q001,553850,english,aspect,crashes after the patch,,,,\n"
        "q002,553850,french,event,la mise a jour,20260901,20260915,,\n",
        encoding="utf-8",
        newline="",
    )
    monkeypatch.setattr(queries, "_tracked_games", lambda: {553850})

    assert main(["retrieval", "queries", "check"]) == 0
    out = capsys.readouterr().out
    assert "2 queries" in out
    assert "1 with extra filters" in out
    assert main(["retrieval", "queries", "freeze"]) == 0
    assert (folder / "queries.sha256").is_file()
    (folder / "queries.csv").write_text("query_id\n", encoding="utf-8")
    assert main(["retrieval", "queries", "check"]) == 1  # broken: says why, exits 1
    assert "columns" in capsys.readouterr().err
