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
from patchpulse.retrieval import cli, load, queries, runs
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


def frozen_queries(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = workspace / "data" / "retrieval"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "queries.csv"
    path.write_text(
        "query_id,appid,language,kind,text,start_day,end_day,voted_up,aspect\n"
        "q001,553850,english,aspect,crashes after the patch,,,,\n"
        "q002,553850,english,aspect,server down,,,true,\n",
        encoding="utf-8",
        newline="",
    )
    monkeypatch.setattr(queries, "_tracked_games", lambda: {553850})
    queries.freeze_queries(path)
    return path


def test_run_command_writes_every_run_and_sizes_the_pool(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    frozen_queries(workspace, monkeypatch)
    for name in CANDIDATES:
        write_tiny_embedder(workspace / name, replace(CANDIDATES[name], dims=4, max_tokens=16))
    monkeypatch.setattr(cli, "embedder_dir", lambda name: workspace / name)

    def run_systems(client: object, path: Path, embedders: dict[str, object]) -> runs.Runs:
        assert path == Path("data/retrieval/queries.csv")
        assert list(embedders) == list(CANDIDATES)
        found = pd.DataFrame(
            [("q001", 1, 1, 0.5), ("q001", 2, 2, 0.4), ("q002", 1, 2, 0.3)],
            columns=["query_id", "rank", "review_id", "score"],
        )
        return {(system, variant): found for system in runs.SYSTEMS for variant in runs.VARIANTS}

    monkeypatch.setattr(runs, "run_systems", run_systems)

    assert main(["retrieval", "run"]) == 0

    written = sorted(p.name for p in (workspace / "data" / "retrieval" / "runs").iterdir())
    assert len(written) == 2 * len(runs.SYSTEMS)
    assert "bm25-base.csv" in written
    out = capsys.readouterr().out
    assert "pool: 3 pairs for 2 queries, 1 to 2 a query" in out


def test_run_command_refuses_queries_that_are_not_frozen(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in CANDIDATES:
        write_tiny_embedder(workspace / name, replace(CANDIDATES[name], dims=4, max_tokens=16))
    monkeypatch.setattr(cli, "embedder_dir", lambda name: workspace / name)

    assert main(["retrieval", "run"]) == 1
    assert "frozen" in capsys.readouterr().err
    assert not (workspace / "data" / "retrieval" / "runs").exists()


def test_export_batches_command_writes_local_batches_only(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    frozen_queries(workspace, monkeypatch)
    local = workspace / "data" / "local"
    pd.DataFrame({"appid": [553850], "name": ["HELLDIVERS 2"]}).to_parquet(local / "games.parquet")
    found = pd.DataFrame(
        [("q001", 1, 1, 0.5), ("q002", 1, 2, 0.3)],
        columns=["query_id", "rank", "review_id", "score"],
    )
    runs.write_runs(
        {(system, variant): found for system in runs.SYSTEMS for variant in runs.VARIANTS},
        workspace / "data" / "retrieval" / "runs",
    )

    assert main(["retrieval", "export-batches"]) == 0

    (batch,) = (local / "retrieval_batches").glob("batch-*.jsonl")
    assert SECRET_TEXT.strip() in batch.read_text(encoding="utf-8")
    assert (local / "retrieval_batches" / "manifest.json").is_file()
    out = capsys.readouterr().out
    assert "1 batch, 2 pairs" in out
    assert "nobody may publish" not in out


def test_import_judgments_command_checks_then_merges(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    batch = workspace / "data" / "local" / "retrieval_batches" / "batch-001.jsonl"
    batch.parent.mkdir(parents=True)
    batch.write_text(
        '{"pair": 1, "query_id": "q001", "review_id": 1, "text": "it crashes"}\n'
        '{"pair": 2, "query_id": "q001", "review_id": 2, "text": "nice music"}\n',
        encoding="utf-8",
    )
    answers = workspace / "data" / "local" / "answers-001.csv"
    answers.write_text(
        "query_id,review_id,grade,reason\nq001,1,2,About crashes.\nq001,2,5,Music only.\n",
        encoding="utf-8",
    )
    command = [
        "retrieval", "import-judgments", "--batch", str(batch), "--answers", str(answers),
        "--judge", "claude-opus-5-5",
    ]  # fmt: skip

    assert main([*command, "--guidelines", "v1"]) == 1  # a grade of 5
    assert "grade '5'" in capsys.readouterr().err
    assert not (workspace / "data" / "retrieval" / "judgments.csv").exists()
    answers.write_text(
        "query_id,review_id,grade,reason\nq001,1,2,About crashes.\nq001,2,0,Music only.\n",
        encoding="utf-8",
    )
    assert main([*command, "--guidelines", "v1"]) == 0
    assert "2 pairs graded (0: 1, 1: 0, 2: 1)" in capsys.readouterr().out
    judged = pd.read_csv(workspace / "data" / "retrieval" / "judgments.csv")
    assert judged["grade"].tolist() == [2, 0]
    assert set(judged["judge"]) == {"claude-opus-5-5"}


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


def test_audit_sample_command_draws_300_pairs_once(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    found = pd.DataFrame(
        [(f"q{i % 50:03d}", 1, 9000 + i, 0.5) for i in range(400)],
        columns=["query_id", "rank", "review_id", "score"],
    )
    runs.write_runs(
        {(system, variant): found for system in runs.SYSTEMS for variant in runs.VARIANTS},
        workspace / "data" / "retrieval" / "runs",
    )

    pilot = workspace / "data" / "local" / "retrieval_pilot"
    pilot.mkdir(parents=True)
    seen = [(f"q{i % 50:03d}", 9000 + i) for i in range(100)]  # shown to Seif with grades
    pd.DataFrame(
        [(q, r, 1, "claude", "v1") for q, r in seen],
        columns=["query_id", "review_id", "grade", "judge", "guidelines"],
    ).to_csv(pilot / "judgments.csv", index=False)

    assert main(["retrieval", "audit-sample"]) == 0
    assert main(["retrieval", "audit-sample"]) == 0  # the same draw: nothing changes

    audit = pd.read_csv(workspace / "data" / "retrieval" / "audit.csv", dtype={"query_id": str})
    assert list(audit.columns) == ["query_id", "review_id"]
    assert len(audit) == 300
    assert not set(seen) & set(zip(audit["query_id"], audit["review_id"], strict=True))
    assert "300 pairs" in capsys.readouterr().out


def test_agreement_command_needs_every_grade_then_applies_the_rule(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = workspace / "data" / "retrieval"
    folder.mkdir(parents=True)
    audit = [(f"q{i % 50:03d}", 9000 + i) for i in range(300)]
    pd.DataFrame(audit, columns=["query_id", "review_id"]).to_csv(folder / "audit.csv", index=False)
    pd.DataFrame(
        [(q, r, i % 3, "claude-sonnet-5-5", "v2") for i, (q, r) in enumerate(audit)],
        columns=["query_id", "review_id", "grade", "judge", "guidelines"],
    ).to_csv(folder / "judgments.csv", index=False)

    def grades(rows: list[tuple[str, int, int]]) -> None:
        pd.DataFrame(
            [(q, r, g, False, "2026-10-20T09:00:00+00:00") for q, r, g in rows],
            columns=["query_id", "review_id", "grade", "unsure", "graded_at"],
        ).to_csv(folder / "grades_seif.csv", index=False)

    grades([(q, r, i % 3) for i, (q, r) in enumerate(audit[:299])])
    assert main(["retrieval", "agreement"]) == 1  # one pair left
    assert "1 of the 300" in capsys.readouterr().err

    grades([(q, r, i % 3) for i, (q, r) in enumerate(audit)])
    assert main(["retrieval", "agreement"]) == 0
    report = (workspace / "reports" / "retrieval" / "judge-agreement.md").read_text(
        encoding="utf-8"
    )
    assert "**passed**" in report
    assert "passed" in capsys.readouterr().out

    grades([(q, r, 2 - i % 3) for i, (q, r) in enumerate(audit)])  # Seif's grades reversed
    assert main(["retrieval", "agreement"]) == 1
    assert "failed" in capsys.readouterr().out
