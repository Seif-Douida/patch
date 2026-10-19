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
            "timestamp_updated": pd.to_datetime(["2026-10-01 09:00", "2026-10-05 18:00"]),
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


def test_export_batches_can_rejudge_the_pairs_graded_0_or_1(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # After the guidelines are revised (D7), only the pairs graded below 2 are judged again; they go
    # to their own folder, so the first batches stay as they were.
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
    judgments = workspace / "data" / "retrieval" / "judgments.csv"
    pd.DataFrame(
        [("q001", 1, 2, "claude-opus-5-5", "v2")],
        columns=["query_id", "review_id", "grade", "judge", "guidelines"],
    ).to_csv(judgments, index=False)

    assert main(["retrieval", "export-batches", "--rejudge-grades", "0,1"]) == 1
    assert "1 pool pair has no judgment" in capsys.readouterr().err
    pd.DataFrame(
        [("q001", 1, 2, "claude-opus-5-5", "v2"), ("q002", 2, 1, "claude-opus-5-5", "v2")],
        columns=["query_id", "review_id", "grade", "judge", "guidelines"],
    ).to_csv(judgments, index=False)

    assert main(["retrieval", "export-batches", "--rejudge-grades", "0,1"]) == 0

    (batch,) = (local / "retrieval_rejudge").glob("batch-*.jsonl")
    assert [json.loads(line)["review_id"] for line in batch.read_text("utf-8").splitlines()] == [2]
    assert not (local / "retrieval_batches").exists()
    assert "1 batch, 1 pair in" in capsys.readouterr().out


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


def agreement_workspace(workspace: Path, languages: Sequence[str]) -> list[tuple[str, int]]:
    """An audit of len(languages) pairs, judged i % 3, its reviews in the given languages."""
    folder = workspace / "data" / "retrieval"
    folder.mkdir(parents=True)
    audit = [(f"q{i % 50:03d}", 9000 + i) for i in range(len(languages))]
    pd.DataFrame(audit, columns=["query_id", "review_id"]).to_csv(folder / "audit.csv", index=False)
    pd.DataFrame(
        [(q, r, i % 3, "claude-sonnet-5-5", "v2") for i, (q, r) in enumerate(audit)],
        columns=["query_id", "review_id", "grade", "judge", "guidelines"],
    ).to_csv(folder / "judgments.csv", index=False)
    pd.DataFrame(
        {
            "review_id": [r for _, r in audit],
            "date": "2026-10-01",
            "language": list(languages),
            "text": SECRET_TEXT,
        }
    ).to_parquet(workspace / "data" / "local" / "reviews.parquet")
    return audit


def write_seif_grades(workspace: Path, rows: Sequence[tuple[str, int, int, bool]]) -> None:
    pd.DataFrame(
        [(q, r, g, unsure, "2026-10-20T09:00:00+00:00") for q, r, g, unsure in rows],
        columns=["query_id", "review_id", "grade", "unsure", "graded_at"],
    ).to_csv(workspace / "data" / "retrieval" / "grades_seif.csv", index=False)


def test_agreement_gate_leaves_out_pairs_seif_could_not_read(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 40 Chinese reviews Seif graded 1 with unsure ticked, his default for text he can't read; on
    # the 260 he read he agrees with the judges throughout.
    audit = agreement_workspace(workspace, ["schinese"] * 40 + ["english"] * 260)
    write_seif_grades(
        workspace,
        [(q, r, 1, True) if i < 40 else (q, r, i % 3, False) for i, (q, r) in enumerate(audit)],
    )

    assert main(["retrieval", "agreement"]) == 0
    out = capsys.readouterr().out
    assert "weighted kappa 1.000 on the 260 pairs Seif read" in out
    assert "40 unread left out" in out
    report = (workspace / "reports" / "retrieval" / "judge-agreement-v2.md").read_text(
        encoding="utf-8"
    )
    assert "260 pairs Seif read" in report
    assert "over all 300 pairs" in report


def test_agreement_command_needs_every_grade_then_applies_the_rule(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    audit = agreement_workspace(workspace, ["english"] * 300)

    def grades(rows: list[tuple[str, int, int]]) -> None:
        write_seif_grades(workspace, [(q, r, g, False) for q, r, g in rows])

    grades([(q, r, i % 3) for i, (q, r) in enumerate(audit[:299])])
    assert main(["retrieval", "agreement"]) == 1  # one pair left
    assert "1 of the 300" in capsys.readouterr().err

    grades([(q, r, i % 3) for i, (q, r) in enumerate(audit)])
    assert main(["retrieval", "agreement"]) == 0
    report = (workspace / "reports" / "retrieval" / "judge-agreement-v2.md").read_text(
        encoding="utf-8"
    )
    assert "**passed**" in report
    assert "passed" in capsys.readouterr().out

    grades([(q, r, 2 - i % 3) for i, (q, r) in enumerate(audit)])  # Seif's grades reversed
    assert main(["retrieval", "agreement"]) == 1
    assert "failed" in capsys.readouterr().out


def test_agreement_after_a_rejudge_writes_its_own_report(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The first check's report stays; the second, on pairs re-judged under v3, gets its own file.
    audit = agreement_workspace(workspace, ["english"] * 300)
    write_seif_grades(workspace, [(q, r, i % 3, False) for i, (q, r) in enumerate(audit)])
    assert main(["retrieval", "agreement"]) == 0
    judgments = workspace / "data" / "retrieval" / "judgments.csv"
    rows = pd.read_csv(judgments, dtype={"query_id": str})
    rows.loc[rows["grade"] < 2, "guidelines"] = "v3"
    rows.to_csv(judgments, index=False)

    assert main(["retrieval", "agreement"]) == 0

    reports = workspace / "reports" / "retrieval"
    assert sorted(path.name for path in reports.iterdir()) == [
        "judge-agreement-v2.md",
        "judge-agreement-v3.md",
    ]
    second = (reports / "judge-agreement-v3.md").read_text(encoding="utf-8")
    assert "guidelines v2 (100 pairs) and v3 (200 pairs)" in second
    assert "optimistic" in second
    assert "judge-agreement-v3.md" in capsys.readouterr().out


def report_workspace(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, *, seif_agrees: bool
) -> Path:
    """Two frozen queries, every run, complete judgments, a two-pair audit and the speed record."""
    frozen_queries(workspace, monkeypatch)
    folder = workspace / "data" / "retrieval"
    found = pd.DataFrame(
        [("q001", 1, 1, 0.5), ("q002", 1, 2, 0.3)],
        columns=["query_id", "rank", "review_id", "score"],
    )
    runs.write_runs(
        {(system, variant): found for system in runs.SYSTEMS for variant in runs.VARIANTS},
        folder / "runs",
    )
    pd.DataFrame(
        [("q001", 1, 2, "claude-sonnet-5-5", "v3"), ("q002", 2, 1, "claude-opus-5-5", "v2")],
        columns=["query_id", "review_id", "grade", "judge", "guidelines"],
    ).to_csv(folder / "judgments.csv", index=False)
    pd.DataFrame([("q001", 1), ("q002", 2)], columns=["query_id", "review_id"]).to_csv(
        folder / "audit.csv", index=False
    )
    write_seif_grades(
        workspace,
        [("q001", 1, 2 if seif_agrees else 1, False), ("q002", 2, 1 if seif_agrees else 2, False)],
    )
    reports = workspace / "reports" / "retrieval"
    reports.mkdir(parents=True)
    (reports / "embedder-speed.json").write_text(
        json.dumps(
            {
                "embedders": [
                    {"name": name, "passages_per_s": rate, "query_p50_ms": 7.0,
                     "query_p95_ms": 50.0, "size_mb": 118.0, "passes_floor": True}
                    for name, rate in zip(CANDIDATES, (5.7, 15.8), strict=True)
                ]
            }
        ),
        encoding="utf-8",
    )  # fmt: skip
    monkeypatch.setattr(
        cli, "local_ann_recall", lambda settings, qs: dict.fromkeys(CANDIDATES, 0.98)
    )
    store = workspace / "mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{store.as_posix()}")
    return reports


def test_report_refuses_without_a_passed_audit(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    reports = report_workspace(workspace, monkeypatch, seif_agrees=False)

    assert main(["retrieval", "report"]) == 1
    assert "--accept-failed-agreement" in capsys.readouterr().err
    assert not (reports / "phase-3-eval.md").exists()

    # Accepted on purpose: the report is written, and says the judgments failed the rule.
    assert main(["retrieval", "report", "--accept-failed-agreement"]) == 0
    text = (reports / "phase-3-eval.md").read_text(encoding="utf-8")
    assert "**failed**" in text
    assert "accepted anyway" in text
    choice = json.loads((reports / "embedder-choice.json").read_text(encoding="utf-8"))
    assert choice["chosen"] in CANDIDATES
    assert "chosen" in capsys.readouterr().out


def test_report_holds_no_text_and_is_tracked(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import mlflow

    reports = report_workspace(workspace, monkeypatch, seif_agrees=True)

    assert main(["retrieval", "report"]) == 0

    text = (reports / "phase-3-eval.md").read_text(encoding="utf-8")
    assert "**passed**" in text
    assert "nobody may publish" not in text  # no review text
    assert "crashes after the patch" not in text  # no query text
    client = mlflow.MlflowClient(f"sqlite:///{(workspace / 'mlflow.db').as_posix()}")
    experiment = client.get_experiment_by_name("retrieval")
    assert experiment is not None
    (run,) = client.search_runs([experiment.experiment_id])
    assert len(run.data.params["queries_sha256"]) == 64
    assert len(run.data.params["judgments_sha256"]) == 64
    assert len(run.data.params["pool_sha256"]) == 64
    assert "hybrid_e5-small.base.NDCG_at_10" in run.data.metrics
    assert {a.path for a in client.list_artifacts(run.info.run_id)} >= {
        "phase-3-eval.md",
        "embedder-choice.json",
    }


def test_report_refuses_without_complete_judgments(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report_workspace(workspace, monkeypatch, seif_agrees=True)
    judgments = workspace / "data" / "retrieval" / "judgments.csv"
    rows = pd.read_csv(judgments, dtype={"query_id": str})
    rows.iloc[:1].to_csv(judgments, index=False)

    assert main(["retrieval", "report"]) == 1
    assert "1 pool pair has no judgment" in capsys.readouterr().err
