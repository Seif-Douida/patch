"""The eval commands (Phase 6 plan Task 4): fetching and pinning the published data bundle,
freezing the eval datasets, and refusing a graded run on unfrozen ones."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from agent_fixtures import DATA_VERSION, make_bundle
from eval_fixtures import smoke_metrics

from patchpulse.agent.bundle import BUNDLE_FILE, REPORTS_FILE, dump_bundle
from patchpulse.evals.cli import main
from patchpulse.evals.data import fetch_data, load_pinned_bundle, read_data_pin
from patchpulse.evals.frozen import FROZEN_FILE, freeze, require_frozen


def published(reports_version: int) -> dict[str, bytes]:
    return {
        BUNDLE_FILE: dump_bundle(make_bundle()),
        REPORTS_FILE: json.dumps({"data_version": reports_version, "reports": {}}).encode(),
    }


def test_fetch_pins_the_published_bundle(tmp_path: Path) -> None:
    files = published(DATA_VERSION)
    where, pin_path = tmp_path / "data" / "local" / "evals", tmp_path / "data.yaml"

    pin = fetch_data(files.__getitem__, root=where, pin_path=pin_path)

    assert pin == read_data_pin(pin_path)
    assert pin.data_version == DATA_VERSION
    assert load_pinned_bundle(pin, root=where).data_version == DATA_VERSION


def test_fetch_refuses_a_bundle_and_reports_that_disagree(tmp_path: Path) -> None:
    files = published(DATA_VERSION - 1)

    with pytest.raises(ValueError, match="disagree"):
        fetch_data(files.__getitem__, root=tmp_path / "data" / "local", pin_path=tmp_path / "p")

    assert not (tmp_path / "p").exists()


def test_a_changed_bundle_file_is_refused(tmp_path: Path) -> None:
    where, pin_path = tmp_path / "data" / "local" / "evals", tmp_path / "data.yaml"
    pin = fetch_data(published(DATA_VERSION).__getitem__, root=where, pin_path=pin_path)
    (where / f"data-v{DATA_VERSION}.json.gz").write_bytes(dump_bundle(make_bundle()) + b"x")

    with pytest.raises(ValueError, match="hash"):
        load_pinned_bundle(pin, root=where)


def test_a_missing_pin_says_how_to_fetch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="data --fetch"):
        read_data_pin(tmp_path / "data.yaml")


def frozen_files(base: Path) -> None:
    (base / "evals" / "incidents").mkdir(parents=True)
    (base / "evals" / "injection").mkdir()
    (base / "evals" / "judge").mkdir()
    (base / "evals" / "incidents" / "a.yaml").write_text("id: a\n", encoding="utf-8")
    (base / "evals" / "injection" / "cases.yaml").write_text("cases: []\n", encoding="utf-8")
    (base / "evals" / "judge" / "rubric.md").write_text("# Rubric\n", encoding="utf-8")


def test_freeze_records_and_require_frozen_names_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    frozen_files(tmp_path)

    freeze()
    require_frozen()
    (tmp_path / "evals" / "incidents" / "a.yaml").write_text("id: b\n", encoding="utf-8")

    assert "scoring_version" in FROZEN_FILE.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match=r"a\.yaml"):
        require_frozen()


def test_the_main_arm_needs_frozen_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PP_QDRANT_URL", "http://localhost:1")  # a search is configured
    frozen_files(tmp_path)

    code = main(["run", "--arm", "main", "--encoder-dir", str(tmp_path)])

    assert code == 2
    assert "freeze" in capsys.readouterr().err


KEY = base64.b64encode(b"k" * 32).decode()  # a low-entropy test key


def test_keygen_prints_a_new_key(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["keygen"])

    assert code == 0
    assert len(base64.b64decode(capsys.readouterr().out.strip(), validate=True)) == 32


def test_the_bundle_command_needs_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PP_EVAL_BUNDLE_KEY", raising=False)

    code = main(["bundle", "--version", "1"])

    assert code == 2
    error = capsys.readouterr().err
    assert "PP_EVAL_BUNDLE_KEY" in error
    assert "keygen" in error


def test_a_pinned_bundle_version_is_never_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PP_EVAL_BUNDLE_KEY", KEY)
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "thresholds.yaml").write_text(
        f"bundle:\n  version: 1\n  sha256: {'a' * 64}\n", encoding="utf-8"
    )

    code = main(["bundle", "--version", "1"])

    assert code == 2
    assert "build v2" in capsys.readouterr().err


def test_smoke_needs_a_pinned_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PP_EVAL_BUNDLE_KEY", KEY)
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "thresholds.yaml").write_text("bundle: null\n", encoding="utf-8")

    code = main(["smoke", "--bundle", "eval-bundle-v1.bin", "--out", "metrics.json"])

    assert code == 2
    assert "no eval bundle is pinned" in capsys.readouterr().err


def test_the_gate_command_fails_on_a_breach_after_writing_the_comment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base, worse, comment = tmp_path / "base.json", tmp_path / "pr.json", tmp_path / "comment.md"
    base.write_text(json.dumps(smoke_metrics().to_json()), encoding="utf-8")
    worse.write_text(json.dumps(smoke_metrics(root_cause_correct=10).to_json()), encoding="utf-8")

    failed = main(["gate", "--pr", str(worse), "--base", str(base), "--comment", str(comment)])
    failed_comment = comment.read_text(encoding="utf-8")
    alone = main(["gate", "--pr", str(worse), "--comment", str(comment)])

    assert failed == 1
    assert failed_comment.startswith("## Eval gate: failed")
    assert alone == 0
    assert "no baseline yet" in comment.read_text(encoding="utf-8")
    assert "failed" in capsys.readouterr().out


def test_a_live_run_needs_review_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run without search cites no reviews, and its cases couldn't replay (Final review I2)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PP_GEMINI_API_KEY", "g" * 39)  # a low-entropy test key
    monkeypatch.delenv("PP_QDRANT_URL", raising=False)

    code = main(["run", "--arm", "a2"])

    assert code == 2
    assert "review search" in capsys.readouterr().err


def test_the_injection_arm_needs_the_main_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PP_QDRANT_URL", "http://localhost:1")  # a search is configured
    frozen_files(tmp_path)
    freeze()

    code = main(["run", "--arm", "injection", "--encoder-dir", str(tmp_path)])

    assert code == 2
    assert "run --arm main" in capsys.readouterr().err


def test_the_judge_needs_stored_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    frozen_files(tmp_path)
    freeze()

    code = main(["judge", "--arms", "main,a2"])

    assert code == 2
    assert "run --arm" in capsys.readouterr().err


def test_the_judge_refuses_an_unknown_arm(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["judge", "--arms", "main,a9"])

    assert code == 2
    assert "a9" in capsys.readouterr().err


def test_the_report_needs_a_stored_main_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)

    code = main(["report"])

    assert code == 2
    assert "run --arm main" in capsys.readouterr().err
