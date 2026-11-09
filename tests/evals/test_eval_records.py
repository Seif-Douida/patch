"""Stored cases (Phase 6 plan Task 4): one run saved as JSON under data/local and read back into
the record the scoring reads, review text and all, so it never leaves data/local."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from agent_fixtures import GAME
from agent_world import ANSWERS, Scripted, World

from patchpulse.agent.graph import Refusal, ReportRequest, run_recorded
from patchpulse.evals.records import StoredCase, load_arm, load_case, save_case
from patchpulse.evals.scoring import predicted_category

QUESTION = "Why did Helldivers 2 reviews drop after patch 1.1.0?"


def runs(tmp_path: Path) -> Path:
    return tmp_path / "data" / "local" / "evals" / "runs" / "main"


def test_a_case_round_trips(tmp_path: Path) -> None:
    model = Scripted({**ANSWERS, "summary": lambda _: {"summary": "Reviews moved 99 points."}})
    record = run_recorded(
        ReportRequest.for_question(QUESTION, appid=GAME), World(tmp_path, model).deps
    )
    stored = StoredCase.from_record(
        record, case="c1", arm="main", data_version=52, misses=[], indexed_reviews=1234
    )

    save_case(stored, runs(tmp_path) / "c1.json")
    loaded = load_case(runs(tmp_path) / "c1.json")
    again = loaded.record()

    assert again.result == record.result
    assert again.draft == record.draft
    assert again.raw_summary == record.raw_summary == "Reviews moved 99 points."
    assert again.window == record.window
    assert again.hypotheses == record.hypotheses
    assert again.trace.retrieved == record.trace.retrieved
    assert again.trace.calls == record.trace.calls
    assert again.trace.model_calls == record.trace.model_calls
    assert predicted_category(again.result, again.trace) == predicted_category(
        record.result, record.trace
    )
    assert (loaded.case, loaded.arm, loaded.data_version, loaded.indexed_reviews) == (
        "c1",
        "main",
        52,
        1234,
    )
    assert load_arm("main", root=tmp_path / "data" / "local" / "evals") == {"c1": loaded}


def test_a_refusal_round_trips(tmp_path: Path) -> None:
    model = Scripted({"scope": lambda _: {"refusal": "That isn't about a game's reviews."}})
    record = run_recorded(ReportRequest.for_question("Weather?"), World(tmp_path, model).deps)
    stored = StoredCase.from_record(
        record, case="r1", arm="main", data_version=52, misses=[], indexed_reviews=None
    )

    save_case(stored, runs(tmp_path) / "r1.json")
    again = load_case(runs(tmp_path) / "r1.json").record()

    assert again.result == Refusal("That isn't about a game's reviews.")
    assert (again.window, again.hypotheses, again.draft) == (None, [], None)


def test_records_refuse_a_folder_outside_data_local(tmp_path: Path) -> None:
    record = run_recorded(
        ReportRequest.for_question(QUESTION, appid=GAME), World(tmp_path, Scripted()).deps
    )
    stored = StoredCase.from_record(
        record, case="c1", arm="main", data_version=52, misses=[], indexed_reviews=None
    )

    with pytest.raises(ValueError, match="data/local"):
        save_case(stored, tmp_path / "elsewhere" / "c1.json")


def test_a_run_that_no_longer_fits_fails_without_its_text(tmp_path: Path) -> None:
    """A later schema change makes old runs fail to load; the error (printed, and in CI's public
    log) names where, never the text (Final review C1)."""
    model = Scripted({**ANSWERS, "summary": lambda _: {"summary": "SENTINEL-TEXT summary"}})
    record = run_recorded(
        ReportRequest.for_question(QUESTION, appid=GAME), World(tmp_path, model).deps
    )
    stored = StoredCase.from_record(
        record, case="c1", arm="main", data_version=52, misses=[], indexed_reviews=None
    )
    document = json.loads(stored.model_dump_json())
    claim = document["report"]["claims"][0]
    claim["added_later"] = "SENTINEL-TEXT extra"  # a field this code doesn't know
    del claim["claim_type"]  # a field it needs: the error shows the claim around it
    path = runs(tmp_path) / "c1.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(document).replace("A negative review", "SENTINEL-TEXT"), "utf-8")

    with pytest.raises(ValueError, match="claim_type") as caught:
        load_case(path)

    assert "SENTINEL-TEXT" not in str(caught.value)
