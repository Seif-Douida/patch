"""The workflow graph (plan Task 10, design D4): scope, core evidence, rules, the model's tool
calls, draft, verify, summary, finalize; on the fixture bundle with a fake Qdrant, a fake Steam
and scripted model answers through the router and its record/replay cache (tests/support)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from agent_fixtures import GAME, OTHER_GAME
from agent_world import (
    ANSWERS,
    CHAIN,
    WINDOW,
    Down,
    Scripted,
    World,
    report_of,
    scope_answer,
)

from patchpulse.agent.graph import MAX_MODEL_TOOL_CALLS, Refusal, ReportRequest
from patchpulse.agent.prompts import PROMPT_VERSION


def test_a_nightly_report_runs_every_step_without_a_scope_call(tmp_path: Path) -> None:
    model = Scripted()

    report = report_of(World(tmp_path, model).run(ReportRequest.for_bundle("p1", 7)))

    assert model.steps == ["gather", "draft", "summary"]
    assert (report.scope, report.bundle_id, report.horizon, report.question) == (
        "bundle",
        "p1",
        7,
        None,
    )
    assert (report.window.start, report.window.end) == WINDOW
    assert [h.code for h in report.hypotheses] == ["H1", "H2", "H3", "H4", "H5"]
    assert not report.basic
    verified = {c.claim_id: c for c in report.claims}
    assert set(verified) == {"c1", "c2", "c3"}
    assert verified["c2"].reworded  # the 7-day estimate has a volume spike, no verdict yet
    assert [(w.claim.claim_id, w.rule) for w in report.withheld_claims] == [("c4", "V2")]
    assert report.summary.startswith("Performance mentions")
    assert report.confidence == "medium"


def test_an_ask_report_scopes_then_runs(tmp_path: Path) -> None:
    model = Scripted()

    report = report_of(
        World(tmp_path, model).run(
            ReportRequest.for_question("Why did Helldivers 2 reviews drop after patch 1.1.0?")
        )
    )

    assert model.steps == ["scope", "gather", "draft", "summary"]
    assert (report.scope, report.bundle_id, report.horizon) == ("question", "p1", 7)
    assert report.question == "Why did Helldivers 2 reviews drop after patch 1.1.0?"
    assert (report.window.start, report.window.end) == WINDOW


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        ({"refusal": "That isn't about a game's reviews."}, "That isn't about a game's reviews."),
        ({"appid": 1086940, "start": "2026-09-10", "end": "2026-09-16"}, "isn't one of the games"),
        ({"appid": GAME, "start": "2026-11-01", "end": "2026-11-07"}, "outside the data"),
        ({"appid": GAME, "start": "2026-09-16", "end": "2026-09-10"}, "ends before it starts"),
        ({"appid": GAME, "start": "2026-09-10", "end": "2026-09-16", "patch_id": "zz"}, "patch"),
        ({"appid": GAME}, "dates"),
    ],
)
def test_a_question_the_scope_cant_place_is_refused(
    tmp_path: Path, answer: dict[str, Any], reason: str
) -> None:
    model = Scripted({**ANSWERS, "scope": lambda _: answer})

    result = World(tmp_path, model).run(ReportRequest.for_question("Why did reviews drop?"))

    assert isinstance(result, Refusal)
    assert reason in result.reason
    assert model.steps == ["scope"]  # at most the scope call spent


def test_an_empty_or_long_question_is_refused_without_a_model_call(tmp_path: Path) -> None:
    model = Scripted()
    world = World(tmp_path, model)

    for question in ("", "   ", "x" * 1000):
        assert isinstance(world.run(ReportRequest.for_question(question)), Refusal)
    assert model.steps == []


def test_a_picked_game_is_the_only_one_the_scope_sees(tmp_path: Path) -> None:
    # The Ask page's game picker: the scope call lists that game and its patches only.
    seen: list[str] = []

    def scope(user: str) -> dict[str, Any]:
        seen.append(user)
        return scope_answer(user)

    model = Scripted({**ANSWERS, "scope": scope})

    report = report_of(
        World(tmp_path, model).run(
            ReportRequest.for_question("Why did reviews drop after 1.1.0?", appid=GAME)
        )
    )

    assert report.appid == GAME
    (user,) = seen
    assert "Helldivers 2" in user
    assert "Cities: Skylines II" not in user


def test_a_scope_on_another_game_than_the_picked_one_is_refused(tmp_path: Path) -> None:
    other = {"appid": OTHER_GAME, "start": "2026-10-01", "end": "2026-10-01"}
    model = Scripted({**ANSWERS, "scope": lambda _: other})

    result = World(tmp_path, model).run(
        ReportRequest.for_question("How is Cities: Skylines II doing?", appid=GAME)
    )

    assert isinstance(result, Refusal)
    assert "picked" in result.reason
    assert model.steps == ["scope"]


def test_an_untracked_picked_game_is_refused_without_a_model_call(tmp_path: Path) -> None:
    model = Scripted()

    result = World(tmp_path, model).run(ReportRequest.for_question("Why?", appid=1086940))

    assert isinstance(result, Refusal)
    assert "isn't one of the games" in result.reason
    assert model.steps == []


def test_the_model_gets_at_most_eight_tool_calls(tmp_path: Path) -> None:
    world = World(tmp_path, Scripted())

    world.run(ReportRequest.for_bundle("p1", 7))

    assert MAX_MODEL_TOOL_CALLS == 8
    assert world.steam_requests == 1  # the patch notes, asked for first
    assert world.qdrant.calls == MAX_MODEL_TOOL_CALLS - 1  # 11 searches asked for, 7 run


def test_all_models_down_gives_a_basic_report(tmp_path: Path) -> None:
    down = Down()

    report = report_of(World(tmp_path, down).run(ReportRequest.for_bundle("p1", 7)))

    assert report.basic
    assert down.calls == 2  # gather, on both models; nothing more is asked of them
    assert report.model_ids == {}
    assert report.claims
    assert all(c.claim_id.startswith("b") for c in report.claims)
    assert any(c.value is not None for c in report.claims)
    assert [h.code for h in report.hypotheses] == ["H1", "H2", "H3", "H4", "H5"]
    assert "Supported" in report.summary or "Ruled out" in report.summary
    assert report.withheld_claims == []


def test_a_question_with_every_model_down_is_refused(tmp_path: Path) -> None:
    result = World(tmp_path, Down()).run(ReportRequest.for_question("Why did reviews drop?"))

    assert isinstance(result, Refusal)
    assert "unavailable" in result.reason


def test_report_records_versions_and_pass_rates(tmp_path: Path) -> None:
    report = report_of(World(tmp_path, Scripted()).run(ReportRequest.for_bundle("p1", 7)))

    assert report.prompt_version == PROMPT_VERSION
    assert report.data_version == 52
    assert report.model_ids == {"gather": CHAIN[0], "draft": CHAIN[0], "summary": CHAIN[0]}
    assert report.report_id == "p1-7d"
    for rule in ("V1", "V2", "V3", "V4", "V5", "V6", "V7"):
        assert rule in report.rule_pass_rates, rule
    assert report.rule_pass_rates["V2"] == pytest.approx(2 / 3)  # c1, c2 pass; c4's 40 doesn't
    assert report.rule_pass_rates["V6"] == 1.0


def test_a_failed_summary_gets_the_template(tmp_path: Path) -> None:
    model = Scripted({**ANSWERS, "summary": lambda _: {"summary": "Reviews fell 77 points."}})

    report = report_of(World(tmp_path, model).run(ReportRequest.for_bundle("p1", 7)))

    assert "77" not in report.summary
    assert report.rule_pass_rates["V6"] == 0.0
    assert not report.basic


def test_a_recorded_run_replays_identically(tmp_path: Path) -> None:
    recorded = report_of(
        World(tmp_path, Scripted(), mode="record").run(ReportRequest.for_bundle("p1", 7))
    )

    replayed = report_of(
        World(tmp_path, Down(), mode="replay").run(ReportRequest.for_bundle("p1", 7))
    )

    assert replayed == recorded
