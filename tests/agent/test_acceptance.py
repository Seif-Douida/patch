"""The acceptance run (Phase 5 plan Task 21, design D15): 10 bundles picked by a fixed rule, a
templated question each, then a forced basic report, and the coverage table. A fixture bundle with
twelve estimated bundles and scripted models; no database."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from agent_fixtures import GAME, estimate, game_data, make_bundle
from agent_world import Scripted, World
from pydantic import SecretStr

from patchpulse.agent import cli
from patchpulse.agent.acceptance import (
    pick_bundles,
    run_acceptance,
    write_acceptance,
)
from patchpulse.agent.bundle import Bundle, PatchRow, dump_bundle
from patchpulse.agent.graph import ReportRequest
from patchpulse.agent.report import Report
from patchpulse.agent.tracing import Observer
from patchpulse.config import LabSettings

# Each bundle's 14-day effect; q12's 7-day effect is the largest of all, so a rule reading the
# 7-day rows would pick it first. q2's estimate is confounded by a sale.
EFFECTS = {
    "q1": -15.0, "q2": 9.0, "q3": -0.5, "q4": 4.0, "q5": -2.0, "q6": 0.25,
    "q7": 12.0, "q8": -7.0, "q9": 1.0, "q10": -3.0, "q11": 6.0, "q12": -0.1,
}  # fmt: skip
FIRST_PATCH = date(2026, 8, 10)


def acceptance_bundle() -> Bundle:
    patches, estimates = [], []
    for i, (bundle_id, effect) in enumerate(EFFECTS.items()):
        day = FIRST_PATCH + timedelta(days=3 * i)
        published = datetime(day.year, day.month, day.day, 16, tzinfo=UTC)
        patches.append(PatchRow(bundle_id, bundle_id, published, day, f"Update {i + 1}",
                                "patch", bundle_id))  # fmt: skip
        flags = frozenset({"confounded"}) if bundle_id == "q2" else frozenset()
        for horizon, value in ((14, effect), (7, 30.0 if bundle_id == "q12" else effect)):
            row = estimate(horizon, "synth", value, flags=flags)
            estimates.append(replace(row, bundle_id=bundle_id, patch_ids=(bundle_id,)))
    # Not picked: a bundle whose estimate has no effect, and one with an aspect estimate only.
    day = date(2026, 9, 20)
    for bundle_id, row in (
        ("q13", estimate(14, "synth", None, reason="too few comparison games")),
        ("q14", estimate(14, "its", -40.0, outcome="aspect:performance")),
    ):
        patches.append(PatchRow(bundle_id, bundle_id, datetime(2026, 9, 20, 16, tzinfo=UTC),
                                day, "Hotfix", "hotfix", bundle_id))  # fmt: skip
        estimates.append(replace(row, bundle_id=bundle_id, patch_ids=(bundle_id,)))
    game = replace(game_data(), patches=tuple(patches), estimates=tuple(estimates))
    return make_bundle(game)


def scope_answer(user: str) -> dict[str, Any]:
    """The patch the question names, and the 14 days from its day."""
    (title,) = re.findall(r'What changed after "([^"]+)"', user)
    found = re.search(rf'- (\S+): Helldivers 2, (\S+), "{title}"', user)
    assert found, f"{title} isn't in the scope prompt"
    start = date.fromisoformat(found.group(2))
    end = start + timedelta(days=13)
    return {"appid": GAME, "start": str(start), "end": str(end), "patch_id": found.group(1)}


def draft_answer(user: str) -> dict[str, Any]:
    """A causal claim stating the estimate, and a number the estimate doesn't hold (V2)."""
    found = re.search(r"\[(\S+)\] estimate_patch_effect \{.*?\} -> (\{.*\})", user)
    assert found, "no estimate in the draft prompt"
    effect = json.loads(found.group(2))["horizons"]["14"]["effect"]
    cited = {"call_id": found.group(1), "json_path": "horizons.14.effect"}
    return {
        "findings": [{"title": "The patch", "hypothesis": "H1", "claim_ids": ["c1", "c2"]}],
        "claims": [
            {"claim_id": "c1", "text": "The patch moved the positive share.",
             "claim_type": "causal", "value": effect, "unit": "pp", "evidence": cited},
            {"claim_id": "c2", "text": "The positive share moved by 99 points.",
             "claim_type": "comparative", "value": 99.0, "unit": "pp", "evidence": cited},
        ],
    }  # fmt: skip


ANSWERS = {
    "scope": scope_answer,
    "gather": lambda _: {"calls": []},
    "draft": draft_answer,
    "summary": lambda _: {"summary": "Reviews moved after the update."},
}


class Numbered(Observer):
    def __init__(self, trace_id: str) -> None:
        self._trace_id = trace_id

    @property
    def trace_id(self) -> str | None:
        return self._trace_id


class Traces:
    """A tracer whose runs are traced as t1, t2, ..."""

    def __init__(self) -> None:
        self.started: list[ReportRequest] = []

    def start(self, request: ReportRequest, *, data_version: int) -> Observer:
        self.started.append(request)
        return Numbered(f"t{len(self.started)}")

    def close(self) -> None:
        return None


def table(text: str, heading: str) -> list[str]:
    """The table rows (header and rule excluded) under a markdown heading."""
    section = text.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("|")]
    return rows[2:]


def test_acceptance_picks_bundles_by_the_rule_and_prints_coverage(tmp_path: Path) -> None:
    model, traces = Scripted(ANSWERS), Traces()
    world = World(tmp_path, model, tracer=traces)
    bundle = acceptance_bundle()
    deps = replace(world.deps, bundle=bundle)

    picks = pick_bundles(bundle)
    runs = run_acceptance(deps, picks)
    text = write_acceptance(tmp_path / "out", bundle.data_version, runs)

    # The 5 largest |effect| (14-day served rows), then the 5 smallest.
    assert [p.bundle_id for p in picks] == [
        "q1", "q7", "q2", "q8", "q11", "q12", "q6", "q3", "q9", "q5",
    ]  # fmt: skip
    assert (picks[0].horizon, picks[0].effect) == (14, -15.0)
    assert picks[0].question == 'What changed after "Update 1" (10 Aug 2026)?'
    # Each pick asked as a visitor would, on its game; then the first pick's nightly report with
    # the chain emptied, which the models never see.
    assert traces.started[:10] == [
        ReportRequest.for_question(p.question, appid=GAME) for p in picks
    ]
    assert traces.started[10] == ReportRequest.for_bundle("q1", 14)
    assert model.steps.count("scope") == 10
    assert len(model.steps) == 40
    basic = runs[-1].result
    assert isinstance(basic, Report)
    assert basic.basic

    rows = table(text, "Runs")
    assert len(rows) == 11
    assert rows[0].startswith(
        '| 1 | Helldivers 2 | q1 | -15.0 | What changed after "Update 1" (10 Aug 2026)? | q-'
    )
    assert rows[0].endswith("| 1 | 1 (V2) | t1 |")
    assert rows[10].startswith("| basic | Helldivers 2 | q1 | -15.0 | the nightly report, ")
    assert rows[10].endswith("| t11 |")

    assert table(text, "Explanations")[0] == (
        "| H1 The patch | 4 | 0 | 6 | H1.wide_range 5, H1.gate_allows 4, H1.blocked.confounded 1 |"
    )
    verifier = table(text, "The verifier")
    assert verifier[1].startswith("| V2 | 10 |")
    assert verifier[3].startswith("| V4 | 6 |")
    assert verifier[5].startswith("| V6 | 0 |")
    assert "Allowed by the gate: 4." in text
    v4 = dict(row.split(" | ", 1) for row in table(text, "V4: causal claims"))
    assert v4["| range_includes_zero"] == "5 | 0 |"
    assert v4["| confounded"] == "1 | 0 |"
    assert v4["| volume_spike"] == "0 | 0 |"
    assert "Forced with the router's chain emptied: a basic report (q1-14d)." in text

    never = table(text, "Never seen") or text.split("## Never seen\n", 1)[1].splitlines()
    assert "- H1 ruled out" in never
    assert "- V1" in never
    assert "- V4 volume_spike" in never
    assert "- V2" not in never
    assert "- V4 confounded" not in never
    assert "- basic report" not in never

    out = tmp_path / "out"
    assert (out / "coverage.md").read_text(encoding="utf-8") == text
    copies = sorted(p.name for p in (out / "reports").iterdir())
    assert len(copies) == 11
    assert "q1-14d.json" in copies


KEYS: dict[str, Any] = {
    "gemini_api_key": SecretStr("hunter2-gemini"),
    "langfuse_public_key": "pk-lf-test",
    "langfuse_secret_key": SecretStr("hunter2-hunter2"),
}


def test_the_command_runs_on_a_local_bundle_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bundle.json.gz"
    path.write_bytes(dump_bundle(acceptance_bundle()))
    world = World(tmp_path, Scripted(ANSWERS), tracer=Traces())
    samples: list[float] = []

    def live(stack: Any, settings: Any, encoder_dir: Path, *, ask_sample: float) -> Any:
        samples.append(ask_sample)
        return lambda bundle: replace(world.deps, bundle=bundle)

    monkeypatch.setattr(cli, "get_lab_settings", lambda: LabSettings(**KEYS))
    monkeypatch.setattr(cli, "_live_deps", live)
    out = tmp_path / "out"

    code = cli.main(["acceptance", "--bundle", str(path), "--out", str(out), "--add", "q10"])

    printed = capsys.readouterr().out
    assert code == 0
    assert samples == [1.0]  # every question traced
    assert printed.startswith("# Acceptance run: data version 52\n")
    assert "| 11 | Helldivers 2 | q10 | -3.0 |" in printed
    assert (out / "v52" / "coverage.md").read_text(encoding="utf-8") == printed


def test_the_command_needs_the_gemini_and_langfuse_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_agent(*args: object, **kwargs: object) -> Any:
        raise AssertionError("refusing must not build the agent")

    monkeypatch.setattr(cli, "_live_deps", no_agent)
    for missing in ("gemini_api_key", "langfuse_public_key", "langfuse_secret_key"):
        settings = LabSettings(**{k: v for k, v in KEYS.items() if k != missing})
        monkeypatch.setattr(cli, "get_lab_settings", lambda s=settings: s)

        assert cli.main(["acceptance"]) == 2
        assert "PP_LANGFUSE_SECRET_KEY" in capsys.readouterr().err


def test_acceptance_adds_bundles_chosen_to_hit_a_missing_path() -> None:
    picks = pick_bundles(acceptance_bundle(), extra=["q10", "q4", "q1"])

    assert [p.bundle_id for p in picks][10:] == ["q10", "q4"]  # q1 is already picked
    assert picks[10].question == 'What changed after "Update 10" (6 Sep 2026)?'
