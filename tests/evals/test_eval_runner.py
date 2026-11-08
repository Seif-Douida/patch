"""The runner (Phase 6 plan Task 4): an arm runs every case once as a visitor's question, records
it under data/local, stops cleanly at the day's quota without saving the interrupted case,
resumes from the cache, and names what a replay couldn't find."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_fixtures import GAME
from agent_world import CHAIN, Scripted, World, scope_answer
from fake_clock import FakeClock

from patchpulse.agent.cli import QUOTA_SPENT
from patchpulse.agent.llm_cache import ReplayCache
from patchpulse.agent.tools import ExternalCalls
from patchpulse.evals.arms import ARMS, Arm
from patchpulse.evals.records import load_arm
from patchpulse.evals.runner import Case, make_router, run_arm
from patchpulse.labeling.limits import QuotaExhaustedError
from patchpulse.labeling.llm import LLMResponse

START = datetime(2026, 11, 15, 10, 0, tzinfo=UTC)
CASES = [
    Case("case-a", GAME, "Why did Helldivers 2 reviews drop after patch 1.1.0?"),
    Case("case-b", GAME, "What changed in Helldivers 2's reviews in mid September 2026?"),
]
QUIET = {
    "scope": scope_answer,
    "gather": lambda _: {"calls": []},
    "draft": lambda _: {"findings": [], "claims": []},
    "summary": lambda _: {"summary": "Nothing stands out."},
}


class Budget:
    """A scripted model with a daily quota of `calls` answers; remembers each prompt it answered."""

    def __init__(self, calls: int) -> None:
        self.inner = Scripted()
        self.left = calls
        self.answered: set[str] = set()

    def generate(self, system: str, user: str, **kwargs: Any) -> LLMResponse:
        if self.left <= 0:
            raise QuotaExhaustedError("gemini-3.5-flash-lite's daily quota is used up")
        self.left -= 1
        self.answered.add(hashlib.sha256(f"{system}\n{user}".encode()).hexdigest())
        return self.inner.generate(system, user, **kwargs)


class Spy:
    """A recording hook around the cache that remembers each call's arguments."""

    def __init__(self, inner: ExternalCalls, seen: list[dict[str, Any]]) -> None:
        self.inner = inner
        self.seen = seen

    def external(self, tool: str, args: Mapping[str, Any], data_version: int, live: Any) -> Any:
        self.seen.append(dict(args))
        return self.inner.external(tool, args, data_version, live)


def root(tmp_path: Path, name: str = "evals") -> Path:
    return tmp_path / "data" / "local" / name


def run(
    world: World,
    arm: Arm,
    model: Any,
    cache: ReplayCache,
    where: Path,
    cases: list[Case] = CASES,
) -> Any:
    router = make_router(arm, dict.fromkeys(CHAIN, model), cache, clock=FakeClock(START))
    return run_arm(
        arm,
        cases,
        bundle=world.deps.bundle,
        router=router,
        cache=cache,
        search=world.deps.search,
        steam=world.deps.steam,
        root=where,
    )


def test_an_arm_records_every_case_once(tmp_path: Path) -> None:
    world = World(tmp_path, Scripted())
    cache = ReplayCache(tmp_path / "evals.sqlite", "fill")

    first = run(world, ARMS["main"], Scripted(), cache, root(tmp_path))
    second = run(world, ARMS["main"], Scripted(), cache, root(tmp_path))

    assert (first.recorded, first.already, first.missed, first.stopped) == (2, 0, {}, None)
    assert (second.recorded, second.already) == (0, 2)
    assert set(load_arm("main", root=root(tmp_path))) == {"case-a", "case-b"}


def test_a_run_stopped_by_the_quota_resumes_from_the_cache(tmp_path: Path) -> None:
    world = World(tmp_path, Scripted())
    cache = ReplayCache(tmp_path / "evals.sqlite", "fill")
    first_model = Budget(6)  # case-a's 4 answers, then case-b's scope and gather

    first = run(world, ARMS["main"], first_model, cache, root(tmp_path))

    assert (first.recorded, first.stopped) == (1, QUOTA_SPENT)
    assert set(load_arm("main", root=root(tmp_path))) == {"case-a"}

    second_model = Budget(100)
    second = run(world, ARMS["main"], second_model, cache, root(tmp_path))

    stored = load_arm("main", root=root(tmp_path))
    assert (second.recorded, second.already, second.stopped) == (1, 1, None)
    assert second_model.answered.isdisjoint(first_model.answered)
    # Only case-b's draft: its summary prompt (claims and statuses, no question) is case-a's.
    assert second_model.inner.steps == ["draft"]
    assert stored["case-b"].report is not None
    assert not stored["case-b"].report.basic


def test_misses_are_recorded_with_the_case(tmp_path: Path) -> None:
    world = World(tmp_path, Scripted())
    path = tmp_path / "evals.sqlite"
    run(world, ARMS["main"], Scripted(), ReplayCache(path, "fill"), root(tmp_path), CASES[:1])
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM tool_results")

    replayed = run(
        world, ARMS["main"], Scripted(), ReplayCache(path, "replay"), root(tmp_path, "again"),
        CASES[:1],
    )  # fmt: skip

    stored = load_arm("main", root=root(tmp_path, "again"))["case-a"]
    assert "tool:search_reviews" in stored.misses
    assert replayed.missed == {"case-a": stored.misses}


def test_arms_differ_only_in_what_they_say(tmp_path: Path) -> None:
    world = World(tmp_path, Scripted())
    cache = ReplayCache(tmp_path / "evals.sqlite", "fill")
    seen: list[dict[str, Any]] = []
    spied = [Case("case-a", GAME, CASES[0].question, wrap=lambda inner: Spy(inner, seen))]

    run(world, ARMS["a3"], Scripted(QUIET), cache, root(tmp_path), CASES[:1])
    run(world, ARMS["a2"], Scripted(), cache, root(tmp_path), spied)
    run(world, ARMS["a4"], Scripted(), cache, root(tmp_path), CASES[:1])

    a3 = load_arm("a3", root=root(tmp_path))["case-a"]
    a4 = load_arm("a4", root=root(tmp_path))["case-a"]
    assert all(c.tool != "estimate_patch_effect" for c in a3.trace.calls)
    assert {args["mode"] for args in seen if "query" in args} == {"dense"}
    assert a4.report is not None
    assert set(a4.report.model_ids.values()) == {"gemini-3.1-flash-lite"}
    assert ARMS["main"].chain == ("gemini-3.5-flash-lite",)
    assert ARMS["a4"].chain == ("gemini-3.1-flash-lite",)


def test_the_indexed_review_count_reads_the_agents_filter(tmp_path: Path) -> None:
    world = World(tmp_path, Scripted())
    counted: list[dict[str, Any]] = []

    def count(**kwargs: Any) -> Any:
        counted.append(kwargs)
        return type("Count", (), {"count": 1234})()

    world.qdrant.count = count  # type: ignore[attr-defined]
    cache = ReplayCache(tmp_path / "evals.sqlite", "fill")

    run(world, ARMS["main"], Scripted(), cache, root(tmp_path), CASES[:1])

    assert load_arm("main", root=root(tmp_path))["case-a"].indexed_reviews == 1234
    assert counted[0]["collection_name"] == "reviews"
    assert counted[0]["exact"] is True
