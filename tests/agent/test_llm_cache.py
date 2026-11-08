"""The record/replay cache (Phase 6 plan Task 1, design D2): the fill mode, recorded tool results
and named misses. Model answers through the router; tool results through `external`."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fake_clock import FakeClock

from patchpulse.agent.llm_cache import ReplayCache, ToolNotRecordedError, tool_key
from patchpulse.agent.router import ModelRouter
from patchpulse.labeling.llm import LLMResponse

MODEL = "gemini-3.5-flash-lite"
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"verdict": {"type": "string"}},
    "required": ["verdict"],
}
HITS = [{"review_id": 7, "score": 0.5, "text": "Invented review text."}]


class Counting:
    """A model client that answers every prompt with the same JSON and counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, system: str, user: str, **_: Any) -> LLMResponse:
        self.calls += 1
        return LLMResponse('{"verdict": "yes"}', 10, 5, "STOP")


class Live:
    """A tool's live call: returns `value` (or raises it) and counts its calls."""

    def __init__(self, value: Any) -> None:
        self.value = value
        self.calls = 0

    def __call__(self) -> Any:
        self.calls += 1
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def router(path: Path, client: Counting) -> ModelRouter:
    return ModelRouter(
        [MODEL],
        {MODEL: client},
        ReplayCache(path, "fill"),
        clock=FakeClock(datetime(2026, 11, 9, 10, 0, tzinfo=UTC)),
    )


def test_fill_answers_recorded_prompts_and_records_new_ones(tmp_path: Path) -> None:
    path = tmp_path / "answers.sqlite"
    first = Counting()
    for user in ("question one", "question two"):
        router(path, first).generate("draft", "system", user, SCHEMA)

    again = Counting()
    answered = [router(path, again).generate("draft", "system", u, SCHEMA) for u in
                ("question one", "question two")]  # fmt: skip

    assert first.calls == 2
    assert again.calls == 0
    assert all(a.cached for a in answered)


def test_external_replays_a_recorded_tool_result(tmp_path: Path) -> None:
    path = tmp_path / "answers.sqlite"
    filling, live = ReplayCache(path, "fill"), Live(HITS)
    args = {"appid": 1, "query": "crashes"}
    first = filling.external("search_reviews", args, 51, live)
    second = filling.external("search_reviews", args, 51, live)

    offline = Live(RuntimeError("no network in replay"))
    replayed = ReplayCache(path, "replay").external("search_reviews", args, 51, offline)

    assert live.calls == 1
    assert first == second == replayed == HITS
    assert offline.calls == 0


def test_a_tool_miss_is_named_and_raises(tmp_path: Path) -> None:
    cache = ReplayCache(tmp_path / "answers.sqlite", "replay")

    with pytest.raises(ToolNotRecordedError):
        cache.external("search_reviews", {"appid": 1}, 51, Live(HITS))

    assert cache.misses == ["tool:search_reviews"]


def test_a_failed_live_call_is_not_recorded(tmp_path: Path) -> None:
    cache = ReplayCache(tmp_path / "answers.sqlite", "fill")
    args = {"patch_id": "p1"}
    with pytest.raises(TimeoutError):
        cache.external("get_patch_notes", args, 51, Live(TimeoutError("Steam is slow")))

    retried = Live({"found": True, "text": "Fixed crashes", "truncated": False})
    cache.external("get_patch_notes", args, 51, retried)

    assert retried.calls == 1


def test_record_refreshes_and_live_keeps_nothing(tmp_path: Path) -> None:
    path = tmp_path / "answers.sqlite"
    args = {"appid": 1}
    ReplayCache(path, "live").external("search_reviews", args, 51, Live(HITS))
    with pytest.raises(ToolNotRecordedError):
        ReplayCache(path, "replay").external("search_reviews", args, 51, Live(HITS))

    recording = Live([])
    ReplayCache(path, "record").external("search_reviews", args, 51, Live(HITS))
    ReplayCache(path, "record").external("search_reviews", args, 51, recording)

    assert recording.calls == 1  # record asks every time, and keeps the latest
    assert ReplayCache(path, "replay").external("search_reviews", args, 51, Live(HITS)) == []


def test_tool_keys_separate_data_versions_and_arguments() -> None:
    args = {"appid": 1, "query": "crashes", "window": ["2026-09-01", "2026-09-07"]}

    assert tool_key("search_reviews", args, 51) == tool_key(
        "search_reviews", dict(reversed(list(args.items()))), 51
    )
    assert tool_key("search_reviews", args, 51) != tool_key("search_reviews", args, 52)
    assert tool_key("search_reviews", args, 51) != tool_key(
        "search_reviews", {**args, "query": "stutter"}, 51
    )
    assert tool_key("search_reviews", args, 51) != tool_key("get_patch_notes", args, 51)
