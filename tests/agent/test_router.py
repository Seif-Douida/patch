"""The model router (plan Task 8, design D8): every model call goes through the chain and the
record/replay cache; a broken answer, an outage or a timeout moves to the next model at once, a
daily quota retires a model until midnight Pacific, and three failures in a row skip a model for
ten minutes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest
from fake_clock import FakeClock

from patchpulse.agent.llm_cache import ReplayCache, ReplayMissError
from patchpulse.agent.router import AllModelsDownError, ModelRouter, Routed, make_agent_clients
from patchpulse.labeling.limits import QuotaExhaustedError
from patchpulse.labeling.llm import LLMResponse, TeacherRequestError, TeacherUnavailableError
from patchpulse.labeling.teachers import ModelConfig, TeacherModels

FIRST, SECOND = "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"
CHAIN = [FIRST, SECOND]
START = datetime(2026, 10, 10, 10, 0, tzinfo=UTC)  # 03:00 in Los Angeles
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["yes", "no"]},
        "score": {"type": "number"},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict"],
}
YES = '{"verdict": "yes", "score": 0.5}'
NO = '{"verdict": "no"}'


def answer(text: str, finish: str = "STOP") -> LLMResponse:
    return LLMResponse(text=text, input_tokens=100, output_tokens=20, finish_reason=finish)


class FakeClient:
    """Answers (or raises) in order; counts its calls."""

    def __init__(self, *answers: LLMResponse | Exception) -> None:
        self.answers = list(answers)
        self.calls = 0

    def generate(
        self,
        system: str,
        user: str,
        *,
        json_schema: Any,
        expected_output_tokens: int = 512,
        max_output_tokens: int = 4096,
    ) -> LLMResponse:
        self.calls += 1
        item = self.answers.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def router(
    tmp_path: Path, clients: dict[str, Any], *, mode: str = "live", clock: FakeClock | None = None
) -> ModelRouter:
    return ModelRouter(
        CHAIN,
        clients,
        ReplayCache(tmp_path / "answers.sqlite", mode),  # type: ignore[arg-type]
        clock=clock or FakeClock(START),
    )


def down() -> TeacherUnavailableError:
    return TeacherUnavailableError(503, "The model is overloaded.")


@pytest.mark.parametrize(
    "broken",
    [
        answer("Sure! The verdict is yes."),  # not JSON
        answer('{"verdict": "ye', finish="MAX_TOKENS"),  # cut off at the output limit
        answer(YES, finish="MAX_TOKENS"),  # complete-looking, but the model was cut off
        answer('{"verdict": "maybe"}'),  # outside the enum
        answer('{"score": 1}'),  # a required field missing
        answer('{"verdict": "yes", "summary": "extra"}'),  # a field the schema doesn't have
        answer('{"verdict": "yes", "score": "high"}'),  # the wrong type
        answer('["yes"]'),  # not an object
    ],
)
def test_a_broken_answer_falls_over_to_the_next_model(tmp_path: Path, broken: LLMResponse) -> None:
    first, second = FakeClient(broken), FakeClient(answer(NO))

    routed = router(tmp_path, {FIRST: first, SECOND: second}).generate(
        "draft", "system", "user", SCHEMA
    )

    assert (routed.model_id, routed.data, routed.cached) == (SECOND, {"verdict": "no"}, False)
    assert (first.calls, second.calls) == (1, 1)


@pytest.mark.parametrize(
    "fenced",
    [f"```json\n{YES}\n```", f"```\n{YES}\n```", f"{YES}\n```", f"  {YES}  "],
)
def test_a_fenced_answer_is_accepted_after_stripping(tmp_path: Path, fenced: str) -> None:
    second = FakeClient()

    routed = router(tmp_path, {FIRST: FakeClient(answer(fenced)), SECOND: second}).generate(
        "draft", "system", "user", SCHEMA
    )

    assert (routed.model_id, routed.data) == (FIRST, {"verdict": "yes", "score": 0.5})
    assert (routed.input_tokens, routed.output_tokens) == (100, 20)
    assert second.calls == 0


# The real Gemini client as the router builds it: no retries for outages and timeouts.


def ok_response(text: str = NO) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 5},
        },
    )


def quota_429(quota_id: str, retry_delay: str) -> httpx2.Response:
    return httpx2.Response(
        429,
        json={
            "error": {
                "code": 429,
                "message": "You exceeded your current quota.",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{"quotaId": quota_id}],
                    },
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": retry_delay,
                    },
                ],
            }
        },
    )


PER_MINUTE = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
MODELS = TeacherModels.model_validate(
    {
        "candidates": {
            name: {"id": name, "rpm": 15, "tpm": 250_000, "rpd": 500, "json_mode": True,
                   "system_instruction": True, "use_fraction": 1.0}
            for name in CHAIN
        },
        "agent": {"chain": CHAIN},
    }
)  # fmt: skip


class Google:
    """generateContent answers per model, in order; an exception is raised as the transport's."""

    def __init__(self, answers: dict[str, list[httpx2.Response | Exception]]) -> None:
        self.answers = answers
        self.seen: dict[str, int] = {name: 0 for name in answers}

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        name = str(request.url).split("/models/")[1].split(":")[0]
        self.seen[name] += 1
        item = self.answers[name].pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def google_router(tmp_path: Path, google: Google) -> tuple[ModelRouter, FakeClock]:
    clock = FakeClock(START)
    clients = make_agent_clients(
        MODELS,
        "AIzaTEST-not-a-real-key",
        httpx2.Client(transport=httpx2.MockTransport(google)),
        state_path=tmp_path / "usage.json",
        clock=clock,
        sleep=clock.sleep,
    )
    return router(tmp_path, dict(clients), clock=clock), clock


def test_5xx_and_timeouts_fail_over_without_waiting(tmp_path: Path) -> None:
    overloaded = httpx2.Response(503, json={"error": {"code": 503, "message": "overloaded"}})
    timeout = httpx2.ReadTimeout("timed out")
    google = Google({FIRST: [overloaded, timeout], SECOND: [ok_response(), ok_response()]})
    models, clock = google_router(tmp_path, google)

    first = models.generate("draft", "system", "user", SCHEMA)
    second = models.generate("draft", "system", "user", SCHEMA)

    assert [first.model_id, second.model_id] == [SECOND, SECOND]
    assert google.seen == {FIRST: 2, SECOND: 2}  # one try each, no retry
    assert clock.slept == []


def test_per_minute_429_waits_once(tmp_path: Path) -> None:
    google = Google(
        {
            FIRST: [quota_429(PER_MINUTE, "7s"), ok_response(YES)],
            SECOND: [],
        }
    )
    models, clock = google_router(tmp_path, google)

    routed = models.generate("draft", "system", "user", SCHEMA)

    assert routed.model_id == FIRST
    assert clock.slept == [7.0]
    twice = Google(
        {FIRST: [quota_429(PER_MINUTE, "7s"), quota_429(PER_MINUTE, "7s")], SECOND: [ok_response()]}
    )
    models, clock = google_router(tmp_path / "again", twice)
    assert models.generate("draft", "system", "user", SCHEMA).model_id == SECOND
    assert twice.seen == {FIRST: 2, SECOND: 1}
    assert clock.slept == [7.0]


def test_daily_quota_retires_the_model_until_pacific_midnight(tmp_path: Path) -> None:
    clock = FakeClock(START)
    first = FakeClient(QuotaExhaustedError("daily quota"), answer(YES))
    second = FakeClient(answer(NO), answer(NO))
    models = router(tmp_path, {FIRST: first, SECOND: second}, clock=clock)

    assert models.generate("draft", "s", "u", SCHEMA).model_id == SECOND
    clock.now = datetime(2026, 10, 11, 6, 59, tzinfo=UTC)  # 23:59 in Los Angeles
    assert models.generate("draft", "s", "u", SCHEMA).model_id == SECOND
    assert first.calls == 1
    clock.now = datetime(2026, 10, 11, 7, 0, 1, tzinfo=UTC)  # past midnight there
    assert models.generate("draft", "s", "u", SCHEMA).model_id == FIRST


def test_quota_spent_only_when_every_model_rests_for_the_day(tmp_path: Path) -> None:
    # Phase 5 Task 14: the past-patch command stops when the day's quota is used up, not when a
    # model's breaker opens for a few minutes.
    clock = FakeClock(START)
    first = FakeClient(QuotaExhaustedError("daily quota"))
    second = FakeClient(down(), QuotaExhaustedError("daily quota"))
    models = router(tmp_path, {FIRST: first, SECOND: second}, clock=clock)
    assert not models.quota_spent()

    with pytest.raises(AllModelsDownError):
        models.generate("draft", "s", "u", SCHEMA)
    assert not models.quota_spent()  # the second model is only failing
    with pytest.raises(AllModelsDownError):
        models.generate("draft", "s", "u", SCHEMA)
    assert models.quota_spent()
    clock.now = datetime(2026, 10, 11, 7, 0, 1, tzinfo=UTC)  # past midnight in Los Angeles
    assert not models.quota_spent()


def test_breaker_skips_a_model_for_ten_minutes(tmp_path: Path) -> None:
    clock = FakeClock(START)
    first = FakeClient(down(), TeacherRequestError(400, "bad"), answer("nope"), answer(YES))
    second = FakeClient(*[answer(NO)] * 4)
    models = router(tmp_path, {FIRST: first, SECOND: second}, clock=clock)

    for _ in range(3):  # three failures in a row: an outage, a refusal, a broken answer
        assert models.generate("draft", "s", "u", SCHEMA).model_id == SECOND
    clock.now += timedelta(minutes=9, seconds=59)
    assert models.generate("draft", "s", "u", SCHEMA).model_id == SECOND
    assert first.calls == 3  # skipped while the breaker is open
    clock.now += timedelta(seconds=2)
    assert models.generate("draft", "s", "u", SCHEMA).model_id == FIRST


def test_a_success_resets_the_failure_count(tmp_path: Path) -> None:
    first = FakeClient(down(), down(), answer(YES), down(), down(), answer(YES))
    second = FakeClient(*[answer(NO)] * 4)
    models = router(tmp_path, {FIRST: first, SECOND: second})

    used = [models.generate("draft", "s", "u", SCHEMA).model_id for _ in range(6)]

    assert used == [SECOND, SECOND, FIRST, SECOND, SECOND, FIRST]


def test_all_models_down_raises(tmp_path: Path) -> None:
    models = router(tmp_path, {FIRST: FakeClient(down()), SECOND: FakeClient(answer("not json"))})

    with pytest.raises(AllModelsDownError) as raised:
        models.generate("summary", "system text", "user text with a review", SCHEMA)

    message = str(raised.value)
    assert "summary" in message
    assert FIRST in message
    assert SECOND in message
    assert "review" not in message  # no prompt text in errors or logs


def test_replay_answers_from_the_cache_without_a_client(tmp_path: Path) -> None:
    recorder = router(
        tmp_path, {FIRST: FakeClient(answer(YES)), SECOND: FakeClient()}, mode="record"
    )
    recorded = recorder.generate("draft", "system", "user", SCHEMA)

    replayer = router(tmp_path, {}, mode="replay")
    replayed = replayer.generate("draft", "system", "user", SCHEMA)

    assert replayed == Routed(FIRST, recorded.data, 100, 20, cached=True)
    with pytest.raises(ReplayMissError):
        replayer.generate("draft", "system", "another user prompt", SCHEMA)


def test_record_then_replay_is_identical(tmp_path: Path) -> None:
    clients = {FIRST: FakeClient(answer("broken")), SECOND: FakeClient(answer(NO))}
    recorded = router(tmp_path, clients, mode="record").generate("draft", "s", "u", SCHEMA)

    replayed = router(tmp_path, {}, mode="replay").generate("draft", "s", "u", SCHEMA)

    assert recorded.cached is False
    assert (replayed.model_id, replayed.data, replayed.input_tokens, replayed.output_tokens) == (
        recorded.model_id,
        recorded.data,
        recorded.input_tokens,
        recorded.output_tokens,
    )
    assert replayed.model_id == SECOND


def test_live_mode_neither_reads_nor_writes(tmp_path: Path) -> None:
    router(
        tmp_path, {FIRST: FakeClient(answer(YES)), SECOND: FakeClient()}, mode="record"
    ).generate("draft", "s", "u", SCHEMA)
    first = FakeClient(answer(YES))

    router(tmp_path, {FIRST: first, SECOND: FakeClient()}, mode="live").generate(
        "draft", "s", "u", SCHEMA
    )

    assert first.calls == 1  # asked again, though the cache holds the answer


def test_the_agent_chain_must_name_gemini_candidates() -> None:
    with pytest.raises(ValueError, match="not a candidate"):
        TeacherModels.model_validate(
            {
                "candidates": {FIRST: MODELS.candidates[FIRST].model_dump()},
                "agent": {"chain": [SECOND]},
            }
        )
    server = ModelConfig(
        provider="openai_compat", id="gemma", json_mode=True, system_instruction=True
    )
    with pytest.raises(ValueError, match="Gemini"):
        TeacherModels.model_validate(
            {"candidates": {"server": server.model_dump()}, "agent": {"chain": ["server"]}}
        )
