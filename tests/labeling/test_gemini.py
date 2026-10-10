"""The Gemini API client (plan Task 7): request shape, 429 handling, and keeping the key secret."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
import pytest
from fake_clock import FakeClock

from patchpulse.labeling.gemini import (
    MAX_ATTEMPTS,
    GeminiClient,
    GeminiError,
    GeminiResponse,
    GeminiUnavailableError,
)
from patchpulse.labeling.limits import QuotaExhaustedError, RequestLimiter
from patchpulse.labeling.teachers import ModelConfig

KEY = "AIzaTEST-not-a-real-key-123"
START = datetime(2026, 10, 10, 10, 0, tzinfo=UTC)
SCHEMA = {"type": "object", "properties": {"labels": {"type": "array"}}}


def model(*, json_mode: bool, system_instruction: bool) -> ModelConfig:
    return ModelConfig(
        id="gemma-4-31b-it",
        rpm=30,
        tpm=16_000,
        rpd=14_400,
        json_mode=json_mode,
        system_instruction=system_instruction,
        use_fraction=0.5,
    )


def ok(text: str = '{"labels": []}', *, thought: str | None = None) -> httpx2.Response:
    parts: list[dict[str, Any]] = [{"text": thought, "thought": True}] if thought else []
    parts.append({"text": text})
    return httpx2.Response(
        200,
        json={
            "candidates": [{"content": {"parts": parts, "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 120,
                "candidatesTokenCount": 30,
                "thoughtsTokenCount": 5,
                "totalTokenCount": 155,
            },
        },
    )


def quota_429(quota_id: str, retry_delay: str | None) -> httpx2.Response:
    details: list[dict[str, Any]] = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": quota_id, "quotaValue": "30"}],
        }
    ]
    if retry_delay is not None:
        details.append(
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}
        )
    return httpx2.Response(
        429,
        json={
            "error": {
                "code": 429,
                "message": "You exceeded your current quota.",
                "status": "RESOURCE_EXHAUSTED",
                "details": details,
            }
        },
    )


def client(
    tmp_path: Path,
    responses: list[httpx2.Response],
    *,
    config: ModelConfig | None = None,
) -> tuple[GeminiClient, list[httpx2.Request], FakeClock]:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return responses.pop(0)

    clock = FakeClock(START)
    config = config or model(json_mode=True, system_instruction=True)
    limiter = RequestLimiter(
        config, state_path=tmp_path / "usage.json", clock=clock, sleep=clock.sleep
    )
    gemini = GeminiClient(
        KEY,
        httpx2.Client(transport=httpx2.MockTransport(handler)),
        limiter,
        config,
        sleep=clock.sleep,
    )
    return gemini, seen, clock


def body(request: httpx2.Request) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(request.content)
    return parsed


@pytest.mark.parametrize(("json_mode", "system_instruction"), [(True, True), (False, False)])
def test_request_shape_with_and_without_json_mode(
    tmp_path: Path, json_mode: bool, system_instruction: bool
) -> None:
    config = model(json_mode=json_mode, system_instruction=system_instruction)
    gemini, seen, _ = client(tmp_path, [ok()], config=config)

    gemini.generate("Label the reviews.", "1: great combat", json_schema=SCHEMA)

    (request,) = seen
    assert request.method == "POST"
    assert request.url.path == "/v1beta/models/gemma-4-31b-it:generateContent"
    assert request.headers["x-goog-api-key"] == KEY
    assert KEY not in str(request.url)
    sent = body(request)
    generation = sent["generationConfig"]
    assert generation["temperature"] == 0
    text = sent["contents"][0]["parts"][0]["text"]
    if json_mode:
        assert generation["responseMimeType"] == "application/json"
        assert generation["responseJsonSchema"] == SCHEMA
    else:
        assert "responseMimeType" not in generation
        assert "responseJsonSchema" not in generation
    if system_instruction:
        assert sent["systemInstruction"] == {"parts": [{"text": "Label the reviews."}]}
        assert text == "1: great combat"
    else:
        assert "systemInstruction" not in sent
        assert text == "Label the reviews.\n\n1: great combat"


def test_response_text_skips_thoughts_and_counts_every_token(tmp_path: Path) -> None:
    gemini, _, _ = client(tmp_path, [ok('{"labels": [1]}', thought="let me think")])

    response = gemini.generate("system", "user", json_schema=SCHEMA)

    assert response == GeminiResponse(
        text='{"labels": [1]}', input_tokens=120, output_tokens=35, finish_reason="STOP"
    )


def test_retry_delay_is_honoured_on_429(tmp_path: Path) -> None:
    gemini, seen, clock = client(
        tmp_path,
        [quota_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "13s"), ok()],
    )

    response = gemini.generate("system", "user", json_schema=SCHEMA)

    assert response.text == '{"labels": []}'
    assert len(seen) == 2
    assert 13.0 in clock.slept


def test_daily_quota_raises_quota_exhausted(tmp_path: Path) -> None:
    gemini, seen, _ = client(
        tmp_path, [quota_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "30s")]
    )

    with pytest.raises(QuotaExhaustedError, match="gemma-4-31b-it"):
        gemini.generate("system", "user", json_schema=SCHEMA)

    assert len(seen) == 1  # waiting 30 s would not help: the quota resets at Pacific midnight


def overloaded() -> httpx2.Response:
    return httpx2.Response(
        503,
        json={
            "error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}
        },
    )


def test_an_outage_that_outlasts_the_retries_is_not_a_request_error(tmp_path: Path) -> None:
    # An overloaded model isn't the batch's fault: the teacher must stop, not split the batch.
    gemini, seen, clock = client(tmp_path, [overloaded() for _ in range(MAX_ATTEMPTS)])

    with pytest.raises(GeminiUnavailableError, match="503: The model is overloaded") as raised:
        gemini.generate("system", "user", json_schema=SCHEMA)

    assert raised.value.status == 503
    assert not isinstance(raised.value, GeminiError)
    assert len(seen) == MAX_ATTEMPTS == 6
    assert sum(clock.slept) >= 4 * 60  # patient: minutes, not seconds, before giving up


def test_an_overload_that_clears_is_retried_quietly(tmp_path: Path) -> None:
    gemini, seen, _ = client(tmp_path, [overloaded(), overloaded(), ok()])

    assert gemini.generate("system", "user", json_schema=SCHEMA).text == '{"labels": []}'
    assert len(seen) == 3


def test_rate_limits_that_never_clear_are_an_outage(tmp_path: Path) -> None:
    per_minute = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
    gemini, _, _ = client(tmp_path, [quota_429(per_minute, "2s") for _ in range(MAX_ATTEMPTS)])

    with pytest.raises(GeminiUnavailableError) as raised:
        gemini.generate("system", "user", json_schema=SCHEMA)

    assert raised.value.status == 429


def test_api_key_never_appears_in_errors_logs_or_repr(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    echo = httpx2.Response(
        400,
        json={
            "error": {
                "code": 400,
                "message": f"API key {KEY} not valid.",
                "status": "INVALID_ARGUMENT",
            }
        },
    )
    responses = [quota_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "1s"), echo]
    gemini, _, _ = client(tmp_path, responses)
    caplog.set_level(logging.DEBUG)

    with pytest.raises(GeminiError) as raised:
        gemini.generate("system", "user", json_schema=SCHEMA)

    error = raised.value
    assert error.status == 400
    for shown in (str(error), repr(error), repr(gemini), caplog.text):
        assert KEY not in shown
    assert "not valid" in str(error)


def test_blocked_prompt_is_an_error_not_an_empty_answer(tmp_path: Path) -> None:
    blocked = httpx2.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})
    gemini, _, _ = client(tmp_path, [blocked])

    with pytest.raises(GeminiError, match="SAFETY"):
        gemini.generate("system", "user", json_schema=SCHEMA)
