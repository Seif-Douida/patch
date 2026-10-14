"""The client for a self-hosted, OpenAI-style server (ADR-020): request shape, failures, secrecy."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx2
import pytest
from fake_clock import FakeClock

from patchpulse.labeling.llm import (
    MAX_ATTEMPTS,
    LLMResponse,
    TeacherRequestError,
    TeacherUnavailableError,
)
from patchpulse.labeling.openai_compat import OpenAICompatClient
from patchpulse.labeling.teachers import ModelConfig

KEY = "sk-test-not-a-real-key-123"
BASE = "https://llm.example.net/v1"
SCHEMA = {"type": "object"}


def model(*, json_mode: bool = True, system_instruction: bool = True) -> ModelConfig:
    return ModelConfig(
        provider="openai_compat",
        id="gemma-31b",
        json_mode=json_mode,
        system_instruction=system_instruction,
    )


def completion(content: str) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "choices": [
                {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1500, "completion_tokens": 300, "total_tokens": 1800},
        },
    )


def client(
    responses: list[httpx2.Response], config: ModelConfig | None = None
) -> tuple[OpenAICompatClient, list[httpx2.Request], FakeClock]:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return responses.pop(0)

    clock = FakeClock(datetime(2026, 10, 12, 13, 0, tzinfo=UTC))
    server = OpenAICompatClient(
        KEY,
        BASE,
        httpx2.Client(transport=httpx2.MockTransport(handler)),
        config or model(),
        sleep=clock.sleep,
    )
    return server, seen, clock


def body(request: httpx2.Request) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(request.content)
    return parsed


@pytest.mark.parametrize(("json_mode", "system_instruction"), [(True, True), (False, False)])
def test_request_shape(json_mode: bool, system_instruction: bool) -> None:
    config = model(json_mode=json_mode, system_instruction=system_instruction)
    server, seen, _ = client([completion("{}")], config)

    server.generate("Label the reviews.", "1: great combat", json_schema=SCHEMA)

    (request,) = seen
    assert str(request.url) == "https://llm.example.net/v1/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    sent = body(request)
    assert sent["model"] == "gemma-31b"
    assert sent["temperature"] == 0
    if system_instruction:
        assert sent["messages"] == [
            {"role": "system", "content": "Label the reviews."},
            {"role": "user", "content": "1: great combat"},
        ]
    else:
        assert sent["messages"] == [
            {"role": "user", "content": "Label the reviews.\n\n1: great combat"}
        ]
    assert ("response_format" in sent) == json_mode
    if json_mode:
        assert sent["response_format"] == {"type": "json_object"}


def test_thinking_is_dropped_and_tokens_are_counted() -> None:
    server, _, _ = client([completion('<think>labels, {maybe}</think>\n{"labels": []}')])

    answer = server.generate("system", "user", json_schema=SCHEMA)

    assert answer == LLMResponse(
        text='{"labels": []}', input_tokens=1500, output_tokens=300, finish_reason="stop"
    )


def test_an_outage_that_outlasts_the_retries_is_not_a_request_error() -> None:
    down = [httpx2.Response(503, text="model is loading") for _ in range(MAX_ATTEMPTS)]
    server, seen, clock = client(down)

    with pytest.raises(TeacherUnavailableError, match="503") as raised:
        server.generate("system", "user", json_schema=SCHEMA)

    assert not isinstance(raised.value, TeacherRequestError)
    assert len(seen) == MAX_ATTEMPTS
    assert clock.slept


def test_a_server_error_that_clears_is_retried() -> None:
    server, seen, _ = client([httpx2.Response(502, text="bad gateway"), completion("{}")])

    assert server.generate("system", "user", json_schema=SCHEMA).text == "{}"
    assert len(seen) == 2


def test_a_refused_request_is_a_request_error_and_never_shows_the_key() -> None:
    refused = httpx2.Response(401, json={"error": {"message": f"invalid key {KEY}"}})
    server, _, _ = client([refused])

    with pytest.raises(TeacherRequestError) as raised:
        server.generate("system", "user", json_schema=SCHEMA)

    assert raised.value.status == 401
    for shown in (str(raised.value), repr(raised.value), repr(server)):
        assert KEY not in shown
