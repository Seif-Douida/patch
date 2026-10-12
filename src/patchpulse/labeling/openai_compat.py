"""The teacher client for a self-hosted, OpenAI-style server (ADR-020).

One POST per call to `{base_url}/chat/completions`, with the key as a Bearer token in a header.
The server has no rate limits, so nothing paces requests; parallel batches are bounded by the
caller's worker count instead. Failures follow the shared policy (`llm.py`): 5xx answers, 429s and
network errors retry with backoff, and what still fails raises `TeacherUnavailableError`; a refused
request (other 4xx) raises `TeacherRequestError` at once. A model that thinks before answering
(`<think>…</think>`) has its thinking dropped. Nothing raised, logged or printed holds the key.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx2

from patchpulse.labeling.llm import (
    BACKOFF_S,
    MAX_ATTEMPTS,
    LLMResponse,
    TeacherRequestError,
    TeacherUnavailableError,
)
from patchpulse.labeling.teachers import ModelConfig

log = logging.getLogger("patchpulse.labeling.openai_compat")

REQUEST_TIMEOUT_S = 300.0  # a busy GPU can take minutes for a 20-review batch
_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)


class OpenAICompatError(TeacherRequestError):
    """The server refused the request; `status` is the HTTP status."""


class OpenAICompatUnavailableError(TeacherUnavailableError):
    """The server kept failing through every retry: try again later."""


class OpenAICompatClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        http: httpx2.Client,
        model: ModelConfig,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("the server's API key is required (PP_LLM_SERVER_KEY in .env)")
        self._api_key = api_key
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.http = http
        self.model = model
        self.sleep = sleep

    def __repr__(self) -> str:
        return f"OpenAICompatClient(model={self.model.id!r})"

    def generate(
        self,
        system: str,
        user: str,
        *,
        json_schema: Mapping[str, Any] | None,
        expected_output_tokens: int = 512,
        max_output_tokens: int = 4096,
    ) -> LLMResponse:
        """One answer. JSON mode asks for an object; the caller validates the answer anyway."""
        messages = (
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
            if self.model.system_instruction
            else [{"role": "user", "content": f"{system}\n\n{user}"}]
        )
        body: dict[str, Any] = {
            "model": self.model.id,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_output_tokens,
        }
        if self.model.json_mode:
            body["response_format"] = {"type": "json_object"}
        last = OpenAICompatUnavailableError(0, "no attempt made")
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self.http.post(
                    self.url,
                    json=body,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=REQUEST_TIMEOUT_S,
                )
            except httpx2.TransportError as error:  # includes timeouts
                last = OpenAICompatUnavailableError(0, f"network error: {type(error).__name__}")
                self._wait(attempt, last)
                continue
            if response.status_code == httpx2.codes.OK:
                return self._answer(response)
            message = self._scrub(response.text[:300] or response.reason_phrase)
            retryable = response.status_code == httpx2.codes.TOO_MANY_REQUESTS
            if retryable or response.status_code >= httpx2.codes.INTERNAL_SERVER_ERROR:
                last = OpenAICompatUnavailableError(response.status_code, message)
                self._wait(attempt, last)
                continue
            raise OpenAICompatError(response.status_code, message)
        raise last

    def _wait(self, attempt: int, problem: Exception) -> None:
        if attempt == MAX_ATTEMPTS:
            return
        seconds = BACKOFF_S[min(attempt, len(BACKOFF_S)) - 1]
        log.warning(
            "%s: attempt %d failed (%s); retrying in %.0f s",
            self.model.id,
            attempt,
            problem,
            seconds,
        )
        self.sleep(seconds)

    def _answer(self, response: httpx2.Response) -> LLMResponse:
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        return LLMResponse(
            text=_THINKING.sub("", content).strip(),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=str(choice.get("finish_reason", "")),
        )

    def _scrub(self, text: str) -> str:
        return text.replace(self._api_key, "***")
