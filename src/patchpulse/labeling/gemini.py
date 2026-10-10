"""The Gemini API client for the teacher (plan Task 7, spec §7.3; free tier, no billing: C2).

One POST per call to `/v1beta/models/{model}:generateContent`, with the key in the
`x-goog-api-key` header, never in the URL, so no logged URL can carry it. Every attempt goes
through the model's `RequestLimiter`.

- A 429 for a per-minute quota waits Google's `retryDelay` and tries again.
- A 429 for the daily quota raises `QuotaExhaustedError` at once: only Pacific midnight helps.
- 5xx answers (an overloaded model) and network errors back off and retry.

Nothing this module raises, logs or prints contains the key: Google's messages are scrubbed.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx2

from patchpulse.labeling.limits import QuotaExhaustedError, RequestLimiter
from patchpulse.labeling.teachers import ModelConfig

log = logging.getLogger("patchpulse.labeling.gemini")

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
REQUEST_TIMEOUT_S = 120.0
MAX_ATTEMPTS = 4
_BACKOFF_S = (5.0, 15.0, 45.0)
_DEFAULT_RETRY_S = 60.0  # a per-minute 429 without a retryDelay
_CHARS_PER_TOKEN = 3  # a cautious estimate until Google counts the real tokens


@dataclass(frozen=True)
class GeminiResponse:
    text: str
    input_tokens: int
    output_tokens: int  # the answer plus any thinking: both count against TPM
    finish_reason: str


class GeminiError(RuntimeError):
    """Google refused or failed a request; `status` is the HTTP status (0 for a network error)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message


@dataclass(frozen=True)
class _Problem:
    message: str
    quota_ids: tuple[str, ...]
    retry_delay_s: float | None


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        http: httpx2.Client,
        limiter: RequestLimiter,
        model: ModelConfig,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("a Gemini API key is required (PP_GEMINI_API_KEY in .env)")
        self._api_key = api_key
        self.http = http
        self.limiter = limiter
        self.model = model
        self.sleep = sleep

    def __repr__(self) -> str:
        return f"GeminiClient(model={self.model.id!r})"

    def generate(
        self,
        system: str,
        user: str,
        *,
        json_schema: Mapping[str, Any] | None,
        expected_output_tokens: int = 512,
        max_output_tokens: int = 4096,
    ) -> GeminiResponse:
        """One answer from the model. `json_schema` is sent only to models with JSON mode."""
        body = self._body(system, user, json_schema, max_output_tokens)
        estimate = (len(system) + len(user)) // _CHARS_PER_TOKEN + expected_output_tokens
        url = f"{API_ROOT}/models/{self.model.id}:generateContent"
        last = GeminiError(0, "no attempt made")
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.limiter.acquire(estimate)
            backoff = _BACKOFF_S[min(attempt, len(_BACKOFF_S)) - 1]
            try:
                response = self.http.post(
                    url,
                    json=body,
                    headers={"x-goog-api-key": self._api_key},
                    timeout=REQUEST_TIMEOUT_S,
                )
            except httpx2.TransportError as error:  # includes timeouts
                last = GeminiError(0, f"network error: {type(error).__name__}")
                self._wait(attempt, backoff, last)
                continue
            if response.status_code == httpx2.codes.OK:
                answer = self._answer(response)
                self.limiter.settle(answer.input_tokens + answer.output_tokens)
                return answer
            problem = self._problem(response)
            last = GeminiError(response.status_code, problem.message)
            if response.status_code == httpx2.codes.TOO_MANY_REQUESTS:
                if any("PerDay" in quota for quota in problem.quota_ids):
                    raise QuotaExhaustedError(
                        f"{self.model.id}: Google's free daily quota is used up "
                        f"({problem.message}); it resets at midnight Pacific time"
                    )
                self._wait(attempt, problem.retry_delay_s or _DEFAULT_RETRY_S, last)
                continue
            if response.status_code >= httpx2.codes.INTERNAL_SERVER_ERROR:
                self._wait(attempt, backoff, last)
                continue
            raise last
        raise last

    def _wait(self, attempt: int, seconds: float, problem: GeminiError) -> None:
        if attempt == MAX_ATTEMPTS:
            return
        log.warning(
            "%s: attempt %d failed (%s); retrying in %.0f s",
            self.model.id,
            attempt,
            problem,
            seconds,
        )
        self.sleep(seconds)

    def _body(
        self,
        system: str,
        user: str,
        json_schema: Mapping[str, Any] | None,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        generation: dict[str, Any] = {"temperature": 0, "maxOutputTokens": max_output_tokens}
        if self.model.json_mode:
            generation["responseMimeType"] = "application/json"
            if json_schema is not None:
                generation["responseJsonSchema"] = dict(json_schema)
        # Models without system instructions (Gemma 3 refused them) get the system text first.
        text = user if self.model.system_instruction else f"{system}\n\n{user}"
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": generation,
        }
        if self.model.system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    def _answer(self, response: httpx2.Response) -> GeminiResponse:
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise GeminiError(response.status_code, f"no answer: the prompt was blocked ({reason})")
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        usage = data.get("usageMetadata") or {}
        return GeminiResponse(
            text="".join(part.get("text", "") for part in parts if not part.get("thought")),
            input_tokens=int(usage.get("promptTokenCount", 0)),
            output_tokens=int(usage.get("candidatesTokenCount", 0))
            + int(usage.get("thoughtsTokenCount", 0)),
            finish_reason=str(candidate.get("finishReason", "")),
        )

    def _problem(self, response: httpx2.Response) -> _Problem:
        try:
            error = response.json().get("error") or {}
        except ValueError:
            error = {}
        message = str(error.get("message") or response.text[:300] or response.reason_phrase)
        quota_ids: list[str] = []
        retry_delay: float | None = None
        for detail in error.get("details") or []:
            kind = str(detail.get("@type", ""))
            if kind.endswith("QuotaFailure"):
                quota_ids += [str(v.get("quotaId", "")) for v in detail.get("violations") or []]
            elif kind.endswith("RetryInfo"):
                try:
                    retry_delay = float(str(detail.get("retryDelay", "")).removesuffix("s"))
                except ValueError:
                    retry_delay = None
        return _Problem(self._scrub(message), tuple(quota_ids), retry_delay)

    def _scrub(self, text: str) -> str:
        return text.replace(self._api_key, "***")
