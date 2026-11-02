"""The model router (design D8, spec §10.7): the agent's every model call, through the chain in
config/models.yaml and the record/replay cache.

An answer is used only when it is complete, parses as JSON (a code fence stripped) and fits the
step's schema. Anything else counts as that model failing and the next model is tried at once:
a broken answer, an outage, a timeout, a refusal. The client waits once for a per-minute limit; a
daily-quota refusal retires the model until midnight Pacific, when Google resets the free quota;
three failures in a row skip a model for ten minutes. With every model failing, AllModelsDownError,
and the caller writes a basic report. Logs and errors name the step, the model and the reason,
never the prompt.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import httpx2

from patchpulse.agent.llm_cache import CachedAnswer, ReplayCache, ReplayMissError, prompt_hash
from patchpulse.labeling.gemini import GeminiClient
from patchpulse.labeling.limits import PACIFIC, QuotaExhaustedError, RequestLimiter
from patchpulse.labeling.llm import LLMResponse, TeacherRequestError, TeacherUnavailableError
from patchpulse.labeling.teachers import TeacherModels

log = logging.getLogger("patchpulse.agent.router")

AGENT_ATTEMPTS = 2  # one wait for a per-minute limit, then the next model
AGENT_TIMEOUT_S = 60.0  # a draft takes seconds; a minute means the model is stuck
MAX_OUTPUT_TOKENS = 4096
_FENCE = re.compile(r"^```[\w-]*\s*\n?(.*?)\n?\s*```$", re.DOTALL)


class ModelClient(Protocol):
    def generate(
        self,
        system: str,
        user: str,
        *,
        json_schema: Mapping[str, Any] | None,
        expected_output_tokens: int = 512,
        max_output_tokens: int = 4096,
    ) -> LLMResponse: ...


class AllModelsDownError(RuntimeError):
    """Every model in the chain failed or is resting; the message lists why, per model."""


@dataclass(frozen=True)
class Routed:
    model_id: str
    data: dict[str, Any]
    input_tokens: int
    output_tokens: int
    cached: bool


class BadAnswerError(ValueError):
    """The model answered, but not with something the step can use."""


def _strip_fence(text: str) -> str:
    text = text.strip()
    fenced = _FENCE.match(text)
    if fenced:
        return fenced.group(1).strip()
    return text.removesuffix("```").strip()  # Gemma's habit: valid JSON, then a stray fence


def _type_ok(value: Any, kind: str) -> bool:
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    return True


def schema_problem(value: Any, schema: Mapping[str, Any], where: str = "$") -> str | None:
    """The first way `value` breaks `schema` (the subset our schemas use: type, properties,
    required, items, enum; properties not listed are refused), or None."""
    kind = schema.get("type")
    if isinstance(kind, str) and not _type_ok(value, kind):
        return f"{where} should be {kind}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{where} isn't one of the allowed values"
    if isinstance(value, dict) and "properties" in schema:
        properties = schema["properties"]
        for name in schema.get("required", []):
            if name not in value:
                return f"{where}.{name} is missing"
        for name, item in value.items():
            if name not in properties:
                return f"{where}.{name} isn't in the schema"
            problem = schema_problem(item, properties[name], f"{where}.{name}")
            if problem:
                return problem
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            problem = schema_problem(item, schema["items"], f"{where}[{i}]")
            if problem:
                return problem
    return None


def parse_answer(response: LLMResponse, schema: Mapping[str, Any]) -> dict[str, Any]:
    if response.finish_reason == "MAX_TOKENS":
        raise BadAnswerError("the answer was cut off at the output limit")
    try:
        data = json.loads(_strip_fence(response.text))
    except json.JSONDecodeError:
        raise BadAnswerError("the answer isn't valid JSON") from None
    problem = schema_problem(data, schema)
    if problem:
        raise BadAnswerError(f"the answer doesn't fit the schema: {problem}")
    if not isinstance(data, dict):
        raise BadAnswerError("the answer isn't a JSON object")
    return data


def next_pacific_midnight(now: datetime) -> datetime:
    local = now.astimezone(PACIFIC)
    tomorrow = (local + timedelta(days=1)).date()
    return datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=PACIFIC)


class ModelRouter:
    def __init__(
        self,
        chain: Sequence[str],
        clients: Mapping[str, ModelClient],
        cache: ReplayCache,
        *,
        clock: Callable[[], datetime],
        breaker_failures: int = 3,
        breaker_minutes: int = 10,
    ) -> None:
        self.chain = list(chain)
        self.clients = dict(clients)
        self.cache = cache
        self.clock = clock
        self.breaker_failures = breaker_failures
        self.breaker = timedelta(minutes=breaker_minutes)
        self._failures: dict[str, int] = {}
        self._resting_until: dict[str, datetime] = {}  # a breaker or a daily quota
        self._quota_until: dict[str, datetime] = {}  # a daily quota only

    def generate(
        self,
        step: str,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        *,
        expected_output_tokens: int = 1024,
    ) -> Routed:
        key = prompt_hash(system, user, schema)
        if self.cache.mode == "replay":
            return self._replay(step, key, schema)
        reasons: dict[str, str] = {}
        for model in self.chain:
            now = self.clock()
            resting = self._resting_until.get(model)
            if resting is not None and now < resting:
                reasons[model] = f"resting until {resting.isoformat(timespec='minutes')}"
                continue
            try:
                response = self.clients[model].generate(
                    system,
                    user,
                    json_schema=schema,
                    expected_output_tokens=expected_output_tokens,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                )
                data = parse_answer(response, schema)
            except QuotaExhaustedError:
                self._resting_until[model] = next_pacific_midnight(now)
                self._quota_until[model] = self._resting_until[model]
                reasons[model] = "daily quota used up"
                log.warning("%s: %s's daily quota is used up; resting it", step, model)
                continue
            except (TeacherUnavailableError, TeacherRequestError, BadAnswerError) as error:
                reasons[model] = str(error)
                self._fail(model, now)
                log.warning("%s: %s failed (%s); trying the next model", step, model, error)
                continue
            except Exception as error:  # anything else a client raises is that model failing
                failure = type(error).__name__  # the type only: a message could carry the key
                reasons[model] = failure
                self._fail(model, now)
                log.warning("%s: %s failed (%s); trying the next model", step, model, failure)
                continue
            self._failures[model] = 0
            self.cache.put(
                model,
                key,
                CachedAnswer(response.text, response.input_tokens, response.output_tokens),
            )
            return Routed(model, data, response.input_tokens, response.output_tokens, cached=False)
        summary = "; ".join(f"{m}: {r}" for m, r in reasons.items())
        raise AllModelsDownError(f"{step}: every model failed ({summary})")

    def quota_spent(self) -> bool:
        """Whether every model in the chain has used up its daily quota (it rests until a Pacific
        midnight); a breaker's rest of a few minutes doesn't count."""
        now = self.clock()
        return all(now < self._quota_until.get(model, now) for model in self.chain)

    def _fail(self, model: str, now: datetime) -> None:
        self._failures[model] = self._failures.get(model, 0) + 1
        if self._failures[model] >= self.breaker_failures:
            self._failures[model] = 0
            self._resting_until[model] = now + self.breaker
            log.warning("%s failed %d times in a row; skipping it for a while", model,
                        self.breaker_failures)  # fmt: skip

    def _replay(self, step: str, key: str, schema: Mapping[str, Any]) -> Routed:
        for model in self.chain:
            cached = self.cache.get(model, key)
            if cached is None:
                continue
            response = LLMResponse(cached.text, cached.input_tokens, cached.output_tokens, "STOP")
            try:
                data = parse_answer(response, schema)
            except BadAnswerError:
                continue
            return Routed(model, data, cached.input_tokens, cached.output_tokens, cached=True)
        raise ReplayMissError(f"{step}: no recorded answer for this prompt")


def make_agent_clients(
    models: TeacherModels,
    api_key: str,
    http: httpx2.Client,
    *,
    state_path: Path,
    clock: Callable[[], datetime],
    sleep: Callable[[float], None],
) -> dict[str, GeminiClient]:
    """One Gemini client per model in the agent's chain, built for routing: one wait for a
    per-minute limit, no retries for outages or timeouts, a minute's timeout."""
    if models.agent is None:
        raise ValueError("config/models.yaml has no agent section")
    clients = {}
    for name in models.agent.chain:
        config = models.candidates[name]
        limiter = RequestLimiter(config, state_path=state_path, clock=clock, sleep=sleep)
        clients[name] = GeminiClient(
            api_key,
            http,
            limiter,
            config,
            sleep=sleep,
            max_attempts=AGENT_ATTEMPTS,
            retry_errors=False,
            timeout_s=AGENT_TIMEOUT_S,
        )
    return clients
