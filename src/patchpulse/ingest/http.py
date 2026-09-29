"""Polite HTTP for Steam's public endpoints (spec C4).

One shared `RateLimiter` paces every Steam host at one request per 2.25 s, with a descriptive
User-Agent. 5xx responses and network errors are retried with capped exponential backoff that
honours `Retry-After`; other 4xx responses are not retried.

HTTP 429 is Steam's rate limit. Measured on 2026-09-29: /appreviews answered 429 after exactly
150 requests, both at 1 request/s and at 0.8/s, and served again 5 minutes later, so about 150
requests per 5 minutes per IP; the pace stays ~11% under it. On a 429 the client slows the shared
limiter, pauses once for the window to pass, and if Steam still refuses, gives up and fails every
later request at once, so a blocked night ends quickly instead of waiting out the job timeout.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any

import httpx2

log = logging.getLogger("patchpulse.ingest.http")

REPO_URL = "https://github.com/Seif-Douida/patch"
USER_AGENT = f"PatchPulse/{version('patchpulse')} (+{REPO_URL})"
REQUEST_TIMEOUT_S = 30.0
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class SteamError(RuntimeError):
    """Base class for Steam request failures."""


class SteamRequestError(SteamError):
    """A request Steam rejected (4xx other than 429) or answered with something unusable."""


class SteamUnavailableError(SteamError):
    """Steam kept failing (429, 5xx, network errors) after every retry."""


@dataclass
class RateLimiter:
    """Spaces requests at least `min_interval_s` apart. Share one instance for all of Steam."""

    min_interval_s: float = 2.25  # ~133 requests per 5 minutes; Steam allows ~150
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _last: float | None = field(default=None, init=False, repr=False)

    def wait(self) -> None:
        if self._last is not None:
            remaining = self._last + self.min_interval_s - self.clock()
            if remaining > 0:
                self.sleep(remaining)
        self._last = self.clock()

    def slow_down(self, factor: float = 1.5, ceiling_s: float = 5.0) -> None:
        self.min_interval_s = min(self.min_interval_s * factor, ceiling_s)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 6
    base_s: float = 2.0
    cap_s: float = 120.0
    rate_limit_pause_s: float = 300.0  # Steam's window is about 5 minutes
    rate_limit_pauses: int = 1  # a second 429 means a longer block: give up for this run

    def delay(
        self, attempt: int, *, retry_after: float | None, jitter: float, rate_limited: bool = False
    ) -> float:
        """Seconds to wait after failed attempt `attempt` (1-based); jitter is in [0, 1]."""
        if rate_limited:
            return min(retry_after or self.rate_limit_pause_s, self.rate_limit_pause_s)
        if retry_after is not None:
            return min(retry_after, self.cap_s)
        full = min(self.cap_s, self.base_s * 2.0 ** (attempt - 1))
        return full / 2 + full / 2 * jitter  # "equal jitter": at least half the full delay


def _retry_after(response: httpx2.Response) -> float | None:
    try:
        return float(response.headers["Retry-After"])
    except (KeyError, ValueError):
        return None  # absent, or an HTTP date (not used by Steam)


class SteamHttp:
    def __init__(
        self,
        client: httpx2.Client,
        *,
        limiter: RateLimiter,
        policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.client = client
        self.limiter = limiter
        self.policy = policy or RetryPolicy()
        self.sleep = sleep
        self.jitter = jitter
        self.refused = False  # Steam kept rate-limiting after a pause; no more requests this run

    def get_json(self, url: str, params: Mapping[str, str | int]) -> tuple[dict[str, Any], int]:
        """GET a JSON object; returns (payload, HTTP status)."""
        if self.refused:
            raise SteamUnavailableError(f"{url}: not requested: Steam is rate-limiting this run")
        problem = "no attempt made"
        pauses = 0
        rate_limited = False
        for attempt in range(1, self.policy.max_attempts + 1):
            self.limiter.wait()
            retry_after: float | None = None
            try:
                response = self.client.get(
                    url,
                    params=dict(params),
                    headers={"User-Agent": USER_AGENT},
                    timeout=REQUEST_TIMEOUT_S,
                )
            except httpx2.TransportError as error:  # includes timeouts
                problem = type(error).__name__
            else:
                if response.status_code in _RETRYABLE_STATUS:
                    problem = f"HTTP {response.status_code}"
                    retry_after = _retry_after(response)
                elif response.is_error:
                    raise SteamRequestError(f"HTTP {response.status_code} from {url}")
                else:
                    return _json_object(response, url), response.status_code
            rate_limited = problem == "HTTP 429"
            if rate_limited:
                self.limiter.slow_down()
                if pauses >= self.policy.rate_limit_pauses:
                    break
                pauses += 1
            if attempt < self.policy.max_attempts:
                jitter = self.jitter()
                delay = self.policy.delay(
                    attempt, retry_after=retry_after, jitter=jitter, rate_limited=rate_limited
                )
                if rate_limited:
                    log.warning(
                        "Steam rate limit (HTTP 429): pausing %.0f s, then 1 request every %.2f s",
                        delay,
                        self.limiter.min_interval_s,
                    )
                self.sleep(delay)
        self.refused = rate_limited
        raise SteamUnavailableError(f"{url}: gave up after {attempt} attempt(s) (last: {problem})")


def _json_object(response: httpx2.Response, url: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as error:
        raise SteamRequestError(f"invalid JSON from {url}") from error
    if not isinstance(payload, dict):
        raise SteamRequestError(f"expected a JSON object from {url}")
    return payload
