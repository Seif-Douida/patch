"""Steam etiquette (spec C4): at most one request a second, backoff that honours Retry-After."""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import httpx2
import pytest

from patchpulse.ingest.http import (
    USER_AGENT,
    RateLimiter,
    RetryPolicy,
    SteamHttp,
    SteamRequestError,
    SteamUnavailableError,
)

URL = "https://store.steampowered.com/appreviews/1"


class FakeTime:
    def __init__(self) -> None:
        self.now = 1_000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def steam_http(
    handler: Callable[[httpx2.Request], httpx2.Response], fake: FakeTime, **policy: float
) -> SteamHttp:
    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    return SteamHttp(
        client,
        limiter=RateLimiter(min_interval_s=1.0, clock=fake.clock, sleep=fake.sleep),
        policy=RetryPolicy(**policy),  # type: ignore[arg-type]
        sleep=fake.sleep,
        jitter=lambda: 1.0,
    )


def responses(*items: httpx2.Response) -> Callable[[httpx2.Request], httpx2.Response]:
    queue = list(items)
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return queue.pop(0)

    handler.seen = seen  # type: ignore[attr-defined]
    return handler


def test_requests_are_spaced_at_least_one_second_apart() -> None:
    fake = FakeTime()
    times: list[float] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        times.append(fake.now)
        return httpx2.Response(200, json={"success": 1})

    http = steam_http(handler, fake)
    for _ in range(3):
        http.get_json(URL, {"json": 1})

    gaps = [later - earlier for earlier, later in pairwise(times)]
    assert gaps
    assert min(gaps) >= 1.0


def test_rate_limited_requests_back_off_and_honour_retry_after() -> None:
    fake = FakeTime()
    handler = responses(
        httpx2.Response(429, headers={"Retry-After": "7"}),
        httpx2.Response(503),
        httpx2.Response(200, json={"success": 1}),
    )

    payload, status = steam_http(handler, fake).get_json(URL, {"json": 1})

    assert (payload, status) == ({"success": 1}, 200)
    backoffs = [s for s in fake.sleeps if s > 1.0]
    assert backoffs[0] == 7.0  # Retry-After wins over the computed delay
    assert len(handler.seen) == 3  # type: ignore[attr-defined]


def test_gives_up_after_max_attempts_with_steam_unavailable() -> None:
    fake = FakeTime()
    handler = responses(*[httpx2.Response(502) for _ in range(3)])

    with pytest.raises(SteamUnavailableError, match="502"):
        steam_http(handler, fake, max_attempts=3).get_json(URL, {"json": 1})


def test_timeouts_are_retried() -> None:
    fake = FakeTime()
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx2.ReadTimeout("slow", request=request)
        return httpx2.Response(200, json={"success": 1})

    payload, _ = steam_http(handler, fake).get_json(URL, {"json": 1})

    assert payload == {"success": 1}
    assert calls == 2


def test_client_errors_are_not_retried() -> None:
    fake = FakeTime()
    handler = responses(httpx2.Response(404), httpx2.Response(200, json={}))

    with pytest.raises(SteamRequestError, match="404"):
        steam_http(handler, fake).get_json(URL, {"json": 1})

    assert len(handler.seen) == 1  # type: ignore[attr-defined]


def test_backoff_grows_and_is_capped() -> None:
    policy = RetryPolicy(max_attempts=10, base_s=2.0, cap_s=20.0)
    delays = [policy.delay(attempt, retry_after=None, jitter=1.0) for attempt in range(1, 7)]
    assert delays == [2.0, 4.0, 8.0, 16.0, 20.0, 20.0]
    assert policy.delay(1, retry_after=None, jitter=0.0) == 1.0  # equal jitter: at least half
    assert policy.delay(1, retry_after=500.0, jitter=1.0) == 20.0  # Retry-After is capped too


def test_user_agent_is_sent() -> None:
    fake = FakeTime()
    handler = responses(httpx2.Response(200, json={"success": 1}))

    steam_http(handler, fake).get_json(URL, {"json": 1})

    request = handler.seen[0]  # type: ignore[attr-defined]
    assert request.headers["User-Agent"] == USER_AGENT
    assert "github.com/Seif-Douida/patch" in USER_AGENT
