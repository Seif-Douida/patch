"""Pacing Gemini requests under the free tier (plan Task 7, spec C2: TPM is the binding limit)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fake_clock import FakeClock

from patchpulse.labeling.limits import QuotaExhaustedError, RequestLimiter
from patchpulse.labeling.teachers import ModelConfig

# 2026-10-10 10:00 UTC is 03:00 in Los Angeles (PDT, UTC-7).
START = datetime(2026, 10, 10, 10, 0, tzinfo=UTC)


def model(*, rpm: int = 30, tpm: int = 16_000, rpd: int = 14_400) -> ModelConfig:
    return ModelConfig(
        id="gemma-4-31b-it",
        rpm=rpm,
        tpm=tpm,
        rpd=rpd,
        json_mode=False,
        system_instruction=False,
        use_fraction=0.5,
    )


def limiter(tmp_path: Path, clock: FakeClock, config: ModelConfig | None = None) -> RequestLimiter:
    return RequestLimiter(
        config or model(),
        state_path=tmp_path / "gemini_usage.json",
        clock=clock,
        sleep=clock.sleep,
    )


def test_tokens_per_minute_are_paced_at_half_the_limit(tmp_path: Path) -> None:
    clock = FakeClock(START)
    pacer = limiter(tmp_path, clock)  # 16,000 TPM at half: 8,000 a minute

    pacer.acquire(5_000)
    pacer.acquire(3_000)  # exactly 8,000: no wait
    assert clock.slept == []

    pacer.acquire(1_000)  # over 8,000 until the first request leaves the minute

    assert clock.now - START == timedelta(seconds=60)


def test_requests_per_minute_are_paced_at_half_the_limit(tmp_path: Path) -> None:
    clock = FakeClock(START)
    pacer = limiter(tmp_path, clock, model(rpm=4))  # 2 requests a minute

    pacer.acquire(10)
    clock.sleep(10)
    pacer.acquire(10)
    pacer.acquire(10)

    assert clock.now - START == timedelta(seconds=60)


def test_settled_tokens_replace_the_estimate(tmp_path: Path) -> None:
    clock = FakeClock(START)
    pacer = limiter(tmp_path, clock)

    pacer.acquire(7_000)
    pacer.settle(2_000)  # Google counted fewer tokens than estimated
    pacer.acquire(6_000)

    assert clock.slept == []


def test_a_request_larger_than_the_minute_budget_is_refused(tmp_path: Path) -> None:
    pacer = limiter(tmp_path, FakeClock(START))

    with pytest.raises(ValueError, match="8000"):
        pacer.acquire(8_001)


def test_daily_counter_survives_a_restart_and_resets_at_pacific_midnight(tmp_path: Path) -> None:
    clock = FakeClock(START)
    config = model(rpd=4)  # 2 requests a day at half

    first = limiter(tmp_path, clock, config)
    first.acquire(10)
    first.acquire(10)

    restarted = limiter(tmp_path, clock, config)
    with pytest.raises(QuotaExhaustedError, match="gemma-4-31b-it"):
        restarted.acquire(10)

    clock.now = datetime(2026, 10, 11, 6, 59, tzinfo=UTC)  # 23:59 on 10 Oct in Los Angeles
    with pytest.raises(QuotaExhaustedError):
        restarted.acquire(10)

    clock.now = datetime(2026, 10, 11, 7, 1, tzinfo=UTC)  # 00:01 on 11 Oct in Los Angeles
    restarted.acquire(10)

    state = json.loads((tmp_path / "gemini_usage.json").read_text(encoding="utf-8"))
    assert state == {"pacific_date": "2026-10-11", "requests": {"gemma-4-31b-it": 1}}


def test_models_share_the_usage_file_without_overwriting_each_other(tmp_path: Path) -> None:
    clock = FakeClock(START)
    gemma = limiter(tmp_path, clock)
    lite = limiter(tmp_path, clock, model().model_copy(update={"id": "gemini-3.5-flash-lite"}))

    gemma.acquire(10)
    lite.acquire(10)
    gemma.acquire(10)

    state = json.loads((tmp_path / "gemini_usage.json").read_text(encoding="utf-8"))
    assert state["requests"] == {"gemma-4-31b-it": 2, "gemini-3.5-flash-lite": 1}
