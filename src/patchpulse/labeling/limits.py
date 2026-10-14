"""Pace Gemini requests under a model's free-tier limits (plan Task 7, spec C2).

Per minute, requests and tokens are counted over a sliding 60-second window, at `use_fraction` of
the limits. A request's tokens are estimated before it's sent and replaced by Google's count after
(`settle`). Per day, requests are counted in `data/local/gemini_usage.json`, which survives restarts
and resets at midnight in Los Angeles, when Google resets the free quota. Both caps stay at
`use_fraction` because the quota is per project and other work shares it.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from patchpulse.labeling.teachers import ModelConfig

PACIFIC = ZoneInfo("America/Los_Angeles")
USAGE_FILE = "gemini_usage.json"  # in data/local
WINDOW = timedelta(seconds=60)


class QuotaExhaustedError(RuntimeError):
    """A model's daily requests are used up until midnight Pacific time."""


@dataclass
class _Sent:
    at: datetime
    tokens: int


def _utc_now() -> datetime:
    return datetime.now(UTC)


class RequestLimiter:
    def __init__(
        self,
        limits: ModelConfig,
        *,
        state_path: Path,
        clock: Callable[[], datetime] = _utc_now,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = limits.id
        self.requests_per_minute = limits.share(limits.rpm)
        self.tokens_per_minute = limits.share(limits.tpm)
        self.requests_per_day = limits.share(limits.rpd)
        self.state_path = state_path
        self.clock = clock
        self.sleep = sleep
        self._window: deque[_Sent] = deque()  # oldest first

    def acquire(self, tokens: int) -> None:
        """Wait until a request of about `tokens` fits this minute, then count it for the day."""
        if tokens > self.tokens_per_minute:
            raise ValueError(
                f"{self.model}: a {tokens}-token request can never fit {self.tokens_per_minute} "
                "tokens a minute; send smaller batches"
            )
        if self._used_today() >= self.requests_per_day:
            raise QuotaExhaustedError(
                f"{self.model}: {self.requests_per_day} requests used today (the share of the free "
                "daily quota PatchPulse may use); it resets at midnight Pacific time"
            )
        while True:
            now = self.clock()
            while self._window and self._window[0].at <= now - WINDOW:
                self._window.popleft()
            used = sum(sent.tokens for sent in self._window)
            fits = used + tokens <= self.tokens_per_minute
            if fits and len(self._window) < self.requests_per_minute:
                break
            self.sleep(max((self._window[0].at + WINDOW - now).total_seconds(), 0.001))
        self._window.append(_Sent(now, tokens))
        self._count_today()

    def settle(self, tokens: int) -> None:
        """Replace the latest request's estimate with the tokens Google counted."""
        if self._window:
            self._window[-1].tokens = tokens

    def _today(self) -> str:
        return self.clock().astimezone(PACIFIC).date().isoformat()

    def _read(self) -> dict[str, int]:
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        if state.get("pacific_date") != self._today():
            return {}
        return {str(model): int(count) for model, count in state.get("requests", {}).items()}

    def _used_today(self) -> int:
        return self._read().get(self.model, 0)

    def _count_today(self) -> None:
        requests = self._read()
        requests[self.model] = requests.get(self.model, 0) + 1
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"pacific_date": self._today(), "requests": requests}, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)
