"""A wall clock for tests: `sleep` advances it instead of waiting."""

from __future__ import annotations

from datetime import datetime, timedelta


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.now = start
        self.slept: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += timedelta(seconds=seconds)
