"""Where a run reports its steps (design D13): the graph calls a Tracer after each step with counts,
ids and model names only, never review text or prompt bodies. NoTracer drops them; Task 18 adds
Langfuse.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class Tracer(Protocol):
    def step(self, run: str, name: str, fields: Mapping[str, object]) -> None: ...


class NoTracer:
    def step(self, run: str, name: str, fields: Mapping[str, object]) -> None:
        return None
