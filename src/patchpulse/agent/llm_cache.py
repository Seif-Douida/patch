"""The agent's record/replay cache (design D8, spec §11): model answers keyed by the model and a
hash of the prompt, and (Phase 6) the results of the two tools that leave the data bundle, review
search and patch notes, keyed by the tool, its arguments and the data version; one SQLite file,
like Phase 2's label cache.

- `live`: neither reads nor writes (production).
- `record`: asks the models and the tools, and stores each answer that was used.
- `replay`: answers only from the file. A model miss raises ReplayMissError (the router's
  ModelNotRecordedError, which the graph treats as an outage); a tool miss raises
  ToolNotRecordedError. Every miss is named in `misses` (tests and Phase 6's evals run without a
  model or a network).
- `fill`: answers what's recorded, and asks and records the rest (Phase 6's live runs resume
  where a quota stop left them).

The file holds model answers and review text, so it lives in data/local and is never committed.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

CacheMode = Literal["live", "record", "replay", "fill"]
MODES: tuple[CacheMode, ...] = ("live", "record", "replay", "fill")
_STORES: frozenset[CacheMode] = frozenset({"record", "fill"})
# Under data/local, which git ignores: recorded answers can quote reviews.
AGENT_CACHE_FILE = "data/local/agent_answers.sqlite"

_SCHEMA = (
    """
CREATE TABLE IF NOT EXISTS answers (
    model TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    text TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    PRIMARY KEY (model, prompt_hash)
)
""",
    """
CREATE TABLE IF NOT EXISTS tool_results (
    tool TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (tool, key)
)
""",
)


class ReplayMissError(LookupError):
    """Replay mode found no recorded answer for this prompt from any model in the chain."""


class ToolNotRecordedError(LookupError):
    """Replay mode found no recorded result for this tool call."""


@dataclass(frozen=True)
class CachedAnswer:
    text: str
    input_tokens: int
    output_tokens: int


def prompt_hash(system: str, user: str, schema: Mapping[str, Any]) -> str:
    """What a prompt is, for the cache: the system and user text and the answer's schema."""
    payload = json.dumps([system, user, schema], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tool_key(tool: str, args: Mapping[str, Any], data_version: int) -> str:
    """What a tool call is, for the cache: the tool, its arguments and the data version."""
    payload = json.dumps([tool, dict(args), data_version], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ReplayCache:
    def __init__(self, path: Path, mode: CacheMode) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown cache mode {mode!r}; the modes are {', '.join(MODES)}")
        self.path = path
        self.mode = mode
        # Each replay miss, as "model:<step>" or "tool:<name>"; callers clear it per run.
        self.misses: list[str] = []
        if mode != "live":
            path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(path)) as connection, connection:
                for statement in _SCHEMA:
                    connection.execute(statement)

    def get(self, model: str, key: str) -> CachedAnswer | None:
        if self.mode == "live":
            return None
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                "SELECT text, input_tokens, output_tokens FROM answers "
                "WHERE model = ? AND prompt_hash = ?",
                (model, key),
            ).fetchone()
        return None if row is None else CachedAnswer(str(row[0]), int(row[1]), int(row[2]))

    def put(self, model: str, key: str, answer: CachedAnswer) -> None:
        if self.mode not in _STORES:
            return
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?, ?)",
                (model, key, answer.text, answer.input_tokens, answer.output_tokens),
            )

    def external(
        self, tool: str, args: Mapping[str, Any], data_version: int, live: Callable[[], Any]
    ) -> Any:
        """A tool call that leaves the data bundle, through the cache. `live` makes the real
        call; its result must be JSON-able. A live call that raises stores nothing."""
        if self.mode == "live":
            return live()
        key = tool_key(tool, args, data_version)
        if self.mode in ("replay", "fill"):
            with closing(sqlite3.connect(self.path)) as connection:
                row = connection.execute(
                    "SELECT value FROM tool_results WHERE tool = ? AND key = ?", (tool, key)
                ).fetchone()
            if row is not None:
                return json.loads(row[0])
            if self.mode == "replay":
                self.misses.append(f"tool:{tool}")
                raise ToolNotRecordedError(f"{tool}: no recorded result for this call")
        stored = json.dumps(live(), sort_keys=True, ensure_ascii=False)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "INSERT OR REPLACE INTO tool_results VALUES (?, ?, ?)", (tool, key, stored)
            )
        # As a replay would return it, so a recorded run and its replay see the same values.
        return json.loads(stored)
