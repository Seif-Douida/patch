"""The agent's record/replay cache (design D8, spec §11): model answers keyed by the model and a
hash of the prompt, in one SQLite file, like Phase 2's label cache.

- `live`: neither reads nor writes (production).
- `record`: asks the models and stores each answer that was used.
- `replay`: answers only from the file; a miss raises ReplayMissError (tests and Phase 6's evals
  run without a model).

The file holds model answers, which can quote reviews, so it lives in data/local and is never
committed.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

CacheMode = Literal["live", "record", "replay"]
MODES: tuple[CacheMode, ...] = ("live", "record", "replay")
AGENT_CACHE_FILE = "agent_answers.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    model TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    text TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    PRIMARY KEY (model, prompt_hash)
)
"""


class ReplayMissError(LookupError):
    """Replay mode found no recorded answer for this prompt from any model in the chain."""


@dataclass(frozen=True)
class CachedAnswer:
    text: str
    input_tokens: int
    output_tokens: int


def prompt_hash(system: str, user: str, schema: Mapping[str, Any]) -> str:
    """What a prompt is, for the cache: the system and user text and the answer's schema."""
    payload = json.dumps([system, user, schema], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ReplayCache:
    def __init__(self, path: Path, mode: CacheMode) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown cache mode {mode!r}; the modes are {', '.join(MODES)}")
        self.path = path
        self.mode = mode
        if mode != "live":
            path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute(_SCHEMA)

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
        if self.mode != "record":
            return
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?, ?)",
                (model, key, answer.text, answer.input_tokens, answer.output_tokens),
            )
