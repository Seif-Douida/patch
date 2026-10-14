"""Teacher labels already paid for (plan Task 7): one SQLite file in data/local, never committed.

A review is sent to Gemini once per (model, prompt): the prompt hash changes whenever the prompt,
the guidelines it quotes or the cleaning does, so a new prompt is a fresh cache. The requests and
tokens those labels cost are kept beside them (`Usage`), so a rerun that finds everything cached
still knows a model's measured throughput.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from patchpulse.labeling.gold import format_aspects, parse_aspects

TEACHER_CACHE_FILE = "teacher_labels.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS teacher_labels (
    review_id INTEGER NOT NULL,
    model TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    aspects TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (review_id, model, prompt_hash)
)
"""
_USAGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS teacher_usage (
    model TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    calls INTEGER NOT NULL,
    tokens INTEGER NOT NULL,
    labeled INTEGER NOT NULL,
    PRIMARY KEY (model, prompt_hash)
)
"""
_BATCH = 500  # stays under SQLite's limit on query parameters


@dataclass(frozen=True)
class Usage:
    """Requests, tokens (Google's count) and labels a model has spent on one prompt, all runs."""

    calls: int
    tokens: int
    labeled: int


class LabelCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute(_SCHEMA)
            connection.execute(_USAGE_SCHEMA)

    def add_usage(
        self, *, model: str, prompt_hash: str, calls: int, tokens: int, labeled: int
    ) -> None:
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "INSERT INTO teacher_usage (model, prompt_hash, calls, tokens, labeled) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT (model, prompt_hash) DO UPDATE SET "
                "calls = calls + excluded.calls, tokens = tokens + excluded.tokens, "
                "labeled = labeled + excluded.labeled",
                (model, prompt_hash, calls, tokens, labeled),
            )

    def usage(self, *, model: str, prompt_hash: str) -> Usage | None:
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                "SELECT calls, tokens, labeled FROM teacher_usage "
                "WHERE model = ? AND prompt_hash = ?",
                (model, prompt_hash),
            ).fetchone()
        return None if row is None else Usage(int(row[0]), int(row[1]), int(row[2]))

    def get_many(
        self, review_ids: Iterable[int], *, model: str, prompt_hash: str
    ) -> dict[int, frozenset[str]]:
        ids = list(review_ids)
        found: dict[int, frozenset[str]] = {}
        with closing(sqlite3.connect(self.path)) as connection:
            for start in range(0, len(ids), _BATCH):
                chunk = ids[start : start + _BATCH]
                marks = ",".join("?" * len(chunk))
                rows = connection.execute(
                    "SELECT review_id, aspects FROM teacher_labels "  # noqa: S608 (only ? marks)
                    f"WHERE model = ? AND prompt_hash = ? AND review_id IN ({marks})",
                    [model, prompt_hash, *chunk],
                )
                found.update({int(i): parse_aspects(a) for i, a in rows})
        return found

    def missing(self, review_ids: Sequence[int], *, model: str, prompt_hash: str) -> list[int]:
        """The ids, in order, that this model and prompt haven't labeled yet."""
        cached = self.get_many(review_ids, model=model, prompt_hash=prompt_hash)
        return [i for i in review_ids if i not in cached]

    def put_many(
        self,
        labels: Mapping[int, frozenset[str]],
        *,
        model: str,
        prompt_hash: str,
        created_at: datetime,
    ) -> None:
        rows = [
            (int(i), model, prompt_hash, format_aspects(a), created_at.isoformat())
            for i, a in labels.items()
        ]
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.executemany(
                "INSERT OR REPLACE INTO teacher_labels "
                "(review_id, model, prompt_hash, aspects, created_at) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
