"""Teacher labels already paid for (plan Task 7): one SQLite file in data/local, never committed.

A review is sent to Gemini once per (model, prompt): the prompt hash changes whenever the prompt,
the guidelines it quotes or the cleaning does, so a new prompt is a fresh cache.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
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
_BATCH = 500  # stays under SQLite's limit on query parameters


class LabelCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute(_SCHEMA)

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
