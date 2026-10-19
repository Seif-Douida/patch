"""Seif's blind relevance audit (design D7, plan Task 11): 300 pool pairs, graded 0, 1 or 2.

The pairs are a simple random sample of the pool, drawn once, saved by id to
`data/retrieval/audit.csv` and shown in the drawn order (not grouped by query). Every grade is
appended and synced to disk the moment it's saved, so closing the app loses nothing and reopening
it resumes at the first ungraded pair; a correction is a new row, and the latest row for a pair
counts. The session reads only the audit's ids and Seif's own grades, never the judges'.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from patchpulse.labeling.audit import append_synced
from patchpulse.models.snapshot import load_snapshot
from patchpulse.models.text import clean_text
from patchpulse.retrieval.queries import load_queries

AUDIT_PATH = Path("data/retrieval/audit.csv")  # committed: ids only
GRADES_PATH = Path("data/retrieval/grades_seif.csv")  # committed: ids and grades only
AUDIT_SIZE = 300
AUDIT_SEED = 20261016
GRADE_COLUMNS = ("query_id", "review_id", "grade", "unsure", "graded_at")
GRADES = (0, 1, 2)

Pair = tuple[str, int]


def audit_sample(
    pool: pd.DataFrame,
    *,
    size: int = AUDIT_SIZE,
    seed: int = AUDIT_SEED,
    exclude: Collection[Pair] = frozenset(),
) -> list[Pair]:
    """`size` pool pairs drawn at random, in the drawn order, whatever the pool's row order.
    `exclude` holds pairs Seif has already seen graded (the pilot's), which can't be audited
    blind."""
    pairs = sorted(
        {(str(q), int(r)) for q, r in zip(pool["query_id"], pool["review_id"], strict=True)}
        - set(exclude)
    )
    if size > len(pairs):
        raise ValueError(f"the pool has only {len(pairs)} pairs to audit, not {size}")
    chosen = np.random.default_rng(seed).choice(len(pairs), size=size, replace=False)
    return [pairs[int(i)] for i in chosen]


def read_audit(path: Path = AUDIT_PATH) -> list[Pair]:
    rows = pd.read_csv(path, dtype={"query_id": str})
    return [(str(q), int(r)) for q, r in zip(rows["query_id"], rows["review_id"], strict=True)]


def write_audit(pairs: Sequence[Pair], path: Path = AUDIT_PATH) -> None:
    """Save the draw. Drawn once: writing the same draw again is a no-op, a different one is
    refused (the audit can't be redrawn after grades exist)."""
    if path.is_file():
        if read_audit(path) == list(pairs):
            return
        raise ValueError(f"{path.name} is already drawn; delete it by hand to draw again")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(pairs), columns=["query_id", "review_id"]).to_csv(
        path, index=False, lineterminator="\n"
    )


class RelevanceAuditSession:
    def __init__(
        self,
        queue: Sequence[Pair],
        grades_path: Path,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.queue = [(str(q), int(r)) for q, r in queue]
        self.grades_path = grades_path
        self._clock = clock

    def grades(self) -> dict[Pair, int]:
        if not self.grades_path.exists():
            return {}
        rows = pd.read_csv(self.grades_path, dtype={"query_id": str})
        latest = rows.drop_duplicates(["query_id", "review_id"], keep="last")
        return {
            (str(q), int(r)): int(g)
            for q, r, g in zip(
                latest["query_id"], latest["review_id"], latest["grade"], strict=True
            )
        }

    def next_item(self) -> Pair | None:
        done = self.grades()
        return next((pair for pair in self.queue if pair not in done), None)

    def progress(self) -> tuple[int, int]:
        done = self.grades()
        return sum(1 for pair in self.queue if pair in done), len(self.queue)

    def save(self, pair: Pair, grade: int, *, unsure: bool) -> None:
        if pair not in self.queue:
            raise ValueError(f"pair {pair[0]}/{pair[1]} is not in this audit")
        if grade not in GRADES:
            raise ValueError(f"grade {grade!r} isn't 0, 1 or 2")
        at = self._clock().isoformat()
        append_synced(self.grades_path, list(GRADE_COLUMNS), [pair[0], pair[1], grade, unsure, at])


@dataclass(frozen=True)
class PairView:
    """What the audit shows for a pair, and nothing else: no grade, rank, system or score."""

    query: str
    game: str
    language: str  # the review's (Steam's tag)
    date: str
    text: str  # cleaned and cut as the systems and the judges saw it


@functools.lru_cache(maxsize=2)
def pair_lookup(data_dir: str, queries_path: str, audit_path: str) -> dict[Pair, PairView]:
    """The audit pairs as the app shows them, read once per process (Streamlit reruns the script
    on every key press; this module stays loaded)."""
    queries = {query.query_id: query for query in load_queries(Path(queries_path))}
    snapshot = load_snapshot(Path(data_dir))
    names = dict(zip(snapshot.games["appid"], snapshot.games["name"], strict=True))
    reviews = snapshot.reviews.set_index("review_id")
    views: dict[Pair, PairView] = {}
    for query_id, review_id in read_audit(Path(audit_path)):
        review, query = reviews.loc[review_id], queries[query_id]
        views[(query_id, review_id)] = PairView(
            query=query.text,
            game=str(names.get(query.appid, query.appid)),
            language=str(review["language"]),
            date=str(pd.Timestamp(str(review["date"])).date()),
            text=clean_text(str(review["text"])),
        )
    return views
