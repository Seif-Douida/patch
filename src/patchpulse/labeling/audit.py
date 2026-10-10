"""Seif's labeling sessions (design D2, D3): the blind audit of 150 test reviews, then the blind
re-label of 30 of them at least 14 days later.

Every answer is appended and synced to disk the moment it's saved, so closing the app loses nothing
and reopening it resumes at the first unlabeled review. A correction is a new row; the latest row
for a review counts. Notes can quote the review, so they go to a local file, never to `data/gold/`.
The re-label session writes to its own file and only ever learns which ids to show, never the
first answers.
"""

from __future__ import annotations

import csv
import functools
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from patchpulse.labeling.gold import format_aspects, parse_aspects
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.snapshot import load_snapshot

SEIF_LABELS_FILE = "labels_seif.csv"
SEIF_RELABEL_FILE = "relabel_seif.csv"
NOTES_FILE = "seif_notes.csv"
RELABEL_SIZE = 30
RELABEL_MIN_DAYS = 14
_COLUMNS = ["review_id", "aspects", "unsure", "labeled_at"]


_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|>~<])")


def as_plain_text(text: str) -> str:
    """Show the review as written: escape Markdown, keep the author's line breaks."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text).replace("\n", "  \n")


# About what the app's 230 px review box shows before it scrolls.
LONG_REVIEW_CHARS = 600


def scroll_hint(text: str) -> str | None:
    """A caption for reviews longer than the box, so the end of a review isn't missed."""
    if len(text) <= LONG_REVIEW_CHARS:
        return None
    return f"Long review ({len(text):,} characters): scroll inside the box to read all of it."


def _append(path: Path, header: list[str], row: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if new:
            writer.writerow(header)
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _latest_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=_COLUMNS)
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    return rows.drop_duplicates("review_id", keep="last")


class AuditSession:
    def __init__(
        self,
        queue_ids: Sequence[int],
        labels_path: Path,
        *,
        notes_path: Path,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.queue = [int(i) for i in queue_ids]
        self.labels_path = labels_path
        self.notes_path = notes_path
        self._clock = clock

    def labels(self) -> dict[int, frozenset[str]]:
        rows = _latest_rows(self.labels_path)
        return {
            int(i): parse_aspects(a)
            for i, a in zip(rows["review_id"], rows["aspects"], strict=True)
        }

    def next_item(self) -> int | None:
        done = self.labels()
        return next((review_id for review_id in self.queue if review_id not in done), None)

    def progress(self) -> tuple[int, int]:
        done = self.labels()
        return sum(1 for review_id in self.queue if review_id in done), len(self.queue)

    def save(
        self, review_id: int, aspects: frozenset[str], *, unsure: bool, note: str = ""
    ) -> None:
        if review_id not in self.queue:
            raise ValueError(f"review {review_id} is not in this session")
        unknown = sorted(aspects - set(ASPECTS))
        if unknown:
            raise ValueError(f"unknown aspect code(s) {unknown}")
        at = self._clock().isoformat()
        _append(self.labels_path, _COLUMNS, [review_id, format_aspects(aspects), unsure, at])
        if note.strip():
            _append(
                self.notes_path, ["review_id", "note", "noted_at"], [review_id, note.strip(), at]
            )


def relabel_selection(
    audit_labels_path: Path,
    *,
    today: date,
    seed: int,
    n: int = RELABEL_SIZE,
    min_days: int = RELABEL_MIN_DAYS,
) -> list[int]:
    """The ids for the blind re-label: `n` audited reviews, available `min_days` after the audit's
    last label (spec §7.2). Only ids leave this function, never the first answers."""
    rows = _latest_rows(audit_labels_path)
    if rows.empty:
        raise ValueError("no audit labels yet: finish the audit first")
    last = max(datetime.fromisoformat(at) for at in rows["labeled_at"]).date()
    opens = last + timedelta(days=min_days)
    if today < opens:
        raise ValueError(
            f"the re-label opens on {opens.isoformat()}, {min_days} days after the audit"
        )
    ids = sorted(int(i) for i in rows["review_id"])
    chosen = np.random.default_rng(seed).choice(ids, size=min(n, len(ids)), replace=False)
    return sorted(int(i) for i in chosen)


@dataclass(frozen=True)
class ReviewView:
    """What the labeling app shows for a review, and nothing else (no split, no patch proximity)."""

    game: str
    language: str
    voted_up: bool
    text: str


@dataclass(frozen=True)
class ReviewLookup:
    reviews: dict[int, ReviewView]


@functools.lru_cache(maxsize=2)
def review_lookup(data_dir: str) -> ReviewLookup:
    """The snapshot as the app needs it, read once per process.

    Streamlit reruns the app script on every key press, but imported modules stay loaded, so this
    cache survives the reruns (a cache defined in the script itself would not).
    """
    snapshot = load_snapshot(Path(data_dir))
    names = {
        int(a): str(n) for a, n in zip(snapshot.games["appid"], snapshot.games["name"], strict=True)
    }
    columns = ["review_id", "appid", "language", "voted_up", "text"]
    return ReviewLookup(
        reviews={
            int(review_id): ReviewView(
                game=names.get(int(appid), str(appid)),
                language=str(language),
                voted_up=bool(voted_up),
                text=str(text),
            )
            for review_id, appid, language, voted_up, text in snapshot.reviews[columns].itertuples(
                index=False
            )
        }
    )
