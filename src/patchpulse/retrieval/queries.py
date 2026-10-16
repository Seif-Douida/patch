"""The eval's queries (design D7): `data/retrieval/queries.csv`, our own wording, committed.

Each query names its game, as Phase 5's `search_reviews` will. Kinds: `aspect` and `event`
queries, and a few `crosslingual` ones (English text against French reviews, reported apart).
Optional filters (a window of days, the thumbs, an aspect) make a query's `filtered` variant.

The file is frozen by its SHA-256 (`queries.sha256` beside it) before any system runs; every run
and the report call `require_frozen`, so an edit afterwards stops them.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.retrieval.search import SearchFilters

COLUMNS = (
    "query_id", "appid", "language", "kind", "text", "start_day", "end_day", "voted_up", "aspect",
)  # fmt: skip
LANGUAGES = ("english", "french")
KINDS = ("aspect", "event", "crosslingual")
CROSSLINGUAL_REVIEWS = "french"
MAX_TEXT = 300


class QueryFileError(ValueError):
    """The queries file breaks a rule; the message names the query and the rule."""


class QueriesNotFrozenError(RuntimeError):
    """The queries aren't frozen, or changed after they were."""


@dataclass(frozen=True)
class Query:
    query_id: str
    appid: int
    language: str  # the query text's language: it picks the BM25 options
    kind: str
    text: str
    start_day: int | None = None
    end_day: int | None = None
    voted_up: bool | None = None
    aspect: str | None = None

    @property
    def has_extra_filters(self) -> bool:
        """Whether the `filtered` variant differs from the `base` one (game only)."""
        extras = (self.start_day, self.end_day, self.voted_up, self.aspect)
        return any(value is not None for value in extras)

    @property
    def filters(self) -> SearchFilters:
        """The `filtered` variant's filters. A cross-lingual query always searches French
        reviews: that's what it measures, in both variants."""
        return SearchFilters(
            start_day=self.start_day,
            end_day=self.end_day,
            voted_up=self.voted_up,
            aspect=self.aspect,
            review_language=CROSSLINGUAL_REVIEWS if self.kind == "crosslingual" else None,
        )

    @property
    def base_filters(self) -> SearchFilters:
        return SearchFilters(
            review_language=CROSSLINGUAL_REVIEWS if self.kind == "crosslingual" else None
        )


def _day(value: str, query_id: str) -> int | None:
    if not value:
        return None
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        raise QueryFileError(f"{query_id}: {value!r} isn't a yyyymmdd day") from None
    return int(value)


def _parse(row: dict[str, str], games: set[int]) -> Query:
    query_id = row["query_id"].strip()

    def fail(rule: str) -> QueryFileError:
        return QueryFileError(f"{query_id}: {rule}")

    if not query_id:
        raise QueryFileError("a query has no query_id")
    if not row["appid"].strip().isdigit() or int(row["appid"]) not in games:
        raise fail(f"appid {row['appid']!r} isn't a tracked game")
    if row["language"] not in LANGUAGES:
        raise fail(f"language {row['language']!r} isn't one of {', '.join(LANGUAGES)}")
    if row["kind"] not in KINDS:
        raise fail(f"kind {row['kind']!r} isn't one of {', '.join(KINDS)}")
    if row["kind"] == "crosslingual" and row["language"] != "english":
        raise fail("a cross-lingual query is written in English")
    text = row["text"].strip()
    if not text or len(text) > MAX_TEXT:
        raise fail(f"the text must have 1 to {MAX_TEXT} characters")
    start, end = _day(row["start_day"], query_id), _day(row["end_day"], query_id)
    if start is not None and end is not None and start > end:
        raise fail("start_day is after end_day")
    if row["voted_up"] not in ("", "true", "false"):
        raise fail(f"voted_up {row['voted_up']!r} isn't true, false or empty")
    if row["aspect"] and row["aspect"] not in ASPECTS:
        raise fail(f"aspect {row['aspect']!r} isn't in the taxonomy")
    return Query(
        query_id=query_id,
        appid=int(row["appid"]),
        language=row["language"],
        kind=row["kind"],
        text=text,
        start_day=start,
        end_day=end,
        voted_up=None if not row["voted_up"] else row["voted_up"] == "true",
        aspect=row["aspect"] or None,
    )


def _tracked_games() -> set[int]:
    from patchpulse.ingest.games import load_games

    return {game.appid for game in load_games(Path("config/games.yaml"))}


def load_queries(path: Path, *, games: set[int] | None = None) -> list[Query]:
    """Every query, checked: known games, languages, kinds and aspects, real days in order, unique
    ids. `games` defaults to config/games.yaml's."""
    known = games if games is not None else _tracked_games()
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise QueryFileError(f"{path.name} must have the columns {', '.join(COLUMNS)}")
        queries = [_parse(row, known) for row in reader]
    seen: set[str] = set()
    for query in queries:
        if query.query_id in seen:
            raise QueryFileError(f"{query.query_id}: the id is used twice")
        seen.add(query.query_id)
    return queries


def _hash_file(path: Path) -> Path:
    return path.with_suffix(".sha256")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_queries(path: Path, *, games: set[int] | None = None) -> str:
    """Check the file and record its SHA-256. Freezing the same file again changes nothing; an
    edited file is refused (a frozen set is never replaced in place)."""
    load_queries(path, games=games)
    digest, hash_file = _digest(path), _hash_file(path)
    if hash_file.is_file():
        frozen = hash_file.read_text(encoding="utf-8").strip()
        if frozen != digest:
            raise QueriesNotFrozenError(
                f"{path.name} is already frozen ({frozen[:12]}) and has changed since"
            )
        return digest
    hash_file.write_text(digest + "\n", encoding="utf-8")
    return digest


def require_frozen(path: Path) -> str:
    """The frozen digest, or an error if the file isn't frozen or changed since."""
    hash_file = _hash_file(path)
    if not hash_file.is_file():
        raise QueriesNotFrozenError(
            f"{path.name} isn't frozen: run `patchpulse-ml retrieval queries freeze` first"
        )
    frozen = hash_file.read_text(encoding="utf-8").strip()
    if _digest(path) != frozen:
        raise QueriesNotFrozenError(f"{path.name} changed after it was frozen ({frozen[:12]})")
    return frozen
