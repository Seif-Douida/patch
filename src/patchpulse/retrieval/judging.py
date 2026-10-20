"""The judging batches (design D7): the pool's pairs, for fresh Claude subagents to grade.

A batch holds whole queries, so a judge grades one query's whole pool at once and in one block;
within a query the pairs are shuffled with a seed, and no rank, system or score is shown. Each pair
carries the query, the game, the review's language and date (event queries are about a time) and
the review text cleaned as the systems saw it.

The batches hold review text, so they're written under `data/local/` only, never committed. A
review longer than 1,900 characters makes a JSON line longer than a file reader shows in one
piece, so `manifest.json` lists those pairs per batch: the judge gets a command to print them
whole (as in Phase 2).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from patchpulse.models.snapshot import Snapshot
from patchpulse.models.text import clean_text
from patchpulse.retrieval.queries import Query

BATCH_SIZE = 150
LONG_TEXT = 1_900
JUDGING_SEED = 20261018
MANIFEST_FILE = "manifest.json"  # each batch's file, queries, pair count and long pairs

# The judges' answers: one CSV per batch, every pair graded 0 (not relevant), 1 (partly) or 2.
ANSWER_COLUMNS = ("query_id", "review_id", "grade", "reason")
GRADES = ("0", "1", "2")
MAX_COPIED_WORDS = 15  # a reason may echo a phrase of the review, never quote a passage
JUDGE = "claude"
JUDGMENTS_PATH = Path("data/retrieval/judgments.csv")  # committed: ids and grades only
JUDGMENT_COLUMNS = ("query_id", "review_id", "grade", "judge", "guidelines")
REASONS_PATH = Path("data/local/retrieval_reasons.csv")  # local: a reason describes the review
REASON_COLUMNS = ("query_id", "review_id", "reason", "guidelines")
_KEY = ["query_id", "review_id"]
_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class JudgingBatch:
    path: Path
    query_ids: tuple[str, ...]
    pairs: int
    long_pairs: tuple[int, ...]  # pair numbers whose text is longer than LONG_TEXT


def _require_local(out_dir: Path) -> None:
    parts = out_dir.resolve().parts
    if not any(parts[i : i + 2] == ("data", "local") for i in range(len(parts) - 1)):
        raise ValueError(
            f"judging batches hold review text: write them under data/local, not {out_dir}"
        )


def _pack(sizes: Sequence[tuple[str, int]], size: int) -> list[list[str]]:
    """Whole queries, in the given order, into batches of at most `size` pairs (a query larger
    than `size` gets a batch of its own)."""
    batches: list[list[str]] = []
    held = 0
    for query_id, n in sizes:
        if not batches or held + n > size:
            batches.append([])
            held = 0
        batches[-1].append(query_id)
        held += n
    return batches


def export_judging_batches(
    pairs: pd.DataFrame,
    queries: Sequence[Query],
    snapshot: Snapshot,
    out_dir: Path,
    *,
    size: int = BATCH_SIZE,
    seed: int,
) -> list[JudgingBatch]:
    """Write `batch-NNN.jsonl` files to `out_dir` (under data/local), replacing any earlier ones."""
    _require_local(out_dir)
    by_id = {query.query_id: query for query in queries}
    unknown = sorted(set(pairs["query_id"]) - set(by_id))
    if unknown:
        raise ValueError(f"pairs for unknown queries: {', '.join(unknown[:10])}")
    reviews = snapshot.reviews.set_index("review_id")
    names = snapshot.games.set_index("appid")["name"]
    rng = np.random.default_rng(seed)

    query_ids = sorted(set(pairs["query_id"]))
    order = [query_ids[i] for i in rng.permutation(len(query_ids))]
    grouped = {
        str(qid): sorted(int(r) for r in rows)
        for qid, rows in pairs.groupby("query_id")["review_id"]
    }
    packed = _pack([(qid, len(grouped[qid])) for qid in order], size)

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("batch-*.jsonl"):
        old.unlink()
    batches = []
    for number, batch_queries in enumerate(packed, start=1):
        path = out_dir / f"batch-{number:03d}.jsonl"
        long_pairs: list[int] = []
        pair = 0
        with path.open("w", encoding="utf-8", newline="\n") as out:
            for query_id in batch_queries:
                query = by_id[query_id]
                ids = grouped[query_id]
                for review_id in (ids[i] for i in rng.permutation(len(ids))):
                    review = reviews.loc[review_id]
                    text = clean_text(str(review["text"]))
                    pair += 1
                    if len(text) > LONG_TEXT:
                        long_pairs.append(pair)
                    line = {
                        "pair": pair,
                        "query_id": query_id,
                        "query": query.text,
                        "game": str(names[query.appid]),
                        "review_id": int(review_id),
                        "language": str(review["language"]),
                        "date": str(pd.Timestamp(review["date"]).date()),
                        # The text is as of its last edit (guidelines v3).
                        "edited": str(pd.Timestamp(review["timestamp_updated"]).date()),
                        "text": text,
                    }
                    out.write(json.dumps(line, ensure_ascii=False) + "\n")
        batches.append(JudgingBatch(path, tuple(batch_queries), pair, tuple(long_pairs)))
    manifest = {
        "seed": seed,
        "batches": [
            {
                "file": batch.path.name,
                "query_ids": list(batch.query_ids),
                "pairs": batch.pairs,
                "long_pairs": list(batch.long_pairs),
            }
            for batch in batches
        ],
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return batches


def copied_words(reason: str, review: str) -> int:
    """The longest run of consecutive words the reason shares with the review, ignoring case and
    punctuation."""
    said, written = _WORD.findall(reason.lower()), _WORD.findall(review.lower())
    longest = 0
    previous = [0] * (len(written) + 1)
    for word in said:
        current = [0] * (len(written) + 1)
        for j, other in enumerate(written, start=1):
            if word == other:
                current[j] = previous[j - 1] + 1
                longest = max(longest, current[j])
        previous = current
    return longest


def _merge(path: Path, new: pd.DataFrame) -> None:
    """Add `new` to the CSV at `path`, replacing the rows of the same pairs."""
    if path.is_file():
        old = pd.read_csv(path, dtype={"query_id": str}, keep_default_na=False)
        replaced = old.set_index(_KEY).index.isin(new.set_index(_KEY).index)
        new = pd.concat([old[~replaced], new])
    path.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values(_KEY).to_csv(path, index=False, lineterminator="\n")


def import_judgments(
    batch_file: Path,
    answer_file: Path,
    *,
    guidelines_version: str,
    judge: str = JUDGE,
    judgments_path: Path = JUDGMENTS_PATH,
    reasons_path: Path = REASONS_PATH,
) -> int:
    """Check a judge's answers against its batch, then merge them in; returns the pairs graded.

    Every pair of the batch must be graded exactly once, 0, 1 or 2, and no reason may copy more
    than 15 consecutive words of its review. Nothing is written unless the whole file passes.
    Grades go to `judgments_path` (committed, no text), each with the `judge` that gave it (the
    model, when more than one judged the pool); reasons to `reasons_path` (local).
    """
    if not judge.strip():
        raise ValueError("name the judge (the model that graded the batch)")
    with batch_file.open(encoding="utf-8") as file:
        texts = {
            (str(pair["query_id"]), int(pair["review_id"])): str(pair["text"])
            for pair in map(json.loads, file)
        }
    rows = pd.read_csv(answer_file, dtype=str, keep_default_na=False)
    if tuple(rows.columns) != ANSWER_COLUMNS:
        raise ValueError(f"{answer_file.name} must have the columns {', '.join(ANSWER_COLUMNS)}")
    try:
        keys = [
            (q.strip(), int(r)) for q, r in zip(rows["query_id"], rows["review_id"], strict=True)
        ]
    except ValueError:
        raise ValueError(f"{answer_file.name}: a review_id isn't a number") from None

    def show(pairs: Sequence[tuple[str, int]]) -> str:
        return ", ".join(f"{q}/{r}" for q, r in pairs[:10])

    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        raise ValueError(f"pairs graded more than once: {show(repeated)}")
    extra, missing = sorted(set(keys) - set(texts)), sorted(set(texts) - set(keys))
    if extra:
        raise ValueError(f"pairs not in this batch: {show(extra)}")
    if missing:
        raise ValueError(f"missing pairs: {show(missing)}")
    for key, grade, reason in zip(keys, rows["grade"], rows["reason"], strict=True):
        if grade.strip() not in GRADES:
            raise ValueError(f"{key[0]}/{key[1]}: grade {grade!r} isn't 0, 1 or 2")
        copied = copied_words(reason, texts[key])
        if copied > MAX_COPIED_WORDS:
            raise ValueError(
                f"{key[0]}/{key[1]}: the reason copies {copied} consecutive words of the review "
                f"(at most {MAX_COPIED_WORDS})"
            )

    query_ids = [q for q, _ in keys]
    review_ids = [r for _, r in keys]
    _merge(
        judgments_path,
        pd.DataFrame(
            {
                "query_id": query_ids,
                "review_id": review_ids,
                "grade": [int(g) for g in rows["grade"]],
                "judge": judge.strip(),
                "guidelines": guidelines_version,
            },
            columns=list(JUDGMENT_COLUMNS),
        ),
    )
    _merge(
        reasons_path,
        pd.DataFrame(
            {
                "query_id": query_ids,
                "review_id": review_ids,
                "reason": list(rows["reason"]),
                "guidelines": guidelines_version,
            },
            columns=list(REASON_COLUMNS),
        ),
    )
    return len(keys)
