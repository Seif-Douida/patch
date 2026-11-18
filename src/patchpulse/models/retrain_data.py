"""The retrain's data (Phase 7 design D5, plan Task 9), from git and SQL only.

- **Train:** the silver set's training labels (`data/silver/labels.csv`: ids and labels, no text)
  plus the teacher spot-check's labels outside the recent holdout (`raw.teacher_label`).
- **Dev and test:** the gold set's committed files (`data/gold/`); dev tunes the thresholds, test
  is reported only.
- **Recent:** the spot-check's holdout fifth (`is_holdout`) labeled in the last 8 weeks: what a
  promotion is decided on.

Each review's text is its latest version in `raw.steam_review`, cleaned by the scorer as usual; a
review no longer there is dropped and counted. Runs in the jobs image: no pandas.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
from sqlalchemy import Connection, bindparam, text

from patchpulse.labeling.taxonomy import ASPECTS, parse_aspects
from patchpulse.models.split import Split
from patchpulse.monitoring.store import read_teacher_labels
from patchpulse.pipeline.spotcheck import is_holdout

SILVER_LABELS = Path("data") / "silver" / "labels.csv"
GOLD_DIR = Path("data") / "gold"
SILVER_COLUMNS = ("review_id", "aspects", "split", "model", "prompt_hash")
RECENT_WEEKS = 8
_ID_BATCH = 1_000

_LATEST = """
SELECT r.recommendation_id, r.language, r.review_text
FROM raw.steam_review AS r
WHERE r.recommendation_id IN :ids
  AND NOT EXISTS (
      SELECT 1 FROM raw.steam_review AS newer
      WHERE newer.recommendation_id = r.recommendation_id
        AND newer.timestamp_updated > r.timestamp_updated)
"""


@dataclass(frozen=True)
class RetrainData:
    train: Split
    dev: Split
    test: Split
    recent: Split
    languages: dict[str, list[str]]  # split name -> each review's language, in the split's order
    missing: int  # labeled reviews no longer in SQL, dropped
    teachers: dict[str, int]  # training labels per teacher model


def write_silver_csv(rows: Iterable[Mapping[str, object]], path: Path = SILVER_LABELS) -> int:
    """The silver labels as committed: one row per review, sorted by id."""
    ordered = sorted(rows, key=lambda row: int(str(row["review_id"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SILVER_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: row[column] for column in SILVER_COLUMNS} for row in ordered)
    return len(ordered)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _texts(connection: Connection, ids: Sequence[int]) -> dict[int, tuple[str, str]]:
    """Review id -> (language, text) of its latest version."""
    statement = text(_LATEST).bindparams(bindparam("ids", expanding=True))
    found: dict[int, tuple[str, str]] = {}
    unique = sorted(set(ids))
    for start in range(0, len(unique), _ID_BATCH):
        chunk = unique[start : start + _ID_BATCH]
        for rid, language, body in connection.execute(statement, {"ids": chunk}):
            found[int(rid)] = (str(language), str(body))
    return found


def _split(
    labeled: Sequence[tuple[int, frozenset[str]]], texts: Mapping[int, tuple[str, str]]
) -> tuple[Split, list[str]]:
    kept = [(rid, aspects) for rid, aspects in labeled if rid in texts]
    labels = np.array([[code in aspects for code in ASPECTS] for _, aspects in kept], dtype=bool)
    split = Split(
        ids=[rid for rid, _ in kept],
        texts=[texts[rid][1] for rid, _ in kept],
        labels=labels.reshape(len(kept), len(ASPECTS)),
    )
    return split, [texts[rid][0] for rid, _ in kept]


def load_retrain_data(
    connection: Connection,
    *,
    today: date,
    silver_csv: Path = SILVER_LABELS,
    gold_dir: Path = GOLD_DIR,
) -> RetrainData:
    silver = _read_csv(silver_csv)
    silver_train = [
        (int(r["review_id"]), parse_aspects(r["aspects"])) for r in silver if r["split"] == "train"
    ]
    teachers: dict[str, int] = {}
    for row in silver:
        if row["split"] == "train":
            teachers[row["model"]] = teachers.get(row["model"], 0) + 1

    spot = read_teacher_labels(connection, source="spotcheck")
    spot_train = [
        (s.recommendation_id, s.aspects) for s in spot if not is_holdout(s.recommendation_id)
    ]
    for label in spot:
        if not is_holdout(label.recommendation_id):
            teachers[label.model] = teachers.get(label.model, 0) + 1
    since = datetime.combine(today - timedelta(weeks=RECENT_WEEKS), time())
    recent = read_teacher_labels(connection, source="spotcheck", since=since)
    recent_holdout = sorted(
        (s.recommendation_id, s.aspects) for s in recent if is_holdout(s.recommendation_id)
    )

    sample = _read_csv(gold_dir / "sample.csv")
    gold = {
        int(r["review_id"]): parse_aspects(r["aspects"])
        for r in _read_csv(gold_dir / "labels_claude.csv")
    }
    dev = sorted(
        (int(r["review_id"]), gold[int(r["review_id"])]) for r in sample if r["split"] == "dev"
    )
    test = sorted(
        (int(r["review_id"]), gold[int(r["review_id"])]) for r in sample if r["split"] == "test"
    )

    train = sorted(dict([*silver_train, *spot_train]).items())
    wanted = [rid for rid, _ in [*train, *dev, *test, *recent_holdout]]
    texts = _texts(connection, wanted)
    named = (("train", train), ("dev", dev), ("test", test), ("recent", recent_holdout))
    splits = {name: _split(rows, texts) for name, rows in named}
    return RetrainData(
        train=splits["train"][0],
        dev=splits["dev"][0],
        test=splits["test"][0],
        recent=splits["recent"][0],
        languages={name: langs for name, (_, langs) in splits.items()},
        missing=len(set(wanted) - set(texts)),
        teachers=dict(sorted(teachers.items())),
    )
