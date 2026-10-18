"""The judging batches (plan Task 9, design D7): one query's whole pool in one batch, pairs
shuffled with a seed, no rank, system or score, and only ever under data/local (they hold review
text)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from patchpulse.models.snapshot import Snapshot
from patchpulse.retrieval.judging import LONG_TEXT, MANIFEST_FILE, export_judging_batches
from patchpulse.retrieval.queries import Query

GAME = 10
LONG_ID = 99


def queries() -> list[Query]:
    def query(n: int) -> Query:
        return Query(f"q{n:03d}", GAME, "english", "aspect", f"query number {n}")

    return [query(n) for n in range(1, 7)]


def pairs() -> pd.DataFrame:
    sizes = {"q001": 40, "q002": 70, "q003": 10, "q004": 90, "q005": 30, "q006": 160}
    rows = [(qid, 1000 * n + k) for n, (qid, size) in enumerate(sizes.items()) for k in range(size)]
    rows.append(("q003", LONG_ID))
    return pd.DataFrame(sorted(rows), columns=["query_id", "review_id"])


def snapshot() -> Snapshot:
    ids = sorted(set(pairs()["review_id"]))
    texts = [("[b]long[/b] " + "word " * 500) if rid == LONG_ID else f"review {rid}" for rid in ids]
    reviews = pd.DataFrame(
        {
            "review_id": ids,
            "appid": [GAME] * len(ids),
            "date": ["2026-10-02"] * len(ids),
            "language": ["french" if rid % 2 else "english" for rid in ids],
            "voted_up": [True] * len(ids),
            "text": texts,
        }
    )
    games = pd.DataFrame({"appid": [GAME], "name": ["Toy Game"]})
    return Snapshot(reviews=reviews, games=games, patches=games)


def lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def local(tmp_path: Path) -> Path:
    return tmp_path / "data" / "local" / "retrieval_batches"


def test_batches_hide_rank_and_system(tmp_path: Path) -> None:
    batches = export_judging_batches(pairs(), queries(), snapshot(), local(tmp_path), seed=1)

    every = [line for batch in batches for line in lines(batch.path)]
    assert len(every) == len(pairs())
    assert {tuple(sorted(line)) for line in every} == {
        ("date", "game", "language", "pair", "query", "query_id", "review_id", "text")
    }
    first = next(line for line in every if line["review_id"] == 1)
    assert first["query"] == "query number 1"
    assert first["game"] == "Toy Game"
    assert first["language"] == "french"
    assert first["date"] == "2026-10-02"
    long = next(line for line in every if line["review_id"] == LONG_ID)
    assert str(long["text"]).startswith("long word")  # cleaned, as the systems saw it
    # A review past 1,900 characters is listed, so the judge gets a command to print it whole.
    holder = next(batch for batch in batches if "q003" in batch.query_ids)
    assert holder.long_pairs == (long["pair"],)
    assert len(str(long["text"])) > LONG_TEXT
    assert all(not batch.long_pairs for batch in batches if batch is not holder)


def test_batches_are_written_only_under_data_local(tmp_path: Path) -> None:
    outside = tmp_path / "reports" / "batches"

    with pytest.raises(ValueError, match="data/local"):
        export_judging_batches(pairs(), queries(), snapshot(), outside, seed=1)

    assert not outside.exists()


def test_batch_assignment_is_reproducible_with_a_seed(tmp_path: Path) -> None:
    def export(seed: int, out: Path) -> list[tuple[tuple[str, ...], list[dict[str, object]]]]:
        found = export_judging_batches(pairs(), queries(), snapshot(), out, size=150, seed=seed)
        return [(batch.query_ids, lines(batch.path)) for batch in found]

    first = export(7, local(tmp_path / "a"))
    again = export(7, local(tmp_path / "b"))
    other = export(8, local(tmp_path / "c"))

    assert first == again
    assert first != other
    for batch_queries, rows in first:
        # One query's whole pool sits in one batch, in one block; a batch holds at most 150
        # pairs unless a single query's pool is larger.
        assert [row["pair"] for row in rows] == list(range(1, len(rows) + 1))
        order = [str(row["query_id"]) for row in rows]
        assert order == sorted(order, key=batch_queries.index)
        assert len(rows) <= 150 or len(batch_queries) == 1
    placed = [qid for batch_queries, _ in first for qid in batch_queries]
    assert sorted(placed) == [q.query_id for q in queries()]
    # Within a query the pairs are shuffled, not left in id order.
    q004 = [
        int(str(row["review_id"])) for _, rows in first for row in rows if row["query_id"] == "q004"
    ]
    assert q004 != sorted(q004)


def test_a_new_export_replaces_the_old_batches(tmp_path: Path) -> None:
    out = local(tmp_path)
    export_judging_batches(pairs(), queries(), snapshot(), out, size=50, seed=1)
    many = sorted(p.name for p in out.glob("batch-*.jsonl"))

    export_judging_batches(pairs(), queries(), snapshot(), out, size=1000, seed=1)

    assert len(many) > 1
    assert sorted(p.name for p in out.glob("batch-*.jsonl")) == ["batch-001.jsonl"]


def test_a_manifest_lists_each_batch_its_queries_and_long_pairs(tmp_path: Path) -> None:
    out = local(tmp_path)

    batches = export_judging_batches(pairs(), queries(), snapshot(), out, seed=1)

    manifest = json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest == {
        "seed": 1,
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
    assert "text" not in json.dumps(manifest)  # ids and counts only
