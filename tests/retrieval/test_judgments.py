"""Importing a judge's grades (plan Task 10, design D7): every pair of the batch graded once, 0 to
2, reasons that don't copy the review, and nothing written unless the whole file is valid.
Grades go to the committed judgments file (no text); reasons stay under data/local."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from patchpulse.retrieval.judging import JUDGMENT_COLUMNS, import_judgments

REVIEW = (
    "the game crashes every time i open the map and the developers have not fixed it after three "
    "patches which is honestly the most frustrating thing about an otherwise great game"
)
BATCH = [
    {"pair": 1, "query_id": "q001", "review_id": 11, "text": REVIEW},
    {"pair": 2, "query_id": "q001", "review_id": 12, "text": "great soundtrack"},
    {"pair": 3, "query_id": "q002", "review_id": 11, "text": REVIEW},  # one review, two queries
]


def batch(tmp_path: Path) -> Path:
    """The batch file, written on first use."""
    path = tmp_path / "data" / "local" / "retrieval_batches" / "batch-001.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True)
        full = [
            {**pair, "query": "crashes", "game": "Toy", "language": "english", "date": "2026-10-01"}
            for pair in BATCH
        ]
        path.write_text("".join(json.dumps(p) + "\n" for p in full), encoding="utf-8")
    return path


def answers(tmp_path: Path, rows: list[tuple[str, int, str, str]]) -> Path:
    path = tmp_path / "answers.csv"
    pd.DataFrame(rows, columns=["query_id", "review_id", "grade", "reason"]).to_csv(
        path, index=False, lineterminator="\n"
    )
    return path


GOOD = [
    ("q001", 11, "2", "Crashes on opening the map are the review's main complaint."),
    ("q001", 12, "0", "Only praises the music."),
    ("q002", 11, "1", "Mentions an unfixed crash in passing."),
]


def run_import(
    tmp_path: Path, rows: list[tuple[str, int, str, str]], *, version: str = "v1"
) -> None:
    import_judgments(
        batch(tmp_path),
        answers(tmp_path, rows),
        guidelines_version=version,
        judgments_path=judged(tmp_path),
        reasons_path=tmp_path / "data" / "local" / "retrieval_reasons.csv",
    )


def judged(tmp_path: Path) -> Path:
    return tmp_path / "data" / "retrieval" / "judgments.csv"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (GOOD[:2], "missing"),  # q002/11 not graded
        ([*GOOD, GOOD[0]], "more than once"),
        ([*GOOD, ("q001", 99, "0", "Not in the batch.")], "not in this batch"),
        ([*GOOD[:2], ("q003", 11, "1", "The right review under the wrong query.")], "not in"),
    ],
)
def test_import_rejects_missing_duplicate_or_foreign_pairs(
    tmp_path: Path, rows: list[tuple[str, int, str, str]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_import(tmp_path, rows)

    assert not judged(tmp_path).exists()  # nothing written from an invalid file


@pytest.mark.parametrize("grade", ["3", "-1", "1.5", "", "relevant"])
def test_import_rejects_grades_outside_0_to_2(tmp_path: Path, grade: str) -> None:
    rows = [*GOOD[:2], ("q002", 11, grade, "Mentions a crash.")]

    with pytest.raises(ValueError, match="grade"):
        run_import(tmp_path, rows)

    assert not judged(tmp_path).exists()


def test_reasons_that_copy_the_review_are_rejected(tmp_path: Path) -> None:
    # More than fifteen consecutive words of the review (case and punctuation aside) are a quote.
    copied = (
        "The game crashes every time I open the map, and the developers have not fixed it after"
    )
    fifteen = "Says: the game crashes every time I open the map and the developers have not fixed"
    assert len(fifteen.split()) == 16  # "Says:" plus fifteen copied words

    with pytest.raises(ValueError, match="copies"):
        run_import(tmp_path, [(*GOOD[0][:3], copied), *GOOD[1:]])

    assert not judged(tmp_path).exists()
    run_import(tmp_path, [(*GOOD[0][:3], fifteen), *GOOD[1:]])  # fifteen is allowed
    assert judged(tmp_path).exists()


def test_judgments_hold_no_text(tmp_path: Path) -> None:
    run_import(tmp_path, GOOD)

    rows = pd.read_csv(judged(tmp_path))
    assert list(rows.columns) == list(JUDGMENT_COLUMNS)
    assert list(rows.columns) == ["query_id", "review_id", "grade", "judge", "guidelines"]
    assert rows.to_dict("records") == [
        {"query_id": "q001", "review_id": 11, "grade": 2, "judge": "claude", "guidelines": "v1"},
        {"query_id": "q001", "review_id": 12, "grade": 0, "judge": "claude", "guidelines": "v1"},
        {"query_id": "q002", "review_id": 11, "grade": 1, "judge": "claude", "guidelines": "v1"},
    ]
    text = judged(tmp_path).read_text(encoding="utf-8")
    assert "crash" not in text
    assert "soundtrack" not in text
    reasons = pd.read_csv(tmp_path / "data" / "local" / "retrieval_reasons.csv")
    assert list(reasons.columns) == ["query_id", "review_id", "reason", "guidelines"]
    assert len(reasons) == 3


def test_a_second_import_replaces_the_same_pairs(tmp_path: Path) -> None:
    run_import(tmp_path, GOOD)
    regraded = [(q, r, "1" if (q, r) == ("q001", 12) else g, why) for q, r, g, why in GOOD]

    run_import(tmp_path, regraded, version="v2")

    rows = pd.read_csv(judged(tmp_path))
    assert len(rows) == 3
    assert set(rows["guidelines"]) == {"v2"}
    assert rows.set_index(["query_id", "review_id"]).loc[("q001", 12), "grade"] == 1


def test_the_judge_is_recorded_per_pair(tmp_path: Path) -> None:
    # Two models judged the pool (Opus, then Sonnet after a usage limit): each pair names its own.
    import_judgments(
        batch(tmp_path),
        answers(tmp_path, GOOD),
        guidelines_version="v2",
        judge="claude-sonnet-5-5",
        judgments_path=judged(tmp_path),
        reasons_path=tmp_path / "data" / "local" / "retrieval_reasons.csv",
    )

    assert set(pd.read_csv(judged(tmp_path))["judge"]) == {"claude-sonnet-5-5"}
    with pytest.raises(ValueError, match="judge"):
        import_judgments(
            batch(tmp_path),
            answers(tmp_path, GOOD),
            guidelines_version="v2",
            judge="",
            judgments_path=judged(tmp_path),
            reasons_path=tmp_path / "data" / "local" / "retrieval_reasons.csv",
        )
