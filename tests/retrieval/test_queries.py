"""The eval's queries (plan Task 8, design D7): a validated file, frozen by its SHA-256 before any
system runs, so an edit afterwards is caught."""

from __future__ import annotations

from pathlib import Path

import pytest

from patchpulse.retrieval.queries import (
    QueriesNotFrozenError,
    QueryFileError,
    freeze_queries,
    load_queries,
    require_frozen,
)
from patchpulse.retrieval.search import SearchFilters

HEADER = "query_id,appid,language,kind,text,start_day,end_day,voted_up,aspect\n"
GOOD = (
    HEADER
    + "q001,553850,english,aspect,crashes after the latest patch,,,,\n"
    + "q002,553850,french,event,avis sur la mise a jour,20260901,20260915,false,performance\n"
    + 'q003,1091500,english,crosslingual,"stutter, frame drops",,,,\n'
)
GAMES = {553850, 1091500}


def write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "queries.csv"
    path.write_text(content, encoding="utf-8", newline="")
    return path


def test_queries_file_is_validated(tmp_path: Path) -> None:
    queries = load_queries(write(tmp_path, GOOD), games=GAMES)

    assert [q.query_id for q in queries] == ["q001", "q002", "q003"]
    assert queries[0].filters == SearchFilters()
    assert not queries[0].has_extra_filters
    assert queries[1].filters == SearchFilters(
        start_day=20260901, end_day=20260915, voted_up=False, aspect="performance"
    )
    assert queries[1].has_extra_filters
    # Cross-lingual: English text against French reviews, part of the query in every variant.
    assert queries[2].text == "stutter, frame drops"
    assert queries[2].filters == SearchFilters(review_language="french")
    assert not queries[2].has_extra_filters
    bad_rows = {
        "unknown game": "q009,42,english,aspect,lag,,,,",
        "language": "q009,553850,german,aspect,lag,,,,",
        "kind": "q009,553850,english,other,lag,,,,",
        "duplicate": "q001,553850,english,aspect,lag,,,,",
        "aspect": "q009,553850,english,aspect,lag,,,,graphics",
        "window order": "q009,553850,english,event,lag,20260915,20260901,,",
        "day": "q009,553850,english,event,lag,20261341,,,",
        "thumbs": "q009,553850,english,aspect,lag,,,maybe,",
        "empty text": "q009,553850,english,aspect,  ,,,,",
        "cross-lingual in French": "q009,553850,french,crosslingual,lag,,,,",
    }
    for problem, row in bad_rows.items():
        with pytest.raises(QueryFileError, match=r"q00[19]"):
            load_queries(write(tmp_path, GOOD + row + "\n"), games=GAMES)
        assert problem  # each row above breaks exactly one rule
    with pytest.raises(QueryFileError, match="columns"):
        load_queries(write(tmp_path, GOOD.replace("aspect\n", "topic\n", 1)), games=GAMES)


def test_freeze_then_any_edit_is_detected(tmp_path: Path) -> None:
    path = write(tmp_path, GOOD)

    digest = freeze_queries(path, games=GAMES)

    assert require_frozen(path) == digest
    assert (tmp_path / "queries.sha256").read_text(encoding="utf-8").strip() == digest
    path.write_text(GOOD.replace("crashes", "crashing"), encoding="utf-8", newline="")
    with pytest.raises(QueriesNotFrozenError, match="changed"):
        require_frozen(path)
    path.write_text(GOOD + "\n", encoding="utf-8", newline="")  # even a blank line
    with pytest.raises(QueriesNotFrozenError):
        require_frozen(path)


def test_runs_refuse_unfrozen_or_changed_queries(tmp_path: Path) -> None:
    path = write(tmp_path, GOOD)

    with pytest.raises(QueriesNotFrozenError, match="freeze"):
        require_frozen(path)
    digest = freeze_queries(path, games=GAMES)
    assert freeze_queries(path, games=GAMES) == digest  # the same file: nothing changes
    path.write_text(GOOD.replace("crashes", "crashing"), encoding="utf-8", newline="")
    with pytest.raises(QueriesNotFrozenError, match="already frozen"):
        freeze_queries(path, games=GAMES)  # an edited file is never re-frozen in place
