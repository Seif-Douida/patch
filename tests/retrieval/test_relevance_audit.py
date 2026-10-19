"""Seif's blind relevance audit (plan Task 11): 300 pool pairs drawn once at random, graded one at
a time, every answer on disk the moment it's saved, and never a sight of the judges' grades."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from patchpulse.retrieval.audit import (
    AUDIT_SEED,
    GRADE_COLUMNS,
    RelevanceAuditSession,
    audit_sample,
    read_audit,
    write_audit,
)


def pool(n: int = 1000) -> pd.DataFrame:
    return pd.DataFrame(
        {"query_id": [f"q{i % 100:03d}" for i in range(n)], "review_id": range(5000, 5000 + n)}
    )


def test_audit_sample_is_reproducible_and_ids_only(tmp_path: Path) -> None:
    first = audit_sample(pool(), size=300, seed=AUDIT_SEED)
    again = audit_sample(pool().sample(frac=1, random_state=1), size=300, seed=AUDIT_SEED)
    other = audit_sample(pool(), size=300, seed=AUDIT_SEED + 1)

    assert first == again  # independent of the pool's row order
    assert first != other
    assert len(first) == len(set(first)) == 300
    known = set(zip(pool()["query_id"], pool()["review_id"], strict=True))
    assert set(first) <= known
    assert first != sorted(first)  # shown in random order, not grouped by query

    path = tmp_path / "data" / "retrieval" / "audit.csv"
    write_audit(first, path)
    assert path.read_text(encoding="utf-8").splitlines()[0] == "query_id,review_id"
    assert read_audit(path) == first
    write_audit(first, path)  # the same draw again changes nothing
    with pytest.raises(ValueError, match="already drawn"):
        write_audit(other, path)  # drawn once: a different sample never replaces it
    with pytest.raises(ValueError, match="only 1000"):
        audit_sample(pool(), size=1001, seed=AUDIT_SEED)


def test_audit_sample_leaves_out_pairs_seif_has_seen_graded() -> None:
    # The pilot's pairs, shown to Seif with the pilot judge's grades, can't be audited blind.
    seen = {("q000", 5000), ("q001", 5001), ("q002", 5002)}

    drawn = audit_sample(pool(), size=997, seed=AUDIT_SEED, exclude=seen)

    assert not seen & set(drawn)
    assert len(drawn) == 997  # every other pair is still drawable
    with pytest.raises(ValueError, match="only 997"):
        audit_sample(pool(), size=998, seed=AUDIT_SEED, exclude=seen)


def test_session_saves_resumes_and_never_holds_claudes_grade(tmp_path: Path) -> None:
    queue = [("q001", 11), ("q002", 12), ("q001", 13)]
    grades = tmp_path / "data" / "retrieval" / "grades_seif.csv"
    # The judges' grades sit next to Seif's file; the session must never read them.
    grades.parent.mkdir(parents=True)
    (grades.parent / "judgments.csv").write_text("query_id,review_id,grade\nq001,11,0\n")
    times = iter(datetime(2026, 10, 20, 9, tzinfo=UTC) + timedelta(minutes=m) for m in range(9))
    session = RelevanceAuditSession(queue, grades, clock=lambda: next(times))

    assert session.next_item() == ("q001", 11)
    session.save(("q001", 11), 2, unsure=False)
    session.save(("q002", 12), 0, unsure=True)

    resumed = RelevanceAuditSession(queue, grades)
    assert resumed.next_item() == ("q001", 13)
    assert resumed.progress() == (2, 3)
    resumed.save(("q001", 11), 1, unsure=False)  # a correction: the latest row wins
    assert resumed.grades() == {("q001", 11): 1, ("q002", 12): 0}
    rows = pd.read_csv(grades)
    assert list(rows.columns) == list(GRADE_COLUMNS)
    assert list(GRADE_COLUMNS) == ["query_id", "review_id", "grade", "unsure", "graded_at"]
    assert len(rows) == 3
    assert not any("judg" in str(value) for value in vars(resumed).values())
    with pytest.raises(ValueError, match="0, 1 or 2"):
        resumed.save(("q001", 13), 3, unsure=False)
    with pytest.raises(ValueError, match="not in this audit"):
        resumed.save(("q009", 99), 1, unsure=False)
