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
    pair_view,
    read_audit,
    unread_pairs,
    write_audit,
)
from patchpulse.retrieval.queries import Query


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


def test_unread_pairs_are_other_languages_graded_1_unsure(tmp_path: Path) -> None:
    # Seif's default (19 Oct) for a review he couldn't read: 1 with unsure ticked. That isn't a
    # relevance judgment, so the agreement gate leaves those pairs out.
    queue = [("q001", 11), ("q001", 12), ("q002", 13), ("q002", 14), ("q003", 15), ("q003", 16)]
    session = RelevanceAuditSession(queue, tmp_path / "grades_seif.csv")
    session.save(("q001", 11), 1, unsure=True)  # Chinese, the default: unread
    session.save(("q001", 12), 1, unsure=True)  # French, read and unsure: kept
    session.save(("q002", 13), 2, unsure=False)  # German, translated: kept
    session.save(("q002", 14), 0, unsure=True)  # Russian, a real grade with doubt: kept
    session.save(("q003", 15), 1, unsure=True)
    session.save(("q003", 15), 2, unsure=False)  # a correction: the latest row counts
    session.save(("q003", 16), 1, unsure=False)  # Spanish, not unsure: kept
    languages = {11: "schinese", 12: "french", 13: "german", 14: "russian", 15: "polish"}
    languages[16] = "spanish"

    assert session.unsure() == {("q001", 11), ("q001", 12), ("q002", 14)}
    assert unread_pairs(session.grades(), session.unsure(), languages) == {("q001", 11)}
    with pytest.raises(ValueError, match="no language for review 11"):
        unread_pairs(session.grades(), session.unsure(), {})


def test_pair_view_shows_a_later_edit() -> None:
    # A review written in January and edited after an October release: its text is as of the edit.
    query = Query("q079", 292030, "english", "event", "the Remastered release")

    def review(updated: str) -> pd.Series:
        return pd.Series(
            {
                "language": "english",
                "date": "2026-01-12",
                "timestamp_updated": pd.Timestamp(updated),
                "text": "[b]Now remastered[/b]",
            }
        )

    edited = pair_view(query, "The Witcher 3", review("2026-10-01 15:32"))
    same_day = pair_view(query, "The Witcher 3", review("2026-01-12 20:05"))

    assert (edited.date, edited.edited) == ("2026-01-12", "2026-10-01")
    assert edited.text == "Now remastered"  # cleaned, as the judges saw it
    assert same_day.edited is None
