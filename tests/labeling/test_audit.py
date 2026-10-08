"""Seif's blind audit and re-label sessions (design D2, D3).

The session is the part that must be right: it saves every answer immediately, resumes where Seif
stopped, never shows him Claude's labels, and keeps his first answers out of the re-label.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from patchpulse.labeling.audit import (
    AuditSession,
    ReviewView,
    as_plain_text,
    relabel_selection,
    review_lookup,
)

APP = Path(__file__).resolve().parents[2] / "src" / "patchpulse" / "labeling" / "app.py"
AUDITED_AT = datetime(2026, 10, 10, 18, 0, tzinfo=UTC)


def session(tmp_path: Path, queue: list[int], name: str = "labels_seif.csv") -> AuditSession:
    return AuditSession(
        queue,
        tmp_path / "gold" / name,
        notes_path=tmp_path / "local" / "seif_notes.csv",
        clock=lambda: AUDITED_AT,
    )


def test_session_resumes_after_a_restart(tmp_path: Path) -> None:
    first = session(tmp_path, [5, 3, 9])
    first.save(5, frozenset({"performance"}), unsure=False)

    again = session(tmp_path, [5, 3, 9])

    assert again.next_item() == 3
    assert again.progress() == (1, 3)


def test_save_is_append_only(tmp_path: Path) -> None:
    labels = session(tmp_path, [1, 2])
    labels.save(1, frozenset({"content"}), unsure=False)
    labels.save(1, frozenset({"content", "monetization"}), unsure=True, note="DLC price?")

    rows = pd.read_csv(tmp_path / "gold" / "labels_seif.csv", keep_default_na=False)
    assert len(rows) == 2  # the correction is a new row; the latest one counts
    assert labels.labels()[1] == frozenset({"content", "monetization"})
    assert "DLC price?" not in (tmp_path / "gold" / "labels_seif.csv").read_text(encoding="utf-8")
    assert "DLC price?" in (tmp_path / "local" / "seif_notes.csv").read_text(encoding="utf-8")


def test_save_rejects_unknown_reviews_and_codes(tmp_path: Path) -> None:
    labels = session(tmp_path, [1])

    with pytest.raises(ValueError, match="not in this session"):
        labels.save(2, frozenset(), unsure=False)
    with pytest.raises(ValueError, match="graphics"):
        labels.save(1, frozenset({"graphics"}), unsure=False)


def test_finished_session_has_no_next_item(tmp_path: Path) -> None:
    labels = session(tmp_path, [1])
    labels.save(1, frozenset(), unsure=False)

    assert labels.next_item() is None


def audited(tmp_path: Path, ids: list[int]) -> Path:
    labels = session(tmp_path, ids)
    for review_id in ids:
        labels.save(review_id, frozenset({"performance"}), unsure=False)
    return tmp_path / "gold" / "labels_seif.csv"


def test_relabel_refuses_before_fourteen_days(tmp_path: Path) -> None:
    path = audited(tmp_path, list(range(1, 151)))

    with pytest.raises(ValueError, match="2026-10-24"):
        relabel_selection(path, today=AUDITED_AT.date() + timedelta(days=13), seed=1)


def test_relabel_picks_thirty_audited_reviews_after_fourteen_days(tmp_path: Path) -> None:
    path = audited(tmp_path, list(range(1, 151)))

    chosen = relabel_selection(path, today=date(2026, 10, 24), seed=1)

    assert len(chosen) == 30
    assert set(chosen) <= set(range(1, 151))
    assert chosen == relabel_selection(path, today=date(2026, 10, 24), seed=1)


def test_relabel_never_exposes_first_answers(tmp_path: Path) -> None:
    path = audited(tmp_path, list(range(1, 151)))
    chosen = relabel_selection(path, today=date(2026, 10, 24), seed=1)

    relabel = session(tmp_path, chosen, name="relabel_seif.csv")

    assert all(isinstance(review_id, int) for review_id in chosen)
    assert relabel.labels() == {}  # a fresh file: nothing from the audit
    assert relabel.next_item() == chosen[0]


def test_app_never_opens_claudes_labels() -> None:
    source = APP.read_text(encoding="utf-8")

    assert "labels_claude" not in source
    assert "CLAUDE_LABELS_FILE" not in source
    assert "gold_reasons" not in source


def test_review_text_is_shown_as_written() -> None:
    # Markdown in a review ("*bold*", "[link](url)", "# heading") must not render as formatting.
    shown = as_plain_text("a *b* [c](d)\n# not a heading_")

    assert shown == r"a \*b\* \[c\]\(d\)" + "  \n" + r"\# not a heading\_"


def test_review_lookup_reads_the_snapshot_once(tmp_path: Path) -> None:
    # The app reruns its script on every key press; the lookup must stay cached across reruns.
    reviews = pd.DataFrame(
        {
            "review_id": [7],
            "appid": [900001],
            "date": [date(2026, 9, 1)],
            "language": ["french"],
            "voted_up": [False],
            "playtime_at_review": [10],
            "timestamp_updated": [pd.Timestamp("2026-09-01")],
            "last_seen_run_id": [1],
            "text": ["Trop de bugs"],
        }
    )
    reviews.to_parquet(tmp_path / "reviews.parquet", index=False)
    pd.DataFrame({"appid": [900001], "name": ["Example Arena"], "genres": ["x"]}).to_parquet(
        tmp_path / "games.parquet", index=False
    )
    pd.DataFrame(columns=["appid", "date", "title", "patch_type"]).to_parquet(
        tmp_path / "patches.parquet", index=False
    )

    first = review_lookup(str(tmp_path))

    assert first.reviews[7] == ReviewView(
        game="Example Arena", language="french", voted_up=False, text="Trop de bugs"
    )
    assert review_lookup(str(tmp_path)) is first
