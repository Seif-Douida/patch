"""Review text cleaning shared by the teacher, training and Azure scoring (plan Task 8)."""

from __future__ import annotations

import pytest

from patchpulse.models.text import CLEANING_VERSION, clean_text


@pytest.mark.parametrize(
    ("raw", "cleaned"),
    [
        ("", ""),
        ("   \n\t  ", ""),
        ("🔥🔥🔥", "🔥🔥🔥"),
        (
            "[h1]Verdict[/h1]\n[b]Great[/b] [i]combat[/i], [strike]bad[/strike] fine",
            "Verdict\nGreat combat, bad fine",
        ),
        (
            "[url=https://example.com/guide]my guide[/url] and [spoiler]the end[/spoiler]",
            "my guide and the end",
        ),
        ("[list][*]fast[*]pretty[/list]", "fast\npretty"),  # block tags become line breaks
        ("[x] Good  [ ] Bad", "[x] Good [ ] Bad"),  # checklists aren't BBCode
        (
            "see https://store.steampowered.com/app/1 or www.example.com now",
            "see <url> or <url> now",
        ),
        ("Отличная игра,   но вылетает", "Отличная игра, но вылетает"),
        ("优化很差\r\n\r\n\r\n但是剧情很好", "优化很差\n但是剧情很好"),
        ("line one  \n\n\n   line two", "line one\nline two"),
    ],
)
def test_cleaning_handles_odd_reviews(raw: str, cleaned: str) -> None:
    assert clean_text(raw) == cleaned
    assert clean_text(raw) == clean_text(raw)  # deterministic


def test_long_reviews_are_cut_at_a_word_boundary() -> None:
    text = "word " * 1_600  # 8,000 characters

    cleaned = clean_text(text)

    assert len(cleaned) <= 2_000
    assert cleaned.endswith("word")
    assert clean_text("x" * 8_000) == "x" * 2_000  # no space to cut at: a hard cut
    assert len(clean_text("长" * 8_000, max_chars=500)) == 500


def test_cleaning_version_is_pinned() -> None:
    # Changing the cleaning changes every model's input: bump the version with it (it's part of
    # the teacher's prompt hash and is recorded with every trained model).
    assert CLEANING_VERSION == "c1"
