"""The teacher-label cache (plan Task 7): a review is sent to Gemini once per model and prompt."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from patchpulse.labeling.cache import LabelCache

CREATED = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def test_cached_labels_are_never_requested_again(tmp_path: Path) -> None:
    path = tmp_path / "teacher_labels.sqlite"
    LabelCache(path).put_many(
        {1: frozenset({"performance", "stability_bugs"}), 2: frozenset()},
        model="gemma-4-31b-it",
        prompt_hash="p1",
        created_at=CREATED,
    )

    reopened = LabelCache(path)  # a later run

    assert reopened.missing([1, 2, 3], model="gemma-4-31b-it", prompt_hash="p1") == [3]
    assert reopened.get_many([1, 2, 3], model="gemma-4-31b-it", prompt_hash="p1") == {
        1: frozenset({"performance", "stability_bugs"}),
        2: frozenset(),
    }


def test_a_new_prompt_or_model_is_a_cache_miss(tmp_path: Path) -> None:
    cache = LabelCache(tmp_path / "teacher_labels.sqlite")
    cache.put_many({1: frozenset({"content"})}, model="m", prompt_hash="p1", created_at=CREATED)

    assert cache.missing([1], model="m", prompt_hash="p2") == [1]
    assert cache.missing([1], model="other", prompt_hash="p1") == [1]


def test_putting_a_label_again_replaces_it(tmp_path: Path) -> None:
    cache = LabelCache(tmp_path / "teacher_labels.sqlite")
    cache.put_many({1: frozenset({"content"})}, model="m", prompt_hash="p", created_at=CREATED)
    cache.put_many({1: frozenset({"monetization"})}, model="m", prompt_hash="p", created_at=CREATED)

    assert cache.get_many([1], model="m", prompt_hash="p") == {1: frozenset({"monetization"})}
