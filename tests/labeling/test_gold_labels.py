"""Batches out to the labeling subagents, labels back in (design D2).

What leaves: id, game, language, thumbs and text, nothing that could prime the annotator (split,
patch proximity). What comes back is checked strictly before it becomes gold: every id exactly
once, known aspect codes only. Labels are committed; the annotator's reasons stay local because
they can quote reviews (spec C4).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from patchpulse.labeling.gold import (
    batch_ids,
    export_batch,
    import_labels,
    pilot_ids,
    read_labels,
)
from patchpulse.models.snapshot import Snapshot


def gold_sample(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": range(1, n + 1),
            "appid": [100 + i % 3 for i in range(n)],
            "language": ["english" if i % 3 else "french" for i in range(n)],
            "voted_up": [i % 2 == 0 for i in range(n)],
            "length_bucket": "short",
            "near_patch": [i % 5 == 0 for i in range(n)],
            "split": ["dev" if i < 12 else "test" for i in range(n)],
        }
    )


def snapshot(n: int = 40) -> Snapshot:
    reviews = pd.DataFrame(
        {
            "review_id": range(1, n + 1),
            "appid": [100 + i % 3 for i in range(n)],
            "date": date(2026, 9, 1),
            "language": "english",
            "voted_up": [i % 2 == 0 for i in range(n)],
            "playtime_at_review": 10,
            "timestamp_updated": pd.Timestamp("2026-09-01"),
            "last_seen_run_id": 1,
            "text": [f'review "{i}"\nwith a newline' for i in range(1, n + 1)],
        }
    )
    games = pd.DataFrame(
        {"appid": [100, 101, 102], "name": ["Arena", "Colony", "Drift"], "genres": "x"}
    )
    patches = pd.DataFrame(columns=["appid", "date", "title", "patch_type"])
    return Snapshot(reviews=reviews, games=games, patches=patches)


def write_csv(path: Path, rows: Sequence[tuple[object, str, str]]) -> Path:
    pd.DataFrame(rows, columns=["id", "aspects", "reason"]).to_csv(path, index=False)
    return path


def run_import(
    tmp_path: Path, rows: Sequence[tuple[object, str, str]], expected: list[int]
) -> None:
    import_labels(
        write_csv(tmp_path / "labels.csv", rows),
        expected,
        annotator="claude",
        guidelines_version="v1",
        gold_dir=tmp_path / "gold",
        reasons_path=tmp_path / "local" / "gold_reasons.csv",
    )


def test_exported_batch_hides_split_and_patch_proximity(tmp_path: Path) -> None:
    path = tmp_path / "batch.jsonl"

    count = export_batch(gold_sample(), snapshot(), [1, 2, 3], path)

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert count == 3
    assert set(lines[0]) == {"id", "game", "language", "thumbs", "text"}
    assert lines[0]["game"] == "Arena"
    assert [line["thumbs"] for line in lines] == ["up", "down", "up"]
    assert lines[1]["text"] == 'review "2"\nwith a newline'


def test_batches_cover_the_sample_once_and_the_pilot_is_dev() -> None:
    sample = gold_sample()

    batches = [batch_ids(sample, k, seed=1) for k in (1, 2, 3, 4)]
    pilot = pilot_ids(sample, seed=1, n=6)

    assert sorted(i for batch in batches for i in batch) == list(range(1, 41))
    assert set(sample.set_index("review_id").loc[pilot, "split"]) == {"dev"}


def test_import_writes_labels_and_keeps_reasons_local(tmp_path: Path) -> None:
    rows = [(1, "performance|stability_bugs", "stutters, crashes"), (2, "none", "just praise")]

    run_import(tmp_path, rows, [1, 2])

    labels = read_labels(tmp_path / "gold" / "labels_claude.csv")
    assert labels == {1: frozenset({"performance", "stability_bugs"}), 2: frozenset()}
    stored = pd.read_csv(tmp_path / "gold" / "labels_claude.csv")
    assert list(stored.columns) == ["review_id", "aspects", "annotator", "guidelines_version"]
    assert (tmp_path / "local" / "gold_reasons.csv").exists()


def test_reasons_are_never_written_under_data_gold(tmp_path: Path) -> None:
    run_import(tmp_path, [(1, "content", "says it has little content")], [1])

    for path in (tmp_path / "gold").rglob("*"):
        assert "little content" not in path.read_text(encoding="utf-8")


def test_a_later_batch_adds_to_the_labels(tmp_path: Path) -> None:
    run_import(tmp_path, [(1, "content", "r")], [1])
    run_import(tmp_path, [(2, "monetization", "r")], [2])

    assert set(read_labels(tmp_path / "gold" / "labels_claude.csv")) == {1, 2}


def test_import_rejects_missing_or_duplicate_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing"):
        run_import(tmp_path, [(1, "content", "r")], [1, 2])
    with pytest.raises(ValueError, match="more than once"):
        run_import(tmp_path, [(1, "content", "r"), (1, "none", "r")], [1])
    with pytest.raises(ValueError, match="not in this batch"):
        run_import(tmp_path, [(1, "content", "r"), (9, "none", "r")], [1])


def test_import_rejects_unknown_codes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="graphics"):
        run_import(tmp_path, [(1, "graphics", "r")], [1])


def test_import_rejects_none_mixed_with_codes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="none"):
        run_import(tmp_path, [(1, "none|content", "r")], [1])


def test_a_rejected_import_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="graphics"):
        run_import(tmp_path, [(1, "graphics", "r")], [1])

    assert not (tmp_path / "gold" / "labels_claude.csv").exists()
