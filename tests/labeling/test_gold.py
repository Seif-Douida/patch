"""The gold sample (design D2): 600 reviews, split 150 dev / 450 test, audit from test."""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from patchpulse.labeling.gold import EN, FR, choose_audit, sample_gold, split_dev_test
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.cli import main
from patchpulse.models.snapshot import Snapshot

GAMES = [100 + i for i in range(12)]
START = date(2026, 6, 1)
SPEC_ORDER = (  # spec §7.1, in order
    "performance",
    "stability_bugs",
    "gameplay_balance",
    "content",
    "monetization",
    "online_servers",
    "story_world",
    "ux_controls",
    "platform_policy",
    "off_topic",
)


def synthetic_snapshot(*, french_for_first_game: int = 60) -> Snapshot:
    rows = []
    review_id = 1
    for g, appid in enumerate(GAMES):
        languages = [(EN, 120), (FR, french_for_first_game if g == 0 else 60), ("russian", 30)]
        for language, count in languages:
            for i in range(count):
                words = (5, 60, 200)[i % 3]
                rows.append(
                    {
                        "review_id": review_id,
                        "appid": appid,
                        "date": START + timedelta(days=i % 90),
                        "language": language,
                        "voted_up": i % 4 != 0,
                        "playtime_at_review": 100,
                        "timestamp_updated": pd.Timestamp("2026-09-01"),
                        "last_seen_run_id": 1,
                        "text": " ".join(["word"] * words),
                    }
                )
                review_id += 1
    rows.append({**rows[0], "review_id": review_id, "text": "   "})  # an empty review
    reviews = pd.DataFrame(rows)
    games = pd.DataFrame({"appid": GAMES, "name": [f"Game {a}" for a in GAMES], "genres": "RPG"})
    patches = pd.DataFrame(
        {
            "appid": GAMES,
            "date": [START + timedelta(days=30)] * 12,
            "title": "Patch",
            "patch_type": "patch",
        }
    )
    return Snapshot(reviews=reviews, games=games, patches=patches)


def test_taxonomy_matches_the_spec_order() -> None:
    assert ASPECTS == SPEC_ORDER


def test_gold_counts_are_exact() -> None:
    sample = split_dev_test(sample_gold(synthetic_snapshot(), seed=1), seed=1)
    audit = choose_audit(sample, seed=1)

    assert (sample["language"] == EN).sum() == 400
    assert (sample["language"] == FR).sum() == 200
    assert (sample["split"] == "dev").sum() == 150
    assert (sample["split"] == "test").sum() == 450
    audited = sample.set_index("review_id").loc[audit]
    assert (audited["language"] == EN).sum() == 100
    assert (audited["language"] == FR).sum() == 50


def test_gold_sample_is_deterministic_for_a_seed() -> None:
    snapshot = synthetic_snapshot()
    shuffled = Snapshot(
        reviews=snapshot.reviews.sample(frac=1, random_state=3),
        games=snapshot.games,
        patches=snapshot.patches,
    )

    first = split_dev_test(sample_gold(snapshot, seed=7), seed=7)
    second = split_dev_test(sample_gold(shuffled, seed=7), seed=7)

    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))


def test_no_review_is_sampled_twice() -> None:
    sample = sample_gold(synthetic_snapshot(), seed=2)

    assert sample["review_id"].is_unique


def test_audit_comes_from_test_only() -> None:
    sample = split_dev_test(sample_gold(synthetic_snapshot(), seed=3), seed=3)

    audit = choose_audit(sample, seed=3)

    assert set(sample.set_index("review_id").loc[audit, "split"]) == {"test"}
    assert len(set(audit)) == 150


def test_every_game_appears_in_each_language() -> None:
    sample = sample_gold(synthetic_snapshot(), seed=4)

    for language in (EN, FR):
        assert set(sample.loc[sample["language"] == language, "appid"]) == set(GAMES)


def test_french_shortfall_moves_to_other_games() -> None:
    sample = sample_gold(synthetic_snapshot(french_for_first_game=3), seed=5)

    french = sample[sample["language"] == FR]
    assert len(french) == 200
    assert (french["appid"] == GAMES[0]).sum() == 3


def test_empty_reviews_are_never_sampled() -> None:
    snapshot = synthetic_snapshot()
    empty_id = int(snapshot.reviews["review_id"].max())

    sample = sample_gold(snapshot, seed=6)

    assert empty_id not in set(sample["review_id"])


def test_strata_are_recorded() -> None:
    sample = sample_gold(synthetic_snapshot(), seed=8)

    assert set(sample["length_bucket"]) == {"short", "medium", "long"}
    assert sample["near_patch"].any()
    assert not sample["near_patch"].all()


def test_gold_sample_refuses_to_overwrite_existing_labels_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Re-sampling after labeling would orphan the labels: it needs --force.
    for name in list(os.environ):
        if name.startswith("PP_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "gold").mkdir(parents=True)
    (tmp_path / "data" / "gold" / "sample.csv").write_text("review_id\n1\n", encoding="utf-8")

    assert main(["gold", "sample", "--seed", "1"]) == 2
    assert "--force" in capsys.readouterr().err
