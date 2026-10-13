"""An invented working directory for the students' commands: a snapshot, silver labels, gold dev
and test with Claude's labels, and an audit set with Seif's (no real review text)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from toy_reviews import toy_reviews

from patchpulse.labeling.gold import format_aspects
from patchpulse.labeling.taxonomy import ASPECTS

N_SILVER, N_DEV, N_TEST, N_AUDIT = 300, 60, 40, 20
TEACHER = "openai_compat/gemma-31b"


def aspects_of(row: list[bool]) -> str:
    return format_aspects(frozenset(code for code, on in zip(ASPECTS, row, strict=True) if on))


def write_inputs(root: Path) -> list[str]:
    """Writes everything under `root`; returns every review text, review id n at index n - 1."""
    texts, labels = toy_reviews(N_SILVER + N_DEV + N_TEST, seed=9)
    data, gold = root / "data" / "local", root / "data" / "gold"
    (data / "silver").mkdir(parents=True)
    gold.mkdir(parents=True)
    ids = list(range(1, len(texts) + 1))
    languages = ["french" if i % 3 == 0 else "english" for i in ids]
    pd.DataFrame(
        {
            "review_id": ids,
            "appid": 10,
            "language": languages,
            "voted_up": True,
            "text": texts,
        }
    ).to_parquet(data / "reviews.parquet", index=False)
    pd.DataFrame({"appid": [10], "name": ["Toy Game"]}).to_parquet(data / "games.parquet")
    pd.DataFrame({"appid": [], "title": []}).to_parquet(data / "patches.parquet")
    silver = ids[:N_SILVER]
    pd.DataFrame(
        {
            "review_id": silver,
            "split": ["holdout" if i % 10 == 0 else "train" for i in silver],
            "aspects": [aspects_of(list(labels[i - 1])) for i in silver],
            "model": TEACHER,
            "prompt_hash": "abc123",
        }
    ).to_parquet(data / "silver" / "labels.parquet", index=False)
    dev, test = ids[N_SILVER : N_SILVER + N_DEV], ids[N_SILVER + N_DEV :]
    pd.DataFrame(
        {
            "review_id": dev + test,
            "split": ["dev"] * N_DEV + ["test"] * N_TEST,
            "language": [languages[i - 1] for i in dev + test],
        }
    ).to_csv(gold / "sample.csv", index=False)
    pd.DataFrame(
        {
            "review_id": dev + test,
            "aspects": [aspects_of(list(labels[i - 1])) for i in dev + test],
            "annotator": "claude",
            "guidelines_version": "v4",
        }
    ).to_csv(gold / "labels_claude.csv", index=False)
    audit = test[:N_AUDIT]
    pd.DataFrame({"review_id": audit}).to_csv(gold / "audit.csv", index=False)
    pd.DataFrame(
        {"review_id": audit, "aspects": [aspects_of(list(labels[i - 1])) for i in audit]}
    ).to_csv(gold / "labels_seif.csv", index=False)
    return texts
