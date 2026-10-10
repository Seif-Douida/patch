"""The silver set (plan Task 10, spec §7.3): ~30k teacher-labeled reviews the students train on.

- **Sample:** stratified by game, in proportion to each game's usable reviews but capped, so the
  biggest games can't crowd out the rest. Within a game the draw is uniform, so thumbs and language
  keep their real mix. Gold reviews and reviews with no text after cleaning are excluded.
- **Holdout:** a tenth is kept apart to measure how well the students agree with the teacher on
  reviews they never trained on.
- **Files (data/local, never committed):** `silver/ids.parquet` (review_id, split) and
  `silver/labels.parquet` (review_id, split, aspects, model, prompt_hash), exported from the label
  cache after every labeling run. Neither holds review text.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from patchpulse.labeling.cache import LabelCache
from patchpulse.labeling.gold import _proportional_allocation, format_aspects, parse_aspects
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.labeling.teacher import (
    LabelRun,
    TextModel,
    label_reviews,
    prompt_hash,
    review_inputs,
)
from patchpulse.models.snapshot import Snapshot
from patchpulse.models.text import clean_text

SILVER_DIR = "silver"
IDS_FILE = "ids.parquet"
LABELS_FILE = "labels.parquet"
SILVER_SIZE = 30_000
HOLDOUT_FRACTION = 0.1
DEFAULT_CAP_SHARE = 0.2  # no game gets more than a fifth of the silver set, if it can be avoided
TRAIN, HOLDOUT = "train", "holdout"


@dataclass(frozen=True)
class Silver:
    train: pd.DataFrame  # review_id, then one bool column per aspect
    holdout: pd.DataFrame


def _feasible_cap(total: int, sizes: dict[int, int], floor_cap: int) -> int:
    """The smallest cap from `floor_cap` up at which the games can still supply `total` reviews."""
    cap = floor_cap
    while sum(min(size, cap) for size in sizes.values()) < total:
        small = [size for size in sizes.values() if size <= cap]
        if len(small) == len(sizes):
            raise ValueError(f"only {sum(sizes.values())} usable reviews for a sample of {total}")
        cap = math.ceil((total - sum(small)) / (len(sizes) - len(small)))
    return cap


def _capped_allocation(total: int, sizes: dict[int, int], cap: int) -> dict[int, int]:
    if sum(min(size, cap) for size in sizes.values()) < total:
        raise ValueError(f"a per-game cap of {cap} can't supply {total} reviews from these games")
    allocation: dict[int, int] = {}
    open_games = dict(sizes)
    remaining = total
    while True:
        share = _proportional_allocation(remaining, open_games)
        over = [game for game, count in share.items() if count > cap]
        if not over:
            allocation.update(share)
            return allocation
        for game in over:
            allocation[game] = cap
            remaining -= cap
            del open_games[game]


def sample_silver(
    snapshot: Snapshot,
    *,
    exclude_ids: Iterable[int],
    n: int = SILVER_SIZE,
    per_game_cap: int | None = None,
    seed: int,
) -> list[int]:
    reviews = snapshot.reviews
    usable = reviews[~reviews["review_id"].isin(set(exclude_ids))]
    usable = usable[usable["text"].map(lambda text: bool(clean_text(str(text))))]
    counts = usable["appid"].value_counts()
    sizes = {int(appid): int(count) for appid, count in zip(counts.index, counts, strict=True)}
    floor_cap = max(math.ceil(n / len(sizes)), math.ceil(DEFAULT_CAP_SHARE * n))
    cap = per_game_cap if per_game_cap is not None else _feasible_cap(n, sizes, floor_cap)
    allocation = _capped_allocation(n, sizes, cap)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for appid in sorted(allocation):
        pool = np.sort(usable.loc[usable["appid"] == appid, "review_id"].to_numpy())
        chosen += [int(i) for i in rng.choice(pool, size=allocation[appid], replace=False)]
    return sorted(chosen)


def split_holdout(
    ids: Sequence[int], *, fraction: float = HOLDOUT_FRACTION, seed: int
) -> tuple[list[int], list[int]]:
    """(train, holdout), each sorted; the holdout is `fraction` of the ids, rounded."""
    ordered = sorted(int(i) for i in ids)
    shuffled = np.random.default_rng(seed).permutation(ordered)
    size = round(len(ordered) * fraction)
    return sorted(int(i) for i in shuffled[size:]), sorted(int(i) for i in shuffled[:size])


def _silver_dir(data_dir: Path) -> Path:
    return data_dir / SILVER_DIR


def write_silver_ids(data_dir: Path, train: Sequence[int], holdout: Sequence[int]) -> Path:
    path = _silver_dir(data_dir) / IDS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "review_id": [*train, *holdout],
            "split": [TRAIN] * len(train) + [HOLDOUT] * len(holdout),
        }
    ).sort_values("review_id")
    frame.to_parquet(path, index=False)
    return path


def read_silver_ids(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(_silver_dir(data_dir) / IDS_FILE)


def export_silver_labels(cache: LabelCache, data_dir: Path, *, model: str) -> int:
    """Write labels.parquet from the cache for every silver review labeled so far."""
    frame = read_silver_ids(data_dir)
    digest = prompt_hash()
    labels = cache.get_many([int(i) for i in frame["review_id"]], model=model, prompt_hash=digest)
    frame = frame[frame["review_id"].isin(labels)].copy()
    frame["aspects"] = [format_aspects(labels[int(i)]) for i in frame["review_id"]]
    frame["model"] = model
    frame["prompt_hash"] = digest
    frame.to_parquet(_silver_dir(data_dir) / LABELS_FILE, index=False)
    return len(frame)


def label_silver(
    client: TextModel, cache: LabelCache, snapshot: Snapshot, data_dir: Path, *, model: str
) -> LabelRun:
    """Label the silver reviews not yet cached; resumable, and exported even after a quota stop."""
    ids = [int(i) for i in read_silver_ids(data_dir)["review_id"]]
    run = label_reviews(client, cache, review_inputs(snapshot, ids), model=model)
    export_silver_labels(cache, data_dir, model=model)
    return run


def _expand(frame: pd.DataFrame) -> pd.DataFrame:
    aspects = [parse_aspects(cell) for cell in frame["aspects"]]
    expanded = pd.DataFrame({"review_id": frame["review_id"].astype("int64").to_numpy()})
    for code in ASPECTS:
        expanded[code] = [code in row for row in aspects]
    return expanded


def load_silver(data_dir: Path) -> Silver:
    labels = pd.read_parquet(_silver_dir(data_dir) / LABELS_FILE)
    return Silver(
        train=_expand(labels[labels["split"] == TRAIN]),
        holdout=_expand(labels[labels["split"] == HOLDOUT]),
    )


def prevalence_report(data_dir: Path, snapshot: Snapshot, *, top_languages: int = 6) -> str:
    """Positives per aspect, overall and for the most common languages (counts only)."""
    labels = pd.read_parquet(_silver_dir(data_dir) / LABELS_FILE)
    languages = snapshot.reviews.set_index("review_id")["language"]
    expanded = _expand(labels)
    expanded["language"] = [str(languages.get(i, "?")) for i in expanded["review_id"]]
    shown = list(expanded["language"].value_counts().index[:top_languages])
    header = ["aspect", "all", *shown]
    rows = [[f"reviews ({len(expanded):,})", f"{len(expanded):,}"]]
    rows[0] += [f"{(expanded['language'] == lang).sum():,}" for lang in shown]
    for code in ASPECTS:
        row = [code, f"{int(expanded[code].sum()):,}"]
        row += [
            f"{int(expanded.loc[expanded['language'] == lang, code].sum()):,}" for lang in shown
        ]
        rows.append(row)
    lines = [" | ".join(header), " | ".join("---" for _ in header)]
    lines += [" | ".join(row) for row in rows]
    return "\n".join(lines)
