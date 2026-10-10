"""The gold set (spec §7.2, design D2): which reviews, which split, which ones Seif audits.

600 reviews (400 English, 200 French). Within a language every game gets an equal share, capped by
what it has; a game short of reviews passes its shortfall to the others. Within a game the share
is spread over thumbs, length and near-patch strata in proportion to their sizes. Then 150 go to
dev (prompt and threshold tuning) and 450 to test (reported once), and Seif's blind audit takes
100 English and 50 French reviews from test.

Everything is seeded and independent of row order, so the same snapshot and seed give the same
set. Reviews with no text are never sampled: there is nothing to label.
"""

from __future__ import annotations

import json
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.snapshot import Snapshot

EN = "english"  # Steam's language names
FR = "french"
NEAR_PATCH_DAYS = 14
SHORT_WORDS = 20
LONG_WORDS = 150
SAMPLE_FILE = "sample.csv"
AUDIT_FILE = "audit.csv"
STRATA = ["voted_up", "length_bucket", "near_patch"]


def length_bucket(text: str) -> str:
    words = len(text.split())
    if words < SHORT_WORDS:
        return "short"
    return "long" if words > LONG_WORDS else "medium"


def _near_patch(reviews: pd.DataFrame, patches: pd.DataFrame) -> pd.Series:
    """True for a review written within 14 days after one of its game's treatment patches."""
    near = pd.Series(False, index=reviews.index)
    review_days = pd.to_datetime(reviews["date"]).to_numpy().astype("datetime64[D]")
    for appid, game_patches in patches.groupby("appid"):
        mask = (reviews["appid"] == appid).to_numpy()
        if not mask.any():
            continue
        patch_days = np.sort(
            pd.to_datetime(game_patches["date"]).to_numpy().astype("datetime64[D]")
        )
        days = review_days[mask]
        latest = np.searchsorted(patch_days, days, side="right") - 1
        has_patch = latest >= 0
        gap = np.full(days.shape, np.inf)
        gap[has_patch] = (days[has_patch] - patch_days[latest[has_patch]]).astype(float)
        near[mask] = gap <= NEAR_PATCH_DAYS
    return near


def _equal_allocation[K: Hashable](total: int, capacity: Mapping[K, int]) -> dict[K, int]:
    """Split `total` as evenly as possible, never giving a key more than its capacity."""
    allocation = dict.fromkeys(capacity, 0)
    remaining = total
    open_keys = sorted((k for k, c in capacity.items() if c > 0), key=str)
    while remaining > 0 and open_keys:
        share, extra = divmod(remaining, len(open_keys))
        for position, key in enumerate(open_keys):
            give = min(share + (1 if position < extra else 0), capacity[key] - allocation[key])
            allocation[key] += give
            remaining -= give
        open_keys = [k for k in open_keys if allocation[k] < capacity[k]]
    if remaining:
        raise ValueError(f"only {total - remaining} of {total} items available")
    return allocation


def _proportional_allocation[K: Hashable](total: int, sizes: Mapping[K, int]) -> dict[K, int]:
    """Largest-remainder allocation of `total` in proportion to `sizes`, capped by each size."""
    available = sum(sizes.values())
    if total > available:
        raise ValueError(f"only {available} of {total} items available")
    quotas = {k: total * s / available for k, s in sizes.items()} if available else {}
    allocation = {k: min(int(q), sizes[k]) for k, q in quotas.items()}
    remaining = total - sum(allocation.values())
    order = sorted(sizes, key=lambda k: (-(quotas[k] - int(quotas[k])), str(k)))
    while remaining > 0:
        for key in order:
            if remaining and allocation[key] < sizes[key]:
                allocation[key] += 1
                remaining -= 1
    return allocation


def _draw(group: pd.DataFrame, k: int, rng: np.random.Generator) -> pd.DataFrame:
    return group.iloc[np.sort(rng.choice(len(group), size=k, replace=False))]


def sample_gold(snapshot: Snapshot, *, seed: int, n_en: int = 400, n_fr: int = 200) -> pd.DataFrame:
    reviews = snapshot.reviews.sort_values("review_id", kind="stable")
    reviews = reviews[reviews["language"].isin([EN, FR]) & (reviews["text"].str.strip() != "")]
    reviews = reviews.assign(
        length_bucket=reviews["text"].map(length_bucket),
        near_patch=_near_patch(reviews, snapshot.patches),
    )
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for language, quota in ((EN, n_en), (FR, n_fr)):
        pool = reviews[reviews["language"] == language]
        games = {appid: group for appid, group in pool.groupby("appid", sort=True)}
        try:
            per_game = _equal_allocation(quota, {appid: len(g) for appid, g in games.items()})
        except ValueError as error:
            raise ValueError(f"not enough {language} reviews for the gold set: {error}") from None
        for appid in sorted(per_game, key=str):
            strata = {key: group for key, group in games[appid].groupby(STRATA, sort=True)}
            allocation = _proportional_allocation(
                per_game[appid], {key: len(group) for key, group in strata.items()}
            )
            for key in sorted(allocation, key=str):
                if allocation[key]:
                    parts.append(_draw(strata[key], allocation[key], rng))
    sample = pd.concat(parts).sort_values("review_id", kind="stable")
    columns = ["review_id", "appid", "language", "voted_up", "length_bucket", "near_patch"]
    return sample[columns].reset_index(drop=True)


def split_dev_test(sample: pd.DataFrame, *, seed: int, n_dev: int = 150) -> pd.DataFrame:
    """Mark `n_dev` reviews as dev, in proportion over language and game; the rest is test."""
    ordered = sample.sort_values("review_id", kind="stable").reset_index(drop=True)
    cells = {key: group for key, group in ordered.groupby(["language", "appid"], sort=True)}
    allocation = _proportional_allocation(n_dev, {k: len(g) for k, g in cells.items()})
    rng = np.random.default_rng(seed)
    dev_ids: set[int] = set()
    for key in sorted(allocation, key=str):
        if allocation[key]:
            dev_ids.update(int(i) for i in _draw(cells[key], allocation[key], rng)["review_id"])
    split = np.where(ordered["review_id"].isin(dev_ids), "dev", "test")
    return ordered.assign(split=split)


def choose_audit(sample: pd.DataFrame, *, seed: int, n_en: int = 100, n_fr: int = 50) -> list[int]:
    """Seif's blind audit: random reviews from test only, so his labels report on test."""
    test = sample[sample["split"] == "test"].sort_values("review_id", kind="stable")
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for language, n in ((EN, n_en), (FR, n_fr)):
        pool = test[test["language"] == language]
        if len(pool) < n:
            raise ValueError(f"only {len(pool)} {language} test reviews for an audit of {n}")
        chosen.extend(int(i) for i in _draw(pool, n, rng)["review_id"])
    return sorted(chosen)


def write_gold(sample: pd.DataFrame, audit: list[int], gold_dir: Path) -> None:
    """Ids, splits and strata only: these files are committed, so never any review text."""
    gold_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(gold_dir / SAMPLE_FILE, index=False, lineterminator="\n")
    pd.DataFrame({"review_id": audit}).to_csv(
        gold_dir / AUDIT_FILE, index=False, lineterminator="\n"
    )


# The labeling batches (design D2): a pilot of 30 dev reviews, then the whole sample in four
# batches of 150 for four fresh subagents. Batches mix games and languages (a seeded shuffle).
BATCH_SEED = 20261006
BATCHES = 4
PILOT_SIZE = 30
CLAUDE_LABELS_FILE = "labels_claude.csv"
NONE = "none"


def batch_ids(
    sample: pd.DataFrame, k: int, *, seed: int = BATCH_SEED, batches: int = BATCHES
) -> list[int]:
    if not 1 <= k <= batches:
        raise ValueError(f"batch {k} is outside 1..{batches}")
    ids = sorted(int(i) for i in sample["review_id"])
    order = [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))]
    size = -(-len(order) // batches)
    return sorted(order[(k - 1) * size : k * size])


def pilot_ids(sample: pd.DataFrame, *, seed: int = BATCH_SEED, n: int = PILOT_SIZE) -> list[int]:
    dev = sorted(int(i) for i in sample.loc[sample["split"] == "dev", "review_id"])
    return sorted(int(i) for i in np.random.default_rng(seed).choice(dev, size=n, replace=False))


def export_batch(sample: pd.DataFrame, snapshot: Snapshot, ids: Sequence[int], path: Path) -> int:
    """One JSON line per review: id, game, language, thumbs, text. No split, no patch proximity."""
    unknown = sorted(set(ids) - {int(i) for i in sample["review_id"]})
    if unknown:
        raise ValueError(f"ids not in the gold sample: {unknown[:10]}")
    reviews = snapshot.reviews.set_index("review_id")
    names = snapshot.games.set_index("appid")["name"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for review_id in ids:
            review = reviews.loc[review_id]
            line = {
                "id": int(review_id),
                "game": str(names[review["appid"]]),
                "language": str(review["language"]),
                "thumbs": "up" if bool(review["voted_up"]) else "down",
                "text": str(review["text"]),
            }
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
    return len(ids)


@dataclass(frozen=True)
class GoldLabels:
    labels: dict[int, frozenset[str]]
    annotator: str
    guidelines_version: str


def parse_aspects(cell: str) -> frozenset[str]:
    """`performance|content` or `none`; anything else is an error, never a guess."""
    parts = [part.strip() for part in str(cell).split("|") if part.strip()]
    if parts == [NONE]:
        return frozenset()
    if NONE in parts:
        raise ValueError(f"'none' can't be combined with aspect codes: {cell!r}")
    unknown = [part for part in parts if part not in ASPECTS]
    if unknown or not parts:
        raise ValueError(f"unknown aspect code(s) {unknown or [cell]}; known: {', '.join(ASPECTS)}")
    return frozenset(parts)


def format_aspects(aspects: frozenset[str]) -> str:
    return "|".join(code for code in ASPECTS if code in aspects) or NONE


def _merge_rows(path: Path, new: pd.DataFrame) -> None:
    if path.exists():
        existing = pd.read_csv(path, keep_default_na=False)
        new = pd.concat([existing[~existing["review_id"].isin(new["review_id"])], new])
    path.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values("review_id").to_csv(path, index=False, lineterminator="\n")


def import_labels(
    csv_path: Path,
    expected_ids: Sequence[int],
    *,
    annotator: str,
    guidelines_version: str,
    gold_dir: Path,
    reasons_path: Path,
    labels_file: str = CLAUDE_LABELS_FILE,
) -> GoldLabels:
    """Check a subagent's `id,aspects,reason` file against its batch, then merge it in.

    Labels go to `gold_dir` (committed); reasons go to `reasons_path` (local: they can quote the
    review). Nothing is written unless the whole file is valid.
    """
    rows = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    ids = [int(i) for i in rows["id"]]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise ValueError(f"ids appear more than once: {repeated[:10]}")
    expected = set(expected_ids)
    extra, missing = sorted(set(ids) - expected), sorted(expected - set(ids))
    if extra:
        raise ValueError(f"ids not in this batch: {extra[:10]}")
    if missing:
        raise ValueError(f"missing ids: {missing[:10]}")
    labels: dict[int, frozenset[str]] = {}
    for review_id, cell in zip(ids, rows["aspects"], strict=True):
        try:
            labels[review_id] = parse_aspects(cell)
        except ValueError as error:
            raise ValueError(f"review {review_id}: {error}") from None

    _merge_rows(
        gold_dir / labels_file,
        pd.DataFrame(
            {
                "review_id": list(labels),
                "aspects": [format_aspects(a) for a in labels.values()],
                "annotator": annotator,
                "guidelines_version": guidelines_version,
            }
        ),
    )
    _merge_rows(
        reasons_path,
        pd.DataFrame(
            {
                "review_id": ids,
                "reason": list(rows["reason"]),
                "guidelines_version": guidelines_version,
            }
        ),
    )
    return GoldLabels(labels=labels, annotator=annotator, guidelines_version=guidelines_version)


def read_labels(path: Path) -> dict[int, frozenset[str]]:
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {
        int(i): parse_aspects(a) for i, a in zip(rows["review_id"], rows["aspects"], strict=True)
    }
