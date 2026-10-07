"""The local snapshot Phase 2 works from (design D1).

`pull` copies each review's latest version (with its text), the games and the treatment-candidate
patches from Azure SQL into Parquet files in `data/local/`, which git ignores: review text never
leaves Seif's machine except for labeling (spec C4). Each pull wakes the database once, so pulls
are rare and incremental: only reviews seen since the previous pull (`last_seen_run_id`) are
fetched and merged, keeping each review's newest version.

It reads the raw tables, not dbt's core: the same rows, without depending on a dbt build. Every
file is written to a temporary name and renamed only after all the reads succeed, so a failed
pull leaves the previous snapshot as it was.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, text

from patchpulse.ingest.patches import CLASSIFIER_VERSION

REVIEWS_FILE = "reviews.parquet"
GAMES_FILE = "games.parquet"
PATCHES_FILE = "patches.parquet"
STATE_FILE = "pull_state.json"
_CHUNK_ROWS = 50_000

_REVIEWS_SQL = """
WITH latest AS (
    SELECT
        recommendation_id, appid, language, voted_up, author_playtime_at_review,
        timestamp_created, timestamp_updated, last_seen_run_id, review_text,
        ROW_NUMBER() OVER (
            PARTITION BY recommendation_id ORDER BY timestamp_updated DESC
        ) AS version_rank
    FROM raw.steam_review
)
SELECT
    recommendation_id AS review_id,
    appid,
    CAST(timestamp_created AS date) AS [date],
    language,
    voted_up,
    author_playtime_at_review AS playtime_at_review,
    timestamp_updated,
    last_seen_run_id,
    review_text AS [text]
FROM latest
WHERE version_rank = 1 AND last_seen_run_id > :since
"""

_GAMES_SQL = "SELECT appid, name, genres FROM raw.game"

_PATCHES_SQL = """
SELECT n.appid, CAST(n.published_at AS date) AS [date], n.title, c.patch_type
FROM raw.steam_news AS n
INNER JOIN raw.news_classification AS c
    ON c.gid = n.gid AND c.classifier_version = :version
WHERE c.is_treatment_candidate = 1
ORDER BY n.appid, n.published_at
"""


@dataclass(frozen=True)
class PullResult:
    reviews_new: int  # rows fetched this time (new reviews, new versions, re-seen reviews)
    reviews_total: int
    max_run_id: int


@dataclass(frozen=True)
class Snapshot:
    reviews: pd.DataFrame
    games: pd.DataFrame
    patches: pd.DataFrame


def _previous_run_id(data_dir: Path) -> int:
    state = data_dir / STATE_FILE
    if not state.exists():
        return 0
    return int(json.loads(state.read_text(encoding="utf-8"))["max_run_id"])


def _read_reviews(engine: Engine, since: int) -> pd.DataFrame:
    with engine.connect() as connection:
        chunks = list(
            pd.read_sql(
                text(_REVIEWS_SQL), connection, params={"since": since}, chunksize=_CHUNK_ROWS
            )
        )
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _merge(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Each review once, as its newest version; a re-seen version takes the newer row."""
    if existing is None or existing.empty:
        merged = new
    elif new.empty:
        merged = existing
    else:
        merged = pd.concat([existing, new], ignore_index=True)
    merged = merged.sort_values(["review_id", "timestamp_updated"], kind="stable")
    return merged.drop_duplicates("review_id", keep="last").reset_index(drop=True)


def _write_atomically(files: dict[Path, pd.DataFrame | dict[str, object]]) -> None:
    staged: list[tuple[Path, Path]] = []
    for path, content in files.items():
        temporary = path.with_name(path.name + ".tmp")
        if isinstance(content, pd.DataFrame):
            content.to_parquet(temporary, index=False)
        else:
            temporary.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
        staged.append((temporary, path))
    for temporary, path in staged:  # the state file goes last
        os.replace(temporary, path)


def pull(engine: Engine, data_dir: Path, *, now: datetime) -> PullResult:
    """Fetch what changed since the previous pull and merge it into the snapshot."""
    since = _previous_run_id(data_dir)
    new = _read_reviews(engine, since)
    with engine.connect() as connection:
        games = pd.read_sql(text(_GAMES_SQL), connection)
        patches = pd.read_sql(
            text(_PATCHES_SQL), connection, params={"version": CLASSIFIER_VERSION}
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    reviews_path = data_dir / REVIEWS_FILE
    existing = pd.read_parquet(reviews_path) if reviews_path.exists() else None
    reviews = _merge(existing, new)
    max_run_id = max([since, *([int(new["last_seen_run_id"].max())] if not new.empty else [])])
    _write_atomically(
        {
            reviews_path: reviews,
            data_dir / GAMES_FILE: games,
            data_dir / PATCHES_FILE: patches,
            data_dir / STATE_FILE: {
                "max_run_id": max_run_id,
                "pulled_at": now.isoformat(),
                "reviews": len(reviews),
            },
        }
    )
    return PullResult(reviews_new=len(new), reviews_total=len(reviews), max_run_id=max_run_id)


def load_snapshot(data_dir: Path) -> Snapshot:
    paths = [data_dir / name for name in (REVIEWS_FILE, GAMES_FILE, PATCHES_FILE)]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"no snapshot in {data_dir} ({', '.join(missing)} missing): run "
            "`patchpulse-ml pull --yes` first"
        )
    reviews, games, patches = (pd.read_parquet(path) for path in paths)
    return Snapshot(reviews=reviews, games=games, patches=patches)


def explain_pull_error(error: BaseException) -> str | None:
    """A fix for the two failures a pull from a home machine meets, or None.

    The firewall message carries Seif's home IP; it's never repeated (personal data).
    """
    message = str(error)
    if "is not allowed to access the server" in message:
        return (
            "Azure SQL's firewall rejected this machine: your public IP has probably changed. "
            "Update the DEV_IP_ADDRESS secret in the GitHub environment 'production', run the "
            "infra workflow, then pull again."
        )
    if any(marker in message for marker in ("DefaultAzureCredential", "AADSTS", "az login")):
        return "Azure sign-in failed: run `az login` (in the browser), then pull again."
    return None
