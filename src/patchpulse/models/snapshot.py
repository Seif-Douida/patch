"""The local snapshot Phase 2 works from (design D1).

`pull` copies each review's latest version (with its text), the games and the treatment-candidate
patches from Azure SQL into Parquet files in `data/local/`, which git ignores: review text never
leaves Seif's machine except for labeling (spec C4). Each pull wakes the database once, so pulls
are rare and incremental: only reviews seen since the previous pull (`last_seen_run_id`) are
fetched and merged, keeping each review's newest version.

It reads the raw tables, not dbt's core: the same rows, without depending on a dbt build. The one
exception is Phase 3's index set, `core.review_index`: which reviews Qdrant holds is decided by
dbt, and the laptop's embeddings must follow the same rules as the nightly job. It holds no text
and is replaced whole on each pull, since the per-game cap moves. Every file is written to a
temporary name and renamed only after all the reads succeed, so a failed pull leaves the previous
snapshot as it was.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, text
from sqlalchemy.exc import DBAPIError

from patchpulse.ingest.patches import CLASSIFIER_VERSION
from patchpulse.retrieval.rules import INDEX_COLUMNS

REVIEWS_FILE = "reviews.parquet"
GAMES_FILE = "games.parquet"
PATCHES_FILE = "patches.parquet"
INDEX_FILE = "index.parquet"
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

# core.review_index's columns (Phase 3), defined with the index rules.
_INDEX_SQL = (
    f"SELECT {', '.join(f'[{name}]' for name in INDEX_COLUMNS)} "  # noqa: S608 (constants)
    "FROM core.review_index ORDER BY review_id"
)

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
    index_total: int  # reviews in Phase 3's index set


@dataclass(frozen=True)
class Snapshot:
    reviews: pd.DataFrame
    games: pd.DataFrame
    patches: pd.DataFrame
    # Phase 3's index set; empty in a snapshot pulled before it existed.
    index: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=list(INDEX_COLUMNS)))


class IndexTableMissingError(RuntimeError):
    """The database has no core.review_index yet (the nightly dbt stage builds it)."""


def _previous_run_id(data_dir: Path) -> int:
    state = data_dir / STATE_FILE
    if not state.exists():
        return 0
    return int(json.loads(state.read_text(encoding="utf-8"))["max_run_id"])


def _read_chunked(engine: Engine, sql: str, params: dict[str, int]) -> pd.DataFrame:
    with engine.connect() as connection:
        chunks = list(pd.read_sql(text(sql), connection, params=params, chunksize=_CHUNK_ROWS))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _read_index(engine: Engine) -> pd.DataFrame:
    try:
        index = _read_chunked(engine, _INDEX_SQL, {})
    except (DBAPIError, pd.errors.DatabaseError) as error:  # pandas wraps the driver's error
        if "review_index" not in str(error):
            raise
        raise IndexTableMissingError(
            "core.review_index isn't in the database yet: the nightly dbt stage builds it once "
            "Phase 3's Milestone A is deployed. Pull again after that night."
        ) from None
    return index if not index.empty else pd.DataFrame(columns=list(INDEX_COLUMNS))


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
    new = _read_chunked(engine, _REVIEWS_SQL, {"since": since})
    index = _read_index(engine)
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
            data_dir / INDEX_FILE: index,
            data_dir / STATE_FILE: {
                "max_run_id": max_run_id,
                "pulled_at": now.isoformat(),
                "reviews": len(reviews),
            },
        }
    )
    return PullResult(
        reviews_new=len(new),
        reviews_total=len(reviews),
        max_run_id=max_run_id,
        index_total=len(index),
    )


def load_snapshot(data_dir: Path) -> Snapshot:
    paths = [data_dir / name for name in (REVIEWS_FILE, GAMES_FILE, PATCHES_FILE)]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"no snapshot in {data_dir} ({', '.join(missing)} missing): run "
            "`patchpulse-ml pull --yes` first"
        )
    reviews, games, patches = (pd.read_parquet(path) for path in paths)
    index_path = data_dir / INDEX_FILE
    if not index_path.exists():
        return Snapshot(reviews=reviews, games=games, patches=patches)
    return Snapshot(
        reviews=reviews, games=games, patches=patches, index=pd.read_parquet(index_path)
    )


def explain_pull_error(error: BaseException) -> str | None:
    """A fix for the two failures a pull from a home machine meets, or None.

    The firewall message carries Seif's home IP; it's never repeated (personal data).
    """
    if isinstance(error, IndexTableMissingError):
        return str(error)
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
