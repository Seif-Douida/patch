"""The scoring stage (plan Task 15, design D8): aspect scores for every review's latest version.

One rule covers new reviews, edited reviews and a new champion: score each review's latest version
that has no score from the current model version, newest first, until the time budget runs out.
Each page is committed on its own, so a stop at the budget, or a failure, keeps what was scored
and the next night carries on. Every model's scores stay in raw.

dbt reads a version's latest score as the one with the highest run id. A return to an earlier
champion (a rollback) would leave the later model's scores the latest, so the stage first re-stamps
the current model's rows that another model has scored since with this run: they become the
latest again without predicting anything. A cheap check skips this on every other night.

Newest first is keyset paging on (timestamp_created, recommendation_id), descending, through
`ix_steam_review_created` (migration 0002): each page starts where the last one ended, so a
backlog of a million reviews costs one pass, not one sort per page.

Imported by the runner only when a model is configured: it needs the `scoring` group.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Engine, text

from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.artifact import Scorer

log = logging.getLogger("patchpulse.pipeline.scoring")

MODEL_NAME = "aspect-classifier"
BATCH = 1_000

_LATEST_UNSCORED = """
    NOT EXISTS (
        SELECT 1 FROM raw.steam_review AS newer
        WHERE newer.recommendation_id = r.recommendation_id
          AND newer.timestamp_updated > r.timestamp_updated)
    AND NOT EXISTS (
        SELECT 1 FROM raw.review_aspect_score AS s
        WHERE s.recommendation_id = r.recommendation_id
          AND s.timestamp_updated = r.timestamp_updated
          AND s.model_version = :model_version)
"""
_PAGE = f"""
SELECT TOP (:batch) r.recommendation_id, r.timestamp_updated, r.timestamp_created, r.review_text
FROM raw.steam_review AS r
WHERE {_LATEST_UNSCORED} {{after}}
ORDER BY r.timestamp_created DESC, r.recommendation_id DESC
"""  # noqa: S608 - SQL from constants; every value is a bound parameter
_AFTER = (
    "AND (r.timestamp_created < :created "
    "OR (r.timestamp_created = :created AND r.recommendation_id < :rid))"
)
_REMAINING = f"SELECT COUNT(*) FROM raw.steam_review AS r WHERE {_LATEST_UNSCORED}"  # noqa: S608
# A page goes in as one JSON document unpacked by OPENJSON: one round trip instead of one per row,
# which the driver's executemany costs (measured: inserts were ~80% of the stage's time).
_COLUMNS = (
    ("recommendation_id", "BIGINT"),
    ("timestamp_updated", "DATETIME2(0)"),
    ("model_version", "NVARCHAR(64)"),
    ("model_name", "NVARCHAR(64)"),
    ("scored_run_id", "BIGINT"),
    *((f"p_{code}", "REAL") for code in ASPECTS),
    *((f"l_{code}", "BIT") for code in ASPECTS),
)
_INSERT = (
    f"INSERT INTO raw.review_aspect_score ({', '.join(name for name, _ in _COLUMNS)}) "  # noqa: S608
    f"SELECT {', '.join(name for name, _ in _COLUMNS)} FROM OPENJSON(:page) WITH "
    f"({', '.join(f'{name} {kind}' for name, kind in _COLUMNS)})"
)
# Has another model scored anything since the current one last did? Only after a return to an
# earlier champion; every other night this is one pass over the run ids and the answer is no.
_BEHIND = """
SELECT CASE WHEN EXISTS (
    SELECT 1 FROM raw.review_aspect_score
    WHERE model_version <> :model_version
      AND scored_run_id > (
          SELECT MAX(scored_run_id) FROM raw.review_aspect_score
          WHERE model_version = :model_version)
) THEN 1 ELSE 0 END
"""
_RESTAMP = """
UPDATE TOP (:batch) s SET scored_run_id = :run_id, scored_at = SYSUTCDATETIME()
FROM raw.review_aspect_score AS s
WHERE s.model_version = :model_version
  AND EXISTS (
      SELECT 1 FROM raw.review_aspect_score AS o
      WHERE o.recommendation_id = s.recommendation_id
        AND o.timestamp_updated = s.timestamp_updated
        AND o.scored_run_id > s.scored_run_id)
"""
RESTAMP_BATCH = 50_000


@dataclass(frozen=True)
class ScoreStats:
    scored: int
    remaining: int  # latest review versions this model version hasn't scored yet
    model_version: str
    restamped: int = 0  # this model's scores made the latest again after a rollback


def _iso(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def model_version_of(scorer: Scorer) -> str:
    return f"{scorer.artifact.family}-{scorer.artifact.version}"


def score_reviews(
    engine: Engine,
    scorer: Scorer,
    *,
    run_id: int,
    budget_s: float,
    clock: Callable[[], float] = time.monotonic,
    batch: int = BATCH,
) -> ScoreStats:
    model_version = model_version_of(scorer)
    deadline = clock() + budget_s
    after: tuple[datetime, int] | None = None
    scored = restamped = 0
    with engine.connect() as connection:
        behind: bool = bool(
            connection.execute(text(_BEHIND), {"model_version": model_version}).scalar_one()
        )
    while behind and clock() < deadline:
        with engine.begin() as connection:
            count = connection.execute(
                text(_RESTAMP),
                {"batch": RESTAMP_BATCH, "run_id": run_id, "model_version": model_version},
            ).rowcount
        restamped += count
        behind = count == RESTAMP_BATCH
    while clock() < deadline:
        parameters: dict[str, Any] = {"batch": batch, "model_version": model_version}
        if after is not None:
            parameters.update(created=after[0], rid=after[1])
        with engine.connect() as connection:
            page = connection.execute(
                text(_PAGE.format(after=_AFTER if after is not None else "")), parameters
            ).all()
        if not page:
            break
        probs = scorer.predict_proba([row.review_text for row in page])
        labels = scorer.labels(probs)
        records = [
            {
                "recommendation_id": row.recommendation_id,
                "timestamp_updated": row.timestamp_updated,
                "model_version": model_version,
                "model_name": MODEL_NAME,
                "scored_run_id": run_id,
                **{f"p_{code}": float(probs[i, a]) for a, code in enumerate(ASPECTS)},
                **{f"l_{code}": bool(labels[i, a]) for a, code in enumerate(ASPECTS)},
            }
            for i, row in enumerate(page)
        ]
        with engine.begin() as connection:
            connection.execute(text(_INSERT), {"page": json.dumps(records, default=_iso)})
        scored += len(page)
        after = (page[-1].timestamp_created, page[-1].recommendation_id)
    with engine.connect() as connection:
        remaining: int = connection.execute(
            text(_REMAINING), {"model_version": model_version}
        ).scalar_one()
    log.info(
        "scored %d reviews with %s, re-stamped %d; %d left",
        scored,
        model_version,
        restamped,
        remaining,
    )
    return ScoreStats(
        scored=scored, remaining=int(remaining), model_version=model_version, restamped=restamped
    )
