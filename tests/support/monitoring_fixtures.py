"""Phase 7: reviews and their scores in the SQL Server test database, for the monitoring stages."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import Engine, insert

from patchpulse.db import tables
from patchpulse.labeling.taxonomy import ASPECTS

SENTINEL_TEXT = "SENTINEL-REVIEW-the patch broke my saves and the support ignored me"


def review_row(
    rid: int,
    created: datetime,
    *,
    language: str = "english",
    text: str = SENTINEL_TEXT,
    updated: datetime | None = None,
    first_seen_run_id: int = 1,
    appid: int = 553850,
) -> dict[str, Any]:
    return {
        "recommendation_id": rid,
        "timestamp_updated": updated or created,
        "appid": appid,
        "author_hash": "a" * 64,
        "language": language,
        "review_text": text,
        "timestamp_created": created,
        "voted_up": False,
        "votes_up": 0,
        "votes_funny": 0,
        "weighted_vote_score": 0.0,
        "comment_count": 0,
        "steam_purchase": True,
        "received_for_free": False,
        "written_during_early_access": False,
        "first_seen_run_id": first_seen_run_id,
        "last_seen_run_id": first_seen_run_id,
    }


def score_row(
    rid: int,
    updated: datetime,
    *,
    model_version: str,
    aspects: Iterable[str] = (),
    prob: float = 0.9,
) -> dict[str, Any]:
    on = set(aspects)
    return {
        "recommendation_id": rid,
        "timestamp_updated": updated,
        "model_version": model_version,
        "model_name": "aspect-classifier",
        "scored_run_id": 1,
        "scored_at": updated,
        **{f"p_{code}": (prob if code in on else 1 - prob) for code in ASPECTS},
        **{f"l_{code}": code in on for code in ASPECTS},
    }


def add_rows(
    engine: Engine, reviews: Sequence[dict[str, Any]], scores: Sequence[dict[str, Any]] = ()
) -> None:
    with engine.begin() as connection:
        if reviews:
            connection.execute(insert(tables.steam_review), list(reviews))
        if scores:
            connection.execute(insert(tables.review_aspect_score), list(scores))
