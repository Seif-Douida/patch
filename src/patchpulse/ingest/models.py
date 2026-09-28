"""Typed records for what the pipeline keeps from Steam (spec §6.1), plus raw-page envelopes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from patchpulse.ingest.privacy import KEPT_AUTHOR_FIELDS, PERSONAL_AUTHOR_FIELDS, author_hash

# Every review field Steam sent on 2026-09-28, plus the ones that appear only sometimes. Anything
# else is reported as schema drift; only the spec §6.1 fields are kept.
KNOWN_REVIEW_FIELDS = frozenset(
    {
        "recommendationid",
        "author",
        "language",
        "review",
        "timestamp_created",
        "timestamp_updated",
        "voted_up",
        "votes_up",
        "votes_funny",
        "weighted_vote_score",
        "comment_count",
        "steam_purchase",
        "received_for_free",
        "refunded",
        "written_during_early_access",
        "primarily_steam_deck",
        "reactions",
        "app_release_date",
        "developer_response",
        "timestamp_dev_responded",
        "hidden_in_steam_china",
        "steam_china_location",
        # The reviewer's PC specs (os, cpu_name, adapter_description, system_ram, vram_size) on
        # ~20% of reviews; no identifiers. Kept in the raw payload only, for now.
        "hardware",
        # Undocumented boolean seen on 11 of 3,360 Cities: Skylines II reviews (all negative,
        # English) on 2026-09-28; a candidate off-topic signal for Phase 4. Raw payload only.
        "bBeWary",
    }
)
KNOWN_AUTHOR_FIELDS = KEPT_AUTHOR_FIELDS | PERSONAL_AUTHOR_FIELDS | {"deck_playtime_at_review"}


@dataclass(frozen=True)
class RawPage:
    """One Steam response as stored in raw.steam_page (personal fields already removed)."""

    source: str  # "reviews" | "news" | "prices"
    appid: int | None
    request: Mapping[str, str | int]
    http_status: int
    payload: Mapping[str, Any]
    n_items: int


class ReviewRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    recommendation_id: int
    appid: int
    author_hash: str
    author_num_games_owned: int | None
    author_num_reviews: int | None
    author_playtime_forever: int | None
    author_playtime_at_review: int | None
    author_playtime_last_two_weeks: int | None
    language: str
    review_text: str
    timestamp_created: datetime
    timestamp_updated: datetime
    voted_up: bool
    votes_up: int
    votes_funny: int
    weighted_vote_score: float
    comment_count: int
    steam_purchase: bool
    received_for_free: bool
    written_during_early_access: bool
    primarily_steam_deck: bool | None = None

    @classmethod
    def from_steam(cls, review: Mapping[str, Any], *, appid: int, salt: str) -> ReviewRecord:
        """Parse one `appreviews` review. Raises (ValidationError, KeyError, ...) if unusable."""
        author = review["author"]
        return cls.model_validate(
            {
                "recommendation_id": review.get("recommendationid"),
                "appid": appid,
                "author_hash": author_hash(str(author["steamid"]), salt=salt),
                "author_num_games_owned": author.get("num_games_owned"),
                "author_num_reviews": author.get("num_reviews"),
                "author_playtime_forever": author.get("playtime_forever"),
                "author_playtime_at_review": author.get("playtime_at_review"),
                "author_playtime_last_two_weeks": author.get("playtime_last_two_weeks"),
                "language": review.get("language"),
                "review_text": review.get("review"),
                "timestamp_created": review.get("timestamp_created"),
                "timestamp_updated": review.get("timestamp_updated"),
                "voted_up": review.get("voted_up"),
                "votes_up": review.get("votes_up"),
                "votes_funny": review.get("votes_funny"),
                "weighted_vote_score": review.get("weighted_vote_score"),
                "comment_count": review.get("comment_count"),
                "steam_purchase": review.get("steam_purchase"),
                "received_for_free": review.get("received_for_free"),
                "written_during_early_access": review.get("written_during_early_access"),
                "primarily_steam_deck": review.get("primarily_steam_deck"),
            }
        )


def schema_drift(review: Mapping[str, Any]) -> frozenset[str]:
    """Field paths Steam sent that this code doesn't know about (e.g. "author.new_field")."""
    unknown = {name for name in review if name not in KNOWN_REVIEW_FIELDS}
    author = review.get("author")
    if isinstance(author, Mapping):
        unknown |= {f"author.{name}" for name in author if name not in KNOWN_AUTHOR_FIELDS}
    return frozenset(unknown)


class NewsItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    gid: str
    appid: int
    title: str
    url: str
    contents: str
    published_at: datetime
    feedname: str
    feedlabel: str
    tags: list[str] = []

    @classmethod
    def from_steam(cls, item: Mapping[str, Any]) -> NewsItem:
        fields = {**item, "gid": str(item.get("gid")), "published_at": item.get("date")}
        return cls.model_validate(fields)


class PriceSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    currency: str
    initial: int
    final: int
    discount_percent: int
