"""Personal data never reaches storage (spec C4).

Review authors are identified only by `author_hash`: an HMAC-SHA256 of the Steam ID with a secret
salt, stable so the same author deduplicates across runs. Raw payloads keep only an allowlist of
author fields, so a personal field Steam adds later is dropped by default.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping
from typing import Any

KEPT_AUTHOR_FIELDS = frozenset(
    {
        "num_games_owned",
        "num_reviews",
        "playtime_forever",
        "playtime_at_review",
        "playtime_last_two_weeks",
    }
)
# The personal fields Steam sent on 2026-09-28; documented so tests can check they're gone.
PERSONAL_AUTHOR_FIELDS = frozenset(
    {"steamid", "personaname", "profile_url", "avatar", "persona_status", "last_played"}
)


def author_hash(steamid: str, *, salt: str) -> str:
    if not salt:
        raise ValueError("author_hash needs a non-empty salt (PP_AUTHOR_HASH_SALT)")
    return hmac.new(salt.encode(), steamid.encode(), hashlib.sha256).hexdigest()


def strip_personal_fields(review: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of one review with only the allowlisted author fields."""
    stripped = dict(review)
    author = review.get("author")
    if isinstance(author, Mapping):
        stripped["author"] = {k: v for k, v in author.items() if k in KEPT_AUTHOR_FIELDS}
    return stripped


def strip_review_page(payload: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of an `appreviews` response with every review stripped."""
    stripped = dict(payload)
    reviews = payload.get("reviews")
    if isinstance(reviews, list):
        stripped["reviews"] = [
            strip_personal_fields(r) if isinstance(r, Mapping) else r for r in reviews
        ]
    return stripped
