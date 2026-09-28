"""No Steam IDs or profile data are ever stored (spec C4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from patchpulse.ingest.privacy import PERSONAL_AUTHOR_FIELDS, author_hash, strip_personal_fields

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "steam"


def review(**author: Any) -> dict[str, Any]:
    base_author = {
        "steamid": "76561198000000001",
        "personaname": "Someone",
        "profile_url": "https://steamcommunity.com/id/someone/",
        "avatar": "abcdef",
        "persona_status": "Online",
        "last_played": 1759000000,
        "num_games_owned": 12,
        "num_reviews": 3,
        "playtime_forever": 600,
        "playtime_at_review": 300,
        "playtime_last_two_weeks": 0,
    }
    base_author.update(author)
    return {"recommendationid": "1", "review": "Great game", "author": base_author}


def test_personal_fields_are_removed_from_raw_payloads() -> None:
    stripped = strip_personal_fields(review())

    assert set(stripped["author"]) == {
        "num_games_owned",
        "num_reviews",
        "playtime_forever",
        "playtime_at_review",
        "playtime_last_two_weeks",
    }
    assert "76561198000000001" not in json.dumps(stripped)
    assert stripped["review"] == "Great game"


def test_unknown_author_fields_are_dropped_too() -> None:
    # An allowlist: a personal field Steam adds tomorrow is not stored by default.
    stripped = strip_personal_fields(review(real_name="Jane Doe"))

    assert "real_name" not in stripped["author"]


def test_stripping_does_not_modify_the_input() -> None:
    original = review()
    strip_personal_fields(original)

    assert original["author"]["steamid"] == "76561198000000001"


def test_author_hash_is_salted_and_stable() -> None:
    first = author_hash("76561198000000001", salt="salt-a")

    assert first == author_hash("76561198000000001", salt="salt-a")
    assert first != author_hash("76561198000000001", salt="salt-b")
    assert first != author_hash("76561198000000002", salt="salt-a")
    assert len(first) == 64
    assert "76561198000000001" not in first


def test_empty_salt_is_rejected() -> None:
    with pytest.raises(ValueError, match="salt"):
        author_hash("76561198000000001", salt="")


def test_fixtures_contain_no_personal_fields() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in FIXTURES.glob("*.json"))

    assert text  # the fixtures exist
    for field in PERSONAL_AUTHOR_FIELDS:
        assert f'"{field}"' not in text, field
