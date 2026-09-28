"""Paging Steam reviews inside a date window (spec §6.1), with the cursor-loop guard."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx2

from patchpulse.ingest.http import RateLimiter, SteamHttp
from patchpulse.ingest.reviews import ReviewPage, StopReason, fetch_review_pages

START = datetime(2026, 9, 1, tzinfo=UTC)
END = datetime(2026, 9, 28, tzinfo=UTC)
SALT = "test-salt"


def steam_review(rid: int, created: datetime, **overrides: Any) -> dict[str, Any]:
    review: dict[str, Any] = {
        "recommendationid": str(rid),
        "author": {
            "steamid": f"7656119800000{rid:04d}",
            "personaname": "Someone",
            "profile_url": "https://steamcommunity.com/id/someone/",
            "avatar": "abc",
            "persona_status": "Online",
            "last_played": 1759000000,
            "num_games_owned": 10,
            "num_reviews": 2,
            "playtime_forever": 600,
            "playtime_at_review": 300,
            "playtime_last_two_weeks": 5,
        },
        "language": "english",
        "review": f"review {rid}",
        "timestamp_created": int(created.timestamp()),
        "timestamp_updated": int(created.timestamp()),
        "voted_up": True,
        "votes_up": 3,
        "votes_funny": 0,
        "weighted_vote_score": "0.523809552192687988",
        "comment_count": 0,
        "steam_purchase": True,
        "received_for_free": False,
        "refunded": False,
        "written_during_early_access": False,
        "primarily_steam_deck": False,
        "reactions": [],
        "app_release_date": 1700000000,
    }
    review.update(overrides)
    return review


def page(cursor: str, *reviews: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": 1,
        "query_summary": {"num_reviews": len(reviews)},
        "reviews": list(reviews),
        "cursor": cursor,
    }


class Steam:
    """Serves pages by the request's cursor and records every request."""

    def __init__(self, by_cursor: dict[str, dict[str, Any]]) -> None:
        self.by_cursor = by_cursor
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        cursor = parse_qs(urlsplit(str(request.url)).query)["cursor"][0]
        return httpx2.Response(200, json=self.by_cursor[cursor])


def fetch(
    steam: Steam, *, deadline: float | None = None, clock: Callable[[], float] | None = None
) -> list[ReviewPage]:
    http = SteamHttp(
        httpx2.Client(transport=httpx2.MockTransport(steam)),
        limiter=RateLimiter(min_interval_s=0.0),
    )
    pages: Iterator[ReviewPage] = fetch_review_pages(
        http,
        553850,
        start=START,
        end=END,
        salt=SALT,
        deadline=deadline,
        clock=clock or (lambda: 0.0),
    )
    return list(pages)


def day(n: int) -> datetime:
    return datetime(2026, 9, n, 12, tzinfo=UTC)


def test_review_pages_follow_the_cursor_within_the_date_range() -> None:
    steam = Steam(
        {
            "*": page("AoJ4+a/b=", steam_review(2, day(20)), steam_review(1, day(10))),
            "AoJ4+a/b=": page("AoJ4next", steam_review(0, day(5))),
            "AoJ4next": page("AoJ4end"),
        }
    )

    pages = fetch(steam)

    assert [len(p.records) for p in pages] == [2, 1, 0]
    assert pages[-1].next_stop is StopReason.EMPTY
    first = parse_qs(urlsplit(str(steam.requests[0].url)).query)
    assert first["filter"] == ["recent"]
    assert first["language"] == ["all"]
    assert first["num_per_page"] == ["100"]
    assert first["purchase_type"] == ["all"]
    assert first["filter_offtopic_activity"] == ["0"]
    assert first["date_range_type"] == ["include"]
    assert first["start_date"] == [str(int(START.timestamp()))]
    assert first["end_date"] == [str(int(END.timestamp()))]
    assert first["cursor"] == ["*"]
    # The cursor contains + / = and must reach Steam URL-encoded.
    assert "cursor=AoJ4%2Ba%2Fb%3D" in str(steam.requests[1].url)


def test_repeated_cursor_stops_paging_and_is_reported() -> None:
    steam = Steam(
        {
            "*": page("LOOP", steam_review(2, day(20))),
            "LOOP": page("LOOP", steam_review(1, day(10))),
        }
    )

    pages = fetch(steam)

    assert len(steam.requests) == 2
    assert pages[-1].next_stop is StopReason.CURSOR_REPEAT
    assert [r.recommendation_id for p in pages for r in p.records] == [2, 1]


def test_paging_stops_at_the_deadline() -> None:
    steam = Steam(
        {
            "*": page("A", steam_review(2, day(20))),
            "A": page("B", steam_review(1, day(10))),
        }
    )
    readings = iter([0.0, 100.0])  # before the first page, then after it

    pages = fetch(steam, deadline=50.0, clock=lambda: next(readings))

    assert len(steam.requests) == 1
    assert pages[-1].next_stop is StopReason.DEADLINE


def test_no_request_is_made_after_the_deadline() -> None:
    steam = Steam({})

    pages = fetch(steam, deadline=50.0, clock=lambda: 100.0)

    assert pages == []
    assert steam.requests == []


def test_empty_page_ends_paging() -> None:
    steam = Steam({"*": page("X")})

    pages = fetch(steam)

    assert len(pages) == 1
    assert pages[0].next_stop is StopReason.EMPTY


def test_reviews_older_than_the_window_stop_paging() -> None:
    old = datetime(2026, 8, 1, tzinfo=UTC)
    steam = Steam({"*": page("A", steam_review(2, day(20)), steam_review(1, old))})

    pages = fetch(steam)

    assert pages[-1].next_stop is StopReason.WINDOW_PASSED
    assert [r.recommendation_id for r in pages[-1].records] == [2]


def test_reviews_are_parsed_hashed_and_stripped() -> None:
    steam = Steam({"*": page("A", steam_review(7, day(20))), "A": page("B")})

    record = fetch(steam)[0].records[0]
    payload = fetch(steam)[0].payload

    assert record.recommendation_id == 7
    assert record.appid == 553850
    assert record.timestamp_created == day(20)
    assert record.weighted_vote_score == 0.523809552192687988
    assert record.author_playtime_at_review == 300
    assert len(record.author_hash) == 64
    assert "steamid" not in payload["reviews"][0]["author"]


def test_unknown_review_fields_are_reported_as_schema_drift() -> None:
    drifted = steam_review(1, day(20), new_field=1)
    drifted["author"]["new_author_field"] = 2
    steam = Steam({"*": page("A", drifted), "A": page("B")})

    first = fetch(steam)[0]

    assert first.drift == frozenset({"new_field", "author.new_author_field"})
    assert len(first.records) == 1


def test_hardware_block_is_a_known_field_not_drift() -> None:
    # Seen on 2026-09-28 on ~20% of reviews: the reviewer's PC specs (no identifiers).
    with_hardware = steam_review(
        1, day(20), hardware={"os": "Windows 11", "cpu_name": "CPU", "vram_size": 8192}
    )
    steam = Steam({"*": page("A", with_hardware), "A": page("B")})

    assert fetch(steam)[0].drift == frozenset()


def test_be_wary_flag_is_a_known_field_not_drift() -> None:
    # Seen on 2026-09-28: `bBeWary: true` on 11 of 3,360 Cities: Skylines II reviews, all
    # negative English ones. Undocumented; kept in the raw payload for the off-topic analysis.
    steam = Steam({"*": page("A", steam_review(1, day(20), bBeWary=True)), "A": page("B")})

    assert fetch(steam)[0].drift == frozenset()


def test_invalid_review_is_reported_and_skipped() -> None:
    broken = steam_review(1, day(20))
    del broken["review"]
    steam = Steam({"*": page("A", broken, steam_review(2, day(21))), "A": page("B")})

    first = fetch(steam)[0]

    assert first.invalid == 1
    assert [r.recommendation_id for r in first.records] == [2]
