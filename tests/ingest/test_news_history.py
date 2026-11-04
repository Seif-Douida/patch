"""Paging a game's official Steam posts back to its history's start (the patch-history fix,
4 Nov 2026): GetNewsForApp's `enddate` on the official feed, against the in-process fake."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from fake_steam import FakeSteam

from patchpulse.ingest.news import NewsHistory, fetch_news, fetch_news_history

GAME = 553850
FIRST = date(2024, 9, 20)
LAST = date(2025, 1, 1)
START = date(2024, 10, 1)


def noon(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 12, tzinfo=UTC)


def daily_posts() -> FakeSteam:
    """One official post a day at noon UTC, 20 Sep 2024 to 1 Jan 2025 (104 posts)."""
    steam = FakeSteam()
    day = FIRST
    while day <= LAST:
        steam.add_news(GAME, f"g{day:%Y%m%d}", f"Community update {day}", noon(day), tags=())
        day += timedelta(days=1)
    return steam


def news_requests(steam: FakeSteam) -> list[str]:
    return [str(r.url) for r in steam.requests if "GetNewsForApp" in str(r.url)]


def gids(history: NewsHistory) -> set[str]:
    return {item.gid for fetch in history.fetches for item in fetch.items}


def test_history_pages_back_until_a_post_before_the_start() -> None:
    steam = daily_posts()

    history = fetch_news_history(
        steam.http(), GAME, oldest=noon(LAST), until=START, max_pages=5, count=30
    )

    assert history.reached is True
    every_day = {f"g{START + timedelta(days=i):%Y%m%d}" for i in range(93)}  # 1 Oct to 1 Jan
    assert every_day <= gids(history)
    # 30 a page, each from the last page's oldest post: 1 Jan-3 Dec, 3 Dec-4 Nov, 4 Nov-6 Oct,
    # then 6 Oct-20 Sep, which holds 30 Sep: nothing is asked after it.
    assert len(news_requests(steam)) == 4


def test_history_stops_at_the_page_limit() -> None:
    steam = daily_posts()

    history = fetch_news_history(
        steam.http(), GAME, oldest=noon(LAST), until=START, max_pages=2, count=30
    )

    assert len(news_requests(steam)) == 2
    assert history.reached is False


def test_history_stops_when_a_page_brings_nothing_older() -> None:
    # Steam answering the newest page whatever the end date: 1 Jan back to 3 Dec, all stored.
    steam = daily_posts()
    steam.ignore_enddate = True

    history = fetch_news_history(
        steam.http(), GAME, oldest=noon(date(2024, 12, 3)), until=START, max_pages=5, count=30
    )

    assert len(news_requests(steam)) == 1
    assert history.reached is False


def test_history_keeps_posts_sharing_the_boundary_second() -> None:
    # Three a page: page 1 ends on c1, and c2 was posted the same second; the next page starts
    # at that second again, so c2 isn't skipped.
    steam = FakeSteam()
    second = datetime(2025, 3, 1, 9, 30, tzinfo=UTC)
    for gid, at in [
        ("a", second + timedelta(days=2)),
        ("b", second + timedelta(days=1)),
        ("c1", second),
        ("c2", second),
        ("d", second - timedelta(days=1)),
        ("e", second - timedelta(days=2)),
    ]:
        steam.add_news(GAME, gid, f"Post {gid}", at, tags=())

    history = fetch_news_history(
        steam.http(),
        GAME,
        oldest=second + timedelta(days=2),
        until=date(2024, 1, 1),
        max_pages=10,
        count=3,
    )

    assert {"c1", "c2", "d", "e"} <= gids(history)


def test_press_posts_are_never_asked_for() -> None:
    steam = daily_posts()
    steam.add_news(GAME, "press", "Helldivers 2 gets a hotfix", noon(date(2024, 12, 20)),
                   feedname="PC Gamer", tags=())  # fmt: skip

    latest = fetch_news(steam.http(), GAME, count=30)
    history = fetch_news_history(
        steam.http(), GAME, oldest=noon(LAST), until=START, max_pages=5, count=30
    )

    assert "press" not in {item.gid for item in latest.items} | gids(history)
    assert all("feeds=steam_community_announcements" in url for url in news_requests(steam))
