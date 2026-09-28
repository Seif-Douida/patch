"""Patch notes (GetNewsForApp) and price snapshots (appdetails price_overview), spec §6.1."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx2

from patchpulse.ingest.http import RateLimiter, SteamHttp
from patchpulse.ingest.news import fetch_news
from patchpulse.ingest.prices import PRICE_BATCH_SIZE, fetch_prices


class Recorder:
    def __init__(self, respond: Any) -> None:
        self.respond = respond
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return httpx2.Response(200, json=self.respond(request))


def http_for(recorder: Recorder) -> SteamHttp:
    return SteamHttp(
        httpx2.Client(transport=httpx2.MockTransport(recorder)),
        limiter=RateLimiter(min_interval_s=0.0),
    )


def query(request: httpx2.Request) -> dict[str, list[str]]:
    return parse_qs(urlsplit(str(request.url)).query)


def news_item(gid: str, **overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "gid": gid,
        "title": "Devoid of Liberty: 7.1.1",
        "url": "https://steamstore-a.akamaihd.net/news/externalpost/x",
        "is_external_url": True,
        "author": "someone",
        "contents": "Fixes",
        "feedlabel": "Community Announcements",
        "date": 1758704400,
        "feedname": "steam_community_announcements",
        "feed_type": 1,
        "appid": 553850,
        "tags": ["patchnotes"],
    }
    item.update(overrides)
    return item


def test_news_items_are_parsed() -> None:
    recorder = Recorder(
        lambda _: {"appnews": {"appid": 553850, "newsitems": [news_item("1"), news_item("2")]}}
    )

    fetched = fetch_news(http_for(recorder), 553850)

    first = fetched.items[0]
    assert first.gid == "1"
    assert first.published_at == datetime.fromtimestamp(1758704400, UTC)
    assert first.tags == ["patchnotes"]
    assert first.feedname == "steam_community_announcements"
    assert len(fetched.items) == 2


def test_news_without_tags_parses_with_an_empty_list() -> None:
    item = news_item("1")
    del item["tags"]
    recorder = Recorder(lambda _: {"appnews": {"appid": 553850, "newsitems": [item]}})

    assert fetch_news(http_for(recorder), 553850).items[0].tags == []


def test_news_request_asks_for_full_contents() -> None:
    recorder = Recorder(lambda _: {"appnews": {"appid": 553850, "newsitems": []}})

    fetch_news(http_for(recorder), 553850)

    params = query(recorder.requests[0])
    assert params["appid"] == ["553850"]
    assert params["count"] == ["100"]
    assert params["maxlength"] == ["0"]


def price_response(request: httpx2.Request) -> dict[str, Any]:
    ids = query(request)["appids"][0].split(",")
    body: dict[str, Any] = {}
    for appid in ids:
        if appid == "553850":
            body[appid] = {"success": False}  # region-restricted with cc=gb, seen 2026-09-28
        elif appid == "999":
            body[appid] = {"success": True, "data": []}  # free games have no price_overview
        else:
            overview = {"currency": "USD", "initial": 4999, "final": 2499, "discount_percent": 50}
            body[appid] = {"success": True, "data": {"price_overview": overview}}
    return body


def test_price_snapshots_are_parsed_per_app_in_batches() -> None:
    recorder = Recorder(price_response)
    appids = list(range(1000, 1000 + PRICE_BATCH_SIZE + 5))

    fetched = fetch_prices(http_for(recorder), appids)

    assert len(recorder.requests) == 2
    params = query(recorder.requests[0])
    assert params["cc"] == ["us"]
    assert params["filters"] == ["price_overview"]
    snapshot = fetched.snapshots[1000]
    assert snapshot is not None
    assert (snapshot.currency, snapshot.final, snapshot.discount_percent) == ("USD", 2499, 50)
    assert set(fetched.snapshots) == set(appids)


def test_price_failure_is_recorded_not_raised() -> None:
    fetched = fetch_prices(http_for(Recorder(price_response)), [553850, 1091500])

    assert fetched.snapshots[553850] is None
    assert fetched.failed == frozenset({553850})
    assert fetched.snapshots[1091500] is not None


def test_free_game_has_no_price_but_is_not_a_failure() -> None:
    fetched = fetch_prices(http_for(Recorder(price_response)), [999])

    assert fetched.snapshots[999] is None
    assert fetched.failed == frozenset()
