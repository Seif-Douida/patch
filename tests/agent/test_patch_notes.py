"""get_patch_notes (plan Task 4, design D3): Steam's news API asked for the posts up to the patch's
own timestamp, the item matched by its id, cleaned and cut; "not found" is a result, not an
error; the shared Steam limiter paces the request."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx2
from agent_fixtures import GAME, make_bundle

from patchpulse.agent.patch_notes import PATCH_NOTES_CHARS
from patchpulse.agent.tools import Tools, Trace
from patchpulse.ingest.http import RateLimiter, RetryPolicy, SteamHttp

P1_PUBLISHED = datetime(2026, 9, 10, 16, tzinfo=UTC)  # the fixture's patch p1
NOTES = (
    "[h1]Patch 1.1.0[/h1][list][*]Fixed a crash on load[*]Details at "
    "https://store.steampowered.com/news/app/553850[/list]"
    "[img]{STEAM_CLAN_IMAGE}/1234/abcdef.png[/img]<p>Thanks for playing!</p>"
    "[url=https://example.com/forum]Read the forum post[/url]"
)


class CountingLimiter(RateLimiter):
    def __init__(self) -> None:
        super().__init__(min_interval_s=0.0)
        self.waits = 0

    def wait(self) -> None:
        self.waits += 1


class FakeNews:
    """GetNewsForApp answering with the given items; records each request's parameters."""

    def __init__(self, items: list[dict[str, Any]], status: int = 200) -> None:
        self.items = items
        self.status = status
        self.params: list[dict[str, str]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        query = urlsplit(str(request.url)).query
        self.params.append({k: v[0] for k, v in parse_qs(query).items()})
        if self.status != 200:
            return httpx2.Response(self.status)
        return httpx2.Response(200, json={"appnews": {"newsitems": self.items}})


def item(gid: str, contents: str, *, title: str = "Patch 1.1.0") -> dict[str, Any]:
    return {
        "gid": gid,
        "title": title,
        "url": f"https://example.com/{gid}",
        "contents": contents,
        "date": int(P1_PUBLISHED.timestamp()),
        "feedname": "steam_community_announcements",
        "appid": GAME,
    }


def tools_for(fake: FakeNews, limiter: RateLimiter | None = None) -> tuple[Tools, Trace]:
    http = SteamHttp(
        httpx2.Client(transport=httpx2.MockTransport(fake)),
        limiter=limiter or RateLimiter(min_interval_s=0.0),
        policy=RetryPolicy(max_attempts=1),
    )
    trace = Trace()
    return Tools(make_bundle(), trace, steam=http), trace


def test_notes_are_cleaned_and_cut() -> None:
    fake = FakeNews([item("p1", NOTES + " More fixes." * 400)])
    tools, _ = tools_for(fake)

    result = tools.get_patch_notes("p1")

    assert (result["patch_id"], result["found"], result["truncated"]) == ("p1", True, True)
    notes = result["notes"]
    assert notes.startswith("Patch 1.1.0\nFixed a crash on load\nDetails at\nThanks for playing!")
    assert "Read the forum post" in notes
    for leftover in ("http", "[", "{STEAM_CLAN_IMAGE}", "<p>", "<url>", ".png"):
        assert leftover not in notes
    assert len(notes) <= PATCH_NOTES_CHARS
    (params,) = fake.params
    assert params["appid"] == str(GAME)
    assert params["enddate"] == str(int(P1_PUBLISHED.timestamp()) + 1)
    assert params["maxlength"] == "0"
    assert int(params["count"]) <= 10


def test_the_item_is_found_among_several_by_id() -> None:
    fake = FakeNews(
        [
            item("p9", "Another post the same day", title="Patch 1.1.0"),
            item("p1", "The notes we want"),
            item("p8", "An older post"),
        ]
    )
    tools, _ = tools_for(fake)

    result = tools.get_patch_notes("p1")

    assert (result["notes"], result["truncated"], result["title"]) == (
        "The notes we want",
        False,
        "Patch 1.1.0",
    )


def test_an_item_missing_from_the_answer_is_not_found() -> None:
    tools, trace = tools_for(FakeNews([item("p9", "Something else")]))

    result = tools.get_patch_notes("p1")

    assert result == {
        "patch_id": "p1",
        "title": "Patch 1.1.0",
        "day": "2026-09-10",
        "found": False,
        "notes": None,
        "truncated": False,
    }
    assert trace.calls[0].result == result


def test_the_shared_limiter_paces_each_request() -> None:
    limiter = CountingLimiter()
    tools, _ = tools_for(FakeNews([item("p1", "notes")]), limiter)

    tools.get_patch_notes("p1")
    tools.get_patch_notes("p2")

    assert limiter.waits == 2


def test_patch_notes_errors_are_results() -> None:
    down, _ = tools_for(FakeNews([], status=503))
    fake = FakeNews([])
    tools, _ = tools_for(fake)

    errors = [
        down.get_patch_notes("p1"),
        tools.get_patch_notes("999"),
        Tools(make_bundle(), Trace()).get_patch_notes("p1"),
    ]

    assert [e["error"] for e in errors] == [
        "Steam's news service didn't answer",
        "no patch 999 in the data",
        "patch notes aren't available in this run",
    ]
    assert fake.params == []
