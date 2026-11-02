"""The API's data and `/reports` (Phase 5 plan Task 15, design D2 and D9): the bundle and the
reports come from the `agent-data` branch, never SQL; `/health` never depends on them."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from agent_fixtures import GAME, PATCH_DAY, SAMPLE_QUOTE, make_bundle, sample_report
from api_fixtures import Published, Seconds
from fake_clock import FakeClock
from fastapi.testclient import TestClient

from patchpulse.agent.bundle import BUNDLE_FILE
from patchpulse.agent.graph import AgentDeps, ReportRequest, run_report
from patchpulse.agent.llm_cache import ReplayCache
from patchpulse.agent.prompts import PROMPT_NAMES, load_prompt
from patchpulse.agent.report import Report
from patchpulse.agent.router import ModelRouter
from patchpulse.agent.tools import Tools, Trace
from patchpulse.api.app import create_app
from patchpulse.api.bundle_source import BundleSource
from patchpulse.api.settings import ApiSettings
from patchpulse.labeling.llm import LLMResponse

CHAIN = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
SITE = "https://site.example"


def client(published: Published, clock: Seconds, **settings: Any) -> TestClient:
    source = BundleSource(published, clock=clock)
    return TestClient(create_app(ApiSettings(**settings), source))


def test_api_without_a_bundle_stays_healthy() -> None:
    # Review focus 4: the branch not published yet, or the network down.
    published, clock = Published(), Seconds()
    published.down = True
    api = client(published, clock)

    assert api.get("/health").json()["status"] == "ok"
    assert published.fetched == []  # /health never touches the data
    response = api.get("/reports/p1")
    assert response.status_code == 503
    assert response.json() == {"detail": "data not available yet"}

    published.down = False
    published.publish(make_bundle(), sample_report("p1", 7))
    assert api.get("/reports/p1").status_code == 503  # retried, but not on every request
    clock.now += 31
    assert api.get("/reports/p1").status_code == 200  # the API kept retrying the load


def test_reports_serves_the_public_copy() -> None:
    published = Published()
    published.publish(make_bundle(), sample_report("p1", 7), sample_report("p1", 14))
    api = client(published, Seconds())

    response = api.get("/reports/p2")  # p2 is in p1's bundle

    assert response.status_code == 200
    body = response.json()
    assert body["data_version"] == 52
    assert (body["report"]["bundle_id"], body["report"]["horizon"]) == ("p1", 14)
    text = response.text
    assert SAMPLE_QUOTE not in text
    assert "7003" not in text
    assert "The patch caused the drop." not in text
    assert api.get("/reports/p0").status_code == 404  # its bundle has no report yet
    assert api.get("/reports/e1").status_code == 404  # an event post: no bundle
    assert api.get("/reports/nope").status_code == 404
    assert api.get("/reports/" + "9" * 100).status_code == 422


def test_bundle_reloads_when_the_version_changes() -> None:
    published, clock = Published(), Seconds()
    published.publish(make_bundle(), sample_report("p1", 7))
    source = BundleSource(published, clock=clock)
    first = source.current()
    assert first is not None
    assert first.data_version == 52

    published.publish(replace(make_bundle(), data_version=53), sample_report("p1", 14))
    clock.now += 299
    assert source.current() is first  # checked at most every 5 minutes
    clock.now += 2
    second = source.current()
    assert second is not None
    assert second.data_version == 53
    assert second.reports["p1"]["horizon"] == 14
    assert published.fetched.count(BUNDLE_FILE) == 2

    # The same version again: only reports.json is read.
    clock.now += 301
    assert source.current() is second
    assert published.fetched.count(BUNDLE_FILE) == 2

    # A bundle and reports.json from different nights (a cache serving one file late): the
    # previous pair stays, and the load is tried again.
    published.publish(replace(make_bundle(), data_version=54), reports_version=55)
    clock.now += 301
    assert source.current() is second
    published.publish(replace(make_bundle(), data_version=55))
    clock.now += 31
    third = source.current()
    assert third is not None
    assert third.data_version == 55


def test_cors_allows_only_the_site() -> None:
    published = Published()
    published.publish(make_bundle(), sample_report("p1", 7))
    api = client(published, Seconds(), site_origin=SITE)

    allowed = api.get("/reports/p1", headers={"Origin": SITE})
    other = api.get("/reports/p1", headers={"Origin": "https://elsewhere.example"})

    assert allowed.headers["access-control-allow-origin"] == SITE
    assert "access-control-allow-origin" not in other.headers
    preflight = api.options(
        "/reports/p1",
        headers={"Origin": "https://elsewhere.example", "Access-Control-Request-Method": "GET"},
    )
    assert preflight.status_code == 400
    assert (
        client(published, Seconds())
        .get("/reports/p1", headers={"Origin": SITE})
        .headers.get("access-control-allow-origin")
        is None
    )  # no site configured: no cross-origin reads


class Scripted:
    """Answers each step; `on_draft` runs when the draft is asked for."""

    def __init__(self, draft: Any) -> None:
        self.draft = draft
        self.by_prompt = {load_prompt(name).text: name for name in PROMPT_NAMES}

    def generate(self, system: str, user: str, *, json_schema: Any, **_: Any) -> LLMResponse:
        step = self.by_prompt[system]
        answers: dict[str, Any] = {
            "gather": {"calls": []},
            "summary": {"summary": "The positive share in the window is reported below."},
        }
        answer = self.draft(user) if step == "draft" else answers[step]
        return LLMResponse(json.dumps(answer), 900, 120, "STOP")


def test_a_run_keeps_one_bundle_version(tmp_path: Path) -> None:
    # Review focus 5: a new bundle published while a run holds the old one. Every call and V2's
    # re-execution use the version the run started with; the next run sees the new one.
    old = make_bundle()
    game, *others = old.games
    window = (PATCH_DAY, PATCH_DAY + timedelta(days=6))
    newer_days = tuple(replace(d, positives=d.totals) for d in game.days)
    new = replace(old, data_version=53, games=(replace(game, days=newer_days), *others))
    published, clock = Published(), Seconds()
    published.publish(old, sample_report("p1", 7))
    source = BundleSource(published, clock=clock)
    share = Tools(old, Trace()).sentiment_timeseries(GAME, *window, "day")["total"]["share"]
    assert share != Tools(new, Trace()).sentiment_timeseries(GAME, *window, "day")["total"]["share"]

    def draft(user: str) -> dict[str, Any]:
        published.publish(new, sample_report("p1", 7))  # the nightly job publishes mid-run
        clock.now += 3_600
        assert source.current() is not None  # someone else's request loads it
        found = re.search(r"\[(\S+)\] sentiment_timeseries", user)
        assert found
        claim = {
            "claim_id": "c1",
            "text": f"The positive share in the window was {share:.2f}%.",
            "claim_type": "descriptive",
            "value": share,
            "unit": "%",
            "evidence": {"call_id": found.group(1), "json_path": "total.share"},
        }
        return {"findings": [], "claims": [claim]}

    def run() -> Report:
        data = source.current()
        assert data is not None
        cache = ReplayCache(tmp_path / "unused.sqlite", "live")
        model = Scripted(draft)
        router = ModelRouter(
            CHAIN,
            {CHAIN[0]: model, CHAIN[1]: model},
            cache,
            clock=FakeClock(datetime(2026, 10, 11, tzinfo=UTC)),
        )
        result = run_report(ReportRequest.for_bundle("p1", 7), AgentDeps(data.bundle, router))
        assert isinstance(result, Report)
        return result

    report = run()

    assert report.data_version == 52
    assert [c.claim_id for c in report.claims] == ["c1"]  # V2 re-ran it on the run's bundle
    assert report.rule_pass_rates["V2"] == 1.0
    following = source.current()
    assert following is not None
    assert following.data_version == 53


def test_api_settings_keep_secrets_out_of_their_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PP_GEMINI_API_KEY", "hunter2-gemini")
    monkeypatch.setenv("PP_QDRANT_URL", "https://hunter2-cluster.example.cloud:6333")
    monkeypatch.setenv("PP_QDRANT_API_KEY", "hunter2-qdrant")
    monkeypatch.setenv("PP_LANGFUSE_SECRET_KEY", "hunter2-hunter2")
    monkeypatch.setenv("PP_SITE_ORIGIN", SITE)

    settings = ApiSettings()

    assert settings.site_origin == SITE
    assert settings.qdrant_url is not None
    assert settings.qdrant_url.get_secret_value().startswith("https://hunter2-cluster")
    assert "hunter2" not in repr(settings)
