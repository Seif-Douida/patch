"""The comparison-game candidate check (Phase 4, design D3): a few Steam requests per candidate,
printed for Seif to approve; nothing is stored.

A candidate passes with a mean of at least 20 reviews a day over the last 90 days, a release
before 3 Sep 2024 (28 days of fit window before the history starts on 1 Oct 2024), and news that
reaches 1 Oct 2024 within the 100 items the nightly reads (so its patch dates are known). The check
also counts, for each bundle of the tracked games' patches, how many passing candidates of the
same genre group had no patch of their own around it (design D6's patch rule; their daily data
is checked later, when the history exists).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from patchpulse.causal.bundles import Patch, bundle_patches
from patchpulse.causal.rules import DONOR_MIN_MEDIAN, GenreGroup, genre_group, patch_free_donors
from patchpulse.ingest.appdetails import fetch_app_details
from patchpulse.ingest.day_counts import fetch_window_total
from patchpulse.ingest.games import Game
from patchpulse.ingest.http import SteamError, SteamHttp
from patchpulse.ingest.models import NewsItem
from patchpulse.ingest.news import fetch_news
from patchpulse.ingest.patches import classify

HISTORY_START = date(2024, 10, 1)
RELEASED_BEFORE = HISTORY_START - timedelta(days=28)
VOLUME_DAYS = 90
NEWS_COUNT = 100


@dataclass(frozen=True)
class CandidateCheck:
    appid: int
    name: str
    genres: tuple[str, ...]
    group: GenreGroup | None
    release_date: date | None
    mean_per_day_90d: float
    news_reaches: bool
    treatment_patches: int  # since the history starts
    patch_days: tuple[date, ...]
    verdict: str  # "pass" or "fail"
    reasons: tuple[str, ...]


def _treatment_days(items: Sequence[NewsItem]) -> list[date]:
    return sorted(
        item.published_at.date() for item in items if classify(item).is_treatment_candidate
    )


def _check(http: SteamHttp, appid: int, today: date) -> CandidateCheck:
    try:
        details = fetch_app_details(http, appid)
    except SteamError:
        return CandidateCheck(
            appid, "", (), None, None, 0.0, False, 0, (), "fail", ("not found on Steam",)
        )
    total = fetch_window_total(
        http, appid, today - timedelta(days=VOLUME_DAYS), today - timedelta(days=1)
    )
    mean = round(total / VOLUME_DAYS, 1)
    news = fetch_news(http, appid, count=NEWS_COUNT).items
    reaches = len(news) < NEWS_COUNT or min(i.published_at.date() for i in news) <= HISTORY_START
    days = [d for d in _treatment_days(news) if d >= HISTORY_START]
    reasons: list[str] = []
    if mean < DONOR_MIN_MEDIAN:
        reasons.append(f"under {DONOR_MIN_MEDIAN} reviews a day ({mean})")
    if details.release_date is None:
        reasons.append("release date unknown")
    elif details.release_date >= RELEASED_BEFORE:
        reasons.append(
            f"released after {RELEASED_BEFORE - timedelta(days=1)} ({details.release_date})"
        )
    if not reaches:
        reasons.append(f"its {NEWS_COUNT} latest news items start after {HISTORY_START}")
    return CandidateCheck(
        appid=appid,
        name=details.name,
        genres=details.genres,
        group=genre_group(details.genres),
        release_date=details.release_date,
        mean_per_day_90d=mean,
        news_reaches=reaches,
        treatment_patches=len(days),
        patch_days=tuple(days),
        verdict="fail" if reasons else "pass",
        reasons=tuple(reasons),
    )


def check_candidates(
    http: SteamHttp, appids: Sequence[int], tracked: Sequence[Game], *, today: date
) -> tuple[list[CandidateCheck], dict[str, int]]:
    """Each candidate's check, and the passing candidates that qualify for each tracked bundle
    (keyed by the bundle's first patch id)."""
    checks = [_check(http, appid, today) for appid in appids]
    passing = {c.appid: c for c in checks if c.verdict == "pass" and c.group is not None}
    groups = {appid: c.group for appid, c in passing.items() if c.group is not None}
    patch_days = {appid: list(c.patch_days) for appid, c in passing.items()}
    qualifying: dict[str, int] = {}
    for game in tracked:
        items = fetch_news(http, game.appid, count=NEWS_COUNT).items
        patches = [
            Patch(item.gid, game.appid, item.published_at.date())
            for item in items
            if classify(item).is_treatment_candidate and item.published_at.date() >= HISTORY_START
        ]
        group = genre_group(game.genres)
        for bundle in bundle_patches(patches):
            qualifying[bundle.bundle_id] = len(patch_free_donors(bundle, group, groups, patch_days))
    return checks, qualifying


def summary_lines(checks: Sequence[CandidateCheck], qualifying: dict[str, int]) -> list[str]:
    """The table and the qualifying summary the command prints."""
    lines = [
        f"{'appid':>8}  {'name':30}  {'group':12}  {'per day':>7}  {'released':10}  "
        f"{'news':4}  {'patches':>7}  verdict"
    ]
    for c in sorted(checks, key=lambda c: (c.verdict, c.group or "", -c.mean_per_day_90d)):
        why = ": " + "; ".join(c.reasons) if c.reasons else ""
        lines.append(
            f"{c.appid:>8}  {c.name[:30]:30}  {c.group or '-':12}  {c.mean_per_day_90d:>7}  "
            f"{c.release_date or '-'!s:10}  {'yes' if c.news_reaches else 'no':4}  "
            f"{c.treatment_patches:>7}  {c.verdict}{why}"
        )
    counts = list(qualifying.values())
    if counts:
        few = sum(1 for n in counts if n < 5)
        lines.append(
            f"{len(counts)} bundles of tracked patches; {few} with fewer than 5 qualifying "
            f"candidates; median {statistics.median(counts)}"
        )
    return lines
