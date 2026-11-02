"""A report's scope and its core evidence (design D4): the same tool calls for every report.

The graph runs them as its core_evidence step; the report store runs them again to fingerprint a
report's evidence (design D10) without importing the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from patchpulse.agent.bundle import Bundle, PatchRow
from patchpulse.agent.rules import CoreEvidence
from patchpulse.agent.tools import Tools

HORIZONS = (7, 14)
BASELINE_DAYS = 28  # the days before the window, as Phase 4's fit window
NEAR_START_DAYS = 3  # a sale or discount within 3 days of the start (Phase 4's flags)
DAILY_UP_TO_DAYS = 14  # a window up to two weeks is shown by day, a longer one by week


@dataclass(frozen=True)
class Scope:
    appid: int
    window: tuple[date, date]
    baseline: tuple[date, date]
    start: date  # the bundle's first day, or the window's
    bundle_id: str | None
    horizon: int | None
    patch: PatchRow | None


def days_in(window: tuple[date, date]) -> int:
    return (window[1] - window[0]).days + 1


def bundle_scope(bundle: Bundle, bundle_id: str, horizon: int) -> Scope:
    """A patch bundle's report: the `horizon` days from its first patch's day, and the
    BASELINE_DAYS before."""
    for game in bundle.games:
        members = [p for p in game.patches if p.bundle_id == bundle_id]
        if members:
            first = min(members, key=lambda p: (p.day, p.patch_id))
            start = first.day
            return Scope(
                appid=game.appid,
                window=(start, start + timedelta(days=horizon - 1)),
                baseline=(start - timedelta(days=BASELINE_DAYS), start - timedelta(days=1)),
                start=start,
                bundle_id=bundle_id,
                horizon=horizon,
                patch=first,
            )
    raise ValueError(f"no bundle {bundle_id} in the data")


def core_evidence(tools: Tools, scope: Scope) -> CoreEvidence:
    """The core tool calls, recorded in the tools' trace, and what the rules read from them."""
    appid, window, baseline = scope.appid, scope.window, scope.baseline
    ids: dict[str, str] = {}

    def last(name: str) -> None:
        ids[name] = tools.trace.calls[-1].call_id

    tools.get_game_overview(appid)
    tools.list_patches(appid, baseline[0], window[1])
    by = "day" if days_in(window) <= DAILY_UP_TO_DAYS else "week"
    window_series = tools.sentiment_timeseries(appid, *window, by)
    last("window")
    baseline_series = tools.sentiment_timeseries(appid, *baseline, "week")
    last("baseline")
    near = (
        scope.start - timedelta(days=NEAR_START_DAYS),
        scope.start + timedelta(days=NEAR_START_DAYS),
    )
    near_series = tools.sentiment_timeseries(appid, *near, "week")
    last("near_start")
    aspects = tools.aspect_breakdown(appid, window, baseline)
    last("aspects")
    split = tools.decompose_change(appid, window, baseline)
    last("decomposition")
    row = None
    if scope.bundle_id is not None:
        tools.estimate_patch_effect(scope.bundle_id)
        last("estimate")
        served = tools.estimate_rows(tools.trace.calls[-1])
        row = next((r for r in served if r.horizon == scope.horizon), None)
    return CoreEvidence(
        estimate=row,
        window=window_series["total"],
        baseline=baseline_series["total"],
        near_start=near_series["total"],
        aspects=aspects,
        decomposition=split,
        call_ids=ids,
    )
