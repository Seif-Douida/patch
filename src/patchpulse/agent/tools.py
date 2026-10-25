"""The core tools (design D3, spec §10.2): read-only, deterministic for a bundle version, returning
compact JSON-able summaries.

Every call is recorded in the run's Trace as a ToolCall (its id, the tool, its JSON-able args and
its result) and `Tools.rerun` re-executes a recorded call on the same bundle, for V2. A call that
can't be answered (an untracked game, an unknown patch, a bad range) returns `{"error": ...}`,
recorded like any result. Shares, intervals and effects are in percent or percentage points,
rounded to 2 places; intervals are 90%. Phase 4's `serve` and `decompose` are reused as they are.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from math import sqrt
from typing import Any

from patchpulse.agent.bundle import Bundle, GameData
from patchpulse.agent.patch_notes import fetch_patch_notes
from patchpulse.agent.search import DEFAULT_HITS, ReviewSearch, find_reviews, snippet
from patchpulse.causal.decompose import BUCKETS, PARTS, decompose
from patchpulse.causal.its import POSITIVE_SHARE
from patchpulse.causal.store import StoredEstimate, serve
from patchpulse.ingest.http import SteamError, SteamHttp
from patchpulse.labeling.taxonomy import ASPECTS

Z90 = 1.6449  # the marts' Wilson macro
PLACES = 2
NEW_PLAYERS = "lt_2h"  # under 2 hours at review, as the mart's share_new_players
Window = tuple[date, date]  # first and last days, both included
Result = dict[str, Any]


class ToolError(ValueError):
    """A call the tool can't answer; its message becomes the recorded result."""


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool: str
    args: dict[str, Any]
    result: dict[str, Any]


@dataclass(frozen=True)
class ModelCall:
    step: str
    model_id: str
    input_tokens: int
    output_tokens: int
    cached: bool


@dataclass
class Trace:
    """One run's tool and model calls, in order. `run` prefixes the call ids, so evidence cited
    from another run's trace isn't found here. `retrieved` keeps the full text search_reviews
    returned, by review id, for V3 and V7; it never leaves the run."""

    run: str = "run"
    calls: list[ToolCall] = field(default_factory=list)
    model_calls: list[ModelCall] = field(default_factory=list)
    retrieved: dict[int, str] = field(default_factory=dict)

    def find(self, call_id: str) -> ToolCall | None:
        return next((c for c in self.calls if c.call_id == call_id), None)


def _round(value: float | None, places: int = PLACES) -> float | None:
    return None if value is None else round(value, places)


def _pct(part: int, whole: int) -> float | None:
    return None if whole == 0 else round(100 * part / whole, PLACES)


def wilson(positives: int, n: int, z: float = Z90) -> tuple[float, float] | None:
    """Wilson's interval for a proportion, as fractions (the marts' macro)."""
    if n == 0:
        return None
    p = positives / n
    centre = p + z * z / (2 * n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    scale = 1 + z * z / n
    return max(0.0, (centre - half) / scale), min(1.0, (centre + half) / scale)


def newcombe(x1: int, n1: int, x0: int, n0: int) -> tuple[float, float] | None:
    """Newcombe's hybrid score interval for p1 - p0, as fractions (from the two Wilson
    intervals; well behaved at 0 and at small counts)."""
    w1, w0 = wilson(x1, n1), wilson(x0, n0)
    if w1 is None or w0 is None:
        return None
    p1, p0 = x1 / n1, x0 / n0
    (l1, u1), (l0, u0) = w1, w0
    diff = p1 - p0
    return diff - sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2), diff + sqrt(
        (u1 - p1) ** 2 + (p0 - l0) ** 2
    )


def _iso(day: date) -> str:
    return day.isoformat()


def _window(value: list[str]) -> Window:
    first, last = date.fromisoformat(value[0]), date.fromisoformat(value[1])
    if first > last:
        raise ToolError("the range ends before it starts")
    return first, last


def _within(day: date, window: Window) -> bool:
    return window[0] <= day <= window[1]


class Tools:
    """The tools on one loaded bundle, recording into one trace. The model's two tools need their
    sources: `search` (Qdrant and the query encoder) and `steam` (the shared Steam client); without
    one, its tool answers that it isn't available in this run."""

    def __init__(
        self,
        bundle: Bundle,
        trace: Trace,
        *,
        search: ReviewSearch | None = None,
        steam: SteamHttp | None = None,
    ) -> None:
        self.bundle = bundle
        self.trace = trace
        self._search = search
        self._steam = steam
        self._games = {g.appid: g for g in bundle.games}
        self._patches = {p.patch_id: (g, p) for g in bundle.games for p in g.patches}
        self._runners: dict[str, Callable[[dict[str, Any]], Result]] = {
            "get_game_overview": self._overview,
            "list_patches": self._list_patches,
            "sentiment_timeseries": self._timeseries,
            "aspect_breakdown": self._aspects,
            "estimate_patch_effect": self._estimate,
            "decompose_change": self._decompose,
            "search_reviews": self._search_reviews,
            "get_patch_notes": self._patch_notes,
        }

    # Recording and re-execution.

    def _execute(self, tool: str, args: dict[str, Any]) -> Result:
        try:
            return self._runners[tool](args)
        except ToolError as error:
            return {"error": str(error)}

    def _call(self, tool: str, args: dict[str, Any]) -> Result:
        result = self._execute(tool, args)
        call_id = f"{self.trace.run}.{len(self.trace.calls) + 1}"
        self.trace.calls.append(ToolCall(call_id, tool, args, result))
        return result

    def rerun(self, call: ToolCall) -> Result:
        """The recorded call executed again on this bundle, without recording it (V2)."""
        return self._execute(call.tool, call.args)

    def estimate_rows(self, call: ToolCall) -> list[StoredEstimate]:
        """The stored rows an estimate_patch_effect call served, unrounded (for the causal
        gate); none for another tool or an error."""
        if call.tool != "estimate_patch_effect":
            return []
        try:
            return self._served(call.args)[1]
        except ToolError:
            return []

    # The tools, as the agent calls them.

    def get_game_overview(self, appid: int) -> Result:
        return self._call("get_game_overview", {"appid": appid})

    def list_patches(self, appid: int, start: date, end: date) -> Result:
        return self._call("list_patches", {"appid": appid, "start": _iso(start), "end": _iso(end)})

    def sentiment_timeseries(
        self, appid: int, start: date, end: date, granularity: str = "day"
    ) -> Result:
        return self._call(
            "sentiment_timeseries",
            {"appid": appid, "start": _iso(start), "end": _iso(end), "granularity": granularity},
        )

    def aspect_breakdown(self, appid: int, window: Window, baseline_window: Window) -> Result:
        return self._call(
            "aspect_breakdown",
            {
                "appid": appid,
                "window": [_iso(d) for d in window],
                "baseline_window": [_iso(d) for d in baseline_window],
            },
        )

    def estimate_patch_effect(
        self,
        patch_id: str,
        outcome: str = POSITIVE_SHARE,
        method: str | None = None,
        *,
        sample: str = "all",
    ) -> Result:
        return self._call(
            "estimate_patch_effect",
            {"patch_id": patch_id, "outcome": outcome, "method": method, "sample": sample},
        )

    def decompose_change(self, appid: int, window: Window, baseline_window: Window) -> Result:
        return self._call(
            "decompose_change",
            {
                "appid": appid,
                "window": [_iso(d) for d in window],
                "baseline_window": [_iso(d) for d in baseline_window],
            },
        )

    def search_reviews(
        self,
        appid: int,
        query: str,
        window: Window,
        *,
        voted_up: bool | None = None,
        aspect: str | None = None,
        language: str | None = None,
        min_playtime: float | None = None,
        k: int = DEFAULT_HITS,
    ) -> Result:
        return self._call(
            "search_reviews",
            {
                "appid": appid,
                "query": query,
                "window": [_iso(d) for d in window],
                "voted_up": voted_up,
                "aspect": aspect,
                "language": language,
                "min_playtime": min_playtime,
                "k": k,
            },
        )

    def get_patch_notes(self, patch_id: str) -> Result:
        return self._call("get_patch_notes", {"patch_id": patch_id})

    # The implementations, on JSON-able args.

    def _search_reviews(self, args: dict[str, Any]) -> Result:
        game = self._game(args["appid"])
        window = _window(args["window"])
        if self._search is None:
            raise ToolError("review search isn't available in this run")
        try:
            hits = find_reviews(
                self._search,
                appid=game.appid,
                query=args["query"],
                window=window,
                as_of_run_id=self.bundle.data_version,
                voted_up=args["voted_up"],
                aspect=args["aspect"],
                language=args["language"],
                min_playtime=args["min_playtime"],
                k=args["k"],
            )
        except ValueError as error:  # a filter the index can't apply
            raise ToolError(str(error)) from error
        for hit in hits:
            self.trace.retrieved[hit.review_id] = hit.text
        return {
            "appid": game.appid,
            "query": args["query"],
            "window": args["window"],
            "hits": [
                {"review_id": h.review_id, "score": round(h.score, 4), "snippet": snippet(h.text)}
                for h in hits
            ],
        }

    def _patch_notes(self, args: dict[str, Any]) -> Result:
        found = self._patches.get(args["patch_id"])
        if found is None:
            raise ToolError(f"no patch {args['patch_id']} in the data")
        game, patch = found
        if self._steam is None:
            raise ToolError("patch notes aren't available in this run")
        try:
            notes = fetch_patch_notes(
                self._steam, appid=game.appid, gid=patch.gid, published_at=patch.published_at
            )
        except SteamError as error:
            raise ToolError("Steam's news service didn't answer") from error
        return {
            "patch_id": patch.patch_id,
            "title": patch.title,
            "day": _iso(patch.day),
            "found": notes.found,
            "notes": notes.text,
            "truncated": notes.truncated,
        }

    def _game(self, appid: int) -> GameData:
        game = self._games.get(appid)
        if game is None:
            raise ToolError(f"game {appid} isn't tracked")
        return game

    def _overview(self, args: dict[str, Any]) -> Result:
        game = self._game(args["appid"])
        positives = sum(d.positives for d in game.days)
        totals = sum(d.totals for d in game.days)
        return {
            "appid": game.appid,
            "name": game.name,
            "genres": [g.strip() for g in game.genres.split(",") if g.strip()],
            "first_day": _iso(game.days[0].date) if game.days else None,
            "last_day": _iso(game.days[-1].date) if game.days else None,
            "reviews": totals,
            "positives": positives,
            "share": _pct(positives, totals),
            "posts": len(game.patches),
            "patches_with_estimates": sum(1 for p in game.patches if p.bundle_id is not None),
        }

    def _list_patches(self, args: dict[str, Any]) -> Result:
        game = self._game(args["appid"])
        window = _window([args["start"], args["end"]])
        return {
            "appid": game.appid,
            "start": args["start"],
            "end": args["end"],
            "patches": [
                {
                    "patch_id": p.patch_id,
                    "day": _iso(p.day),
                    "title": p.title,
                    "type": p.patch_type,
                    "bundle_id": p.bundle_id,
                }
                for p in game.patches
                if _within(p.day, window)
            ],
        }

    def _timeseries(self, args: dict[str, Any]) -> Result:
        game = self._game(args["appid"])
        first, last = _window([args["start"], args["end"]])
        granularity = args["granularity"]
        if granularity not in ("day", "week"):
            raise ToolError(f"granularity must be day or week, not {granularity!r}")
        step = 1 if granularity == "day" else 7
        points = []
        start = first
        while start <= last:
            end = min(start + timedelta(days=step - 1), last)
            points.append(self._period(game, (start, end)))
            start = end + timedelta(days=1)
        return {
            "appid": game.appid,
            "start": args["start"],
            "end": args["end"],
            "granularity": granularity,
            "points": points,
            "total": self._period(game, (first, last)),
        }

    @staticmethod
    def _period(game: GameData, window: Window) -> Result:
        days = [d for d in game.days if _within(d.date, window)]
        positives = sum(d.positives for d in days)
        totals = sum(d.totals for d in days)
        new = sum(
            b.totals for b in game.buckets if b.bucket == NEW_PLAYERS and _within(b.date, window)
        )
        interval = wilson(positives, totals)
        known = [d.discount for d in days if d.discount is not None]
        span = (window[1] - window[0]).days + 1
        return {
            "start": _iso(window[0]),
            "end": _iso(window[1]),
            "reviews": totals,
            "positives": positives,
            "share": _pct(positives, totals),
            "lo90": None if interval is None else _round(100 * interval[0]),
            "hi90": None if interval is None else _round(100 * interval[1]),
            "new_players": new,
            "new_players_share": _pct(new, totals),
            "sale_days": sum(1 for d in days if d.sale),
            "discount_days": sum(1 for v in known if v > 0),
            "max_discount": max(known) if known else None,
            "price_unknown_days": span - len(known),
        }

    def _aspects(self, args: dict[str, Any]) -> Result:
        game = self._game(args["appid"])
        window, baseline = _window(args["window"]), _window(args["baseline_window"])
        sums: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
        for a in game.aspects:
            for period, span in (("window", window), ("baseline", baseline)):
                if _within(a.date, span):
                    for polarity, mentions, scored in (
                        ("negative", a.neg_mentions, a.neg_scored),
                        ("positive", a.pos_mentions, a.pos_scored),
                    ):
                        counts = sums[(a.aspect, polarity, period)]
                        counts[0] += mentions
                        counts[1] += scored

        def side(code: str, polarity: str) -> Result:
            x1, n1 = sums.get((code, polarity, "window"), (0, 0))
            x0, n0 = sums.get((code, polarity, "baseline"), (0, 0))
            interval = newcombe(x1, n1, x0, n0)
            return {
                "window_mentions": x1,
                "window_scored": n1,
                "window_share": _pct(x1, n1),
                "baseline_mentions": x0,
                "baseline_scored": n0,
                "baseline_share": _pct(x0, n0),
                "change": None if interval is None else _round(100 * (x1 / n1 - x0 / n0)),
                "lo90": None if interval is None else _round(100 * interval[0]),
                "hi90": None if interval is None else _round(100 * interval[1]),
            }

        return {
            "appid": game.appid,
            "window": args["window"],
            "baseline_window": args["baseline_window"],
            "aspects": {
                code: {"negative": side(code, "negative"), "positive": side(code, "positive")}
                for code in ASPECTS
            },
        }

    def _served(self, args: dict[str, Any]) -> tuple[str, list[StoredEstimate]]:
        found = self._patches.get(args["patch_id"])
        if found is None:
            raise ToolError(f"no patch {args['patch_id']} in the data")
        game, patch = found
        if patch.bundle_id is None:
            raise ToolError(
                f"patch {patch.patch_id} ({patch.patch_type}) isn't a treatment candidate, "
                "so it has no estimate"
            )
        chosen = args["method"] or self.bundle.causal.default_method
        rows = [
            e
            for e in game.estimates
            if (e.bundle_id, e.outcome, e.sample)
            == (patch.bundle_id, args["outcome"], args["sample"])
            and e.method in {chosen, "its"}
        ]
        return patch.bundle_id, serve(rows, chosen, fallback=args["method"] is None)

    def _estimate(self, args: dict[str, Any]) -> Result:
        bundle_id, served = self._served(args)
        game, _ = self._patches[args["patch_id"]]
        members = sorted(
            (p for p in game.patches if p.bundle_id == bundle_id), key=lambda p: (p.day, p.patch_id)
        )
        return {
            "patch_id": args["patch_id"],
            "bundle_id": bundle_id,
            "bundle_start": _iso(members[0].day),
            "patch_ids": [p.patch_id for p in members],
            "outcome": args["outcome"],
            "sample": args["sample"],
            "default_method": self.bundle.causal.default_method,
            "horizons": {str(e.horizon): _estimate_view(e) for e in served},
        }

    def _decompose(self, args: dict[str, Any]) -> Result:
        game = self._game(args["appid"])
        window, baseline = _window(args["window"]), _window(args["baseline_window"])
        before, after = _bucket_counts(game, baseline), _bucket_counts(game, window)
        try:
            split = decompose(before, after)
        except ValueError as error:  # an empty window
            raise ToolError(str(error)) from error
        buckets: dict[str, Result] = {}
        present = set(before) | set(after)
        for bucket in [b for b in BUCKETS if b in present] + sorted(present - set(BUCKETS)):
            p0, n0 = before.get(bucket, (0, 0))
            p1, n1 = after.get(bucket, (0, 0))
            buckets[bucket] = {
                "baseline_reviews": n0,
                "window_reviews": n1,
                "baseline_mix": _pct(n0, split.baseline_n),
                "window_mix": _pct(n1, split.window_n),
                "baseline_share": _pct(p0, n0),
                "window_share": _pct(p1, n1),
            }
        return {
            "appid": game.appid,
            "window": args["window"],
            "baseline_window": args["baseline_window"],
            **{part: _round(getattr(split, part)) for part in PARTS},
            "intervals": {part: [_round(v) for v in split.intervals[part]] for part in PARTS},
            "baseline_reviews": split.baseline_n,
            "window_reviews": split.window_n,
            "buckets": buckets,
        }


def _bucket_counts(game: GameData, window: Window) -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    for b in game.buckets:
        if _within(b.date, window):
            positives, totals = counts.get(b.bucket, (0, 0))
            counts[b.bucket] = (positives + b.positives, totals + b.totals)
    return {k: v for k, v in counts.items() if v[1] > 0}


def _estimate_view(row: StoredEstimate) -> Result:
    return {
        "horizon": row.horizon,
        "method": row.method,
        "effect": _round(row.effect),
        "lo90": _round(row.lo90),
        "hi90": _round(row.hi90),
        "p_value": _round(row.p_value, 3),
        "pre_fit_rmse": _round(row.pre_fit_rmse),
        "donors": [[appid, round(weight, 3)] for appid, weight in row.donors],
        "flags": sorted(row.flags),
        "reason": row.reason,
        "run_id": row.run_id,
    }
