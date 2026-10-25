"""The core tools (plan Task 3, design D3): compact, JSON-able results read from the bundle, each
call recorded in the run's trace and re-executable for V2."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest
from agent_fixtures import GAME, NO_SYNTH_14, OTHER_GAME, PATCH_DAY, game_data, make_bundle
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint

from patchpulse.agent.bundle import AspectRow, BucketRow, DayRow, GameData
from patchpulse.agent.tools import Tools, Trace
from patchpulse.causal.decompose import decompose
from patchpulse.causal.store import NOT_RUN, StoredEstimate, serve
from patchpulse.labeling.taxonomy import ASPECTS

D1, D2, D3 = date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 3)


def tiny_game(
    *,
    days: tuple[DayRow, ...] = (),
    aspects: tuple[AspectRow, ...] = (),
    buckets: tuple[BucketRow, ...] = (),
) -> GameData:
    return GameData(
        appid=GAME,
        name="Helldivers 2",
        genres="Action, Shooter",
        days=days,
        aspects=aspects,
        buckets=buckets,
        patches=(),
        estimates=(),
    )


def json_able(result: dict[str, object]) -> None:
    json.dumps(result, allow_nan=False)


def test_sentiment_timeseries_matches_hand_computed_wilson_intervals() -> None:
    game = tiny_game(
        days=(
            DayRow(D1, 8, 10, sale=True, discount=25),
            DayRow(D2, 0, 0, sale=False, discount=0),
            DayRow(D3, 5, 5, sale=False, discount=None),
        ),
        buckets=(
            BucketRow(D1, "lt_2h", 3, 4),
            BucketRow(D1, "10h_50h", 5, 6),
            BucketRow(D3, "lt_2h", 1, 1),
            BucketRow(D3, "unknown", 4, 4),
        ),
    )
    tools = Tools(make_bundle(game), Trace())

    daily = tools.sentiment_timeseries(GAME, D1, D3, "day")
    weekly = tools.sentiment_timeseries(GAME, D1, D3, "week")

    first, empty, last = daily["points"]
    # Wilson at 90% (z = 1.6449, the marts' macro): 8 of 10 gives 0.5408 to 0.9314 by hand.
    assert (first["start"], first["end"]) == ("2026-10-01", "2026-10-01")
    assert (first["reviews"], first["positives"], first["share"]) == (10, 8, 80.0)
    assert (first["lo90"], first["hi90"]) == (54.08, 93.14)
    assert (first["new_players"], first["new_players_share"]) == (4, 40.0)
    assert (first["sale_days"], first["discount_days"], first["max_discount"]) == (1, 1, 25)
    assert empty["reviews"] == 0
    assert (empty["share"], empty["lo90"], empty["hi90"], empty["new_players_share"]) == (
        None,
        None,
        None,
        None,
    )
    assert (empty["discount_days"], empty["max_discount"], empty["price_unknown_days"]) == (0, 0, 0)
    assert last["price_unknown_days"] == 1
    assert last["max_discount"] is None

    (week,) = weekly["points"]
    lo, hi = proportion_confint(13, 15, alpha=0.10, method="wilson")
    assert (week["start"], week["end"], week["reviews"], week["positives"]) == (
        "2026-10-01",
        "2026-10-03",
        15,
        13,
    )
    assert week["lo90"] == pytest.approx(100 * lo, abs=0.01)
    assert week["hi90"] == pytest.approx(100 * hi, abs=0.01)
    assert (week["sale_days"], week["discount_days"], week["price_unknown_days"]) == (1, 1, 1)
    assert weekly["total"] == {k: v for k, v in week.items()}
    json_able(daily)


def test_weeks_count_seven_days_from_the_start() -> None:
    tools = Tools(make_bundle(), Trace())

    first, last = PATCH_DAY - timedelta(days=14), PATCH_DAY + timedelta(days=6)
    result = tools.sentiment_timeseries(GAME, first, last, "week")

    spans = [(p["start"], p["end"]) for p in result["points"]]
    assert spans == [
        ("2026-08-27", "2026-09-02"),
        ("2026-09-03", "2026-09-09"),
        ("2026-09-10", "2026-09-16"),
    ]
    assert result["total"]["reviews"] == sum(p["reviews"] for p in result["points"])


def test_aspect_breakdown_two_proportion_intervals() -> None:
    def row(day: date, mentions: int, scored: int) -> AspectRow:
        return AspectRow(
            day, "performance", neg_mentions=mentions, neg_scored=scored, pos_mentions=0,
            pos_scored=0,
        )  # fmt: skip

    game = tiny_game(aspects=(row(D1, 2, 25), row(D2, 4, 12), row(D3, 2, 8)))
    tools = Tools(make_bundle(game), Trace())

    result = tools.aspect_breakdown(GAME, (D2, D3), (D1, D1))

    negative = result["aspects"]["performance"]["negative"]
    assert (negative["window_mentions"], negative["window_scored"]) == (6, 20)
    assert (negative["baseline_mentions"], negative["baseline_scored"]) == (2, 25)
    assert (negative["window_share"], negative["baseline_share"], negative["change"]) == (
        30.0,
        8.0,
        22.0,
    )
    # Newcombe's interval from the two Wilson intervals, worked by hand: 2.81 to 41.13 points.
    assert (negative["lo90"], negative["hi90"]) == (2.81, 41.13)
    lo, hi = confint_proportions_2indep(6, 20, 2, 25, method="newcomb", compare="diff", alpha=0.1)
    assert negative["lo90"] == pytest.approx(100 * lo, abs=0.01)
    assert negative["hi90"] == pytest.approx(100 * hi, abs=0.01)
    positive = result["aspects"]["performance"]["positive"]
    assert (positive["window_share"], positive["change"], positive["lo90"]) == (None, None, None)
    assert list(result["aspects"]) == list(ASPECTS)  # every aspect, unmentioned ones at zero
    assert result["aspects"]["content"]["negative"]["window_scored"] == 0
    json_able(result)


def test_estimate_patch_effect_serves_like_the_store() -> None:
    bundle = make_bundle()
    rows = game_data().estimates
    tools = Tools(bundle, Trace())

    served = tools.estimate_patch_effect("p2")  # a patch inside bundle p1
    explicit = tools.estimate_patch_effect("p1", method="its")
    asked_synth = tools.estimate_patch_effect("p1", method="synth")
    aspect = tools.estimate_patch_effect("p1", "aspect:performance")
    established = tools.estimate_patch_effect("p1", sample="established")

    def same(outcome: str, sample: str) -> list[StoredEstimate]:
        return [r for r in rows if (r.outcome, r.sample) == (outcome, sample)]

    calls = tools.trace.calls
    assert tools.estimate_rows(calls[0]) == serve(
        same("positive_share", "all"), "synth", fallback=True
    )
    assert tools.estimate_rows(calls[1]) == serve(
        same("positive_share", "all"), "its", fallback=False
    )
    assert (served["bundle_id"], served["bundle_start"], served["patch_ids"]) == (
        "p1",
        "2026-09-10",
        ["p1", "p2"],
    )
    assert served["default_method"] == "synth"
    seven, fourteen = served["horizons"]["7"], served["horizons"]["14"]
    assert (seven["method"], seven["effect"], seven["lo90"], seven["hi90"]) == (
        "synth",
        -12.25,
        -15.35,
        -9.75,
    )
    assert (seven["p_value"], seven["flags"], seven["donors"]) == (
        0.048,
        ["volume_spike"],
        [[1086940, 0.6], [892970, 0.4]],
    )
    assert seven["run_id"] == 51
    # synthetic control has no 14-day estimate: interrupted time series with synth's reason
    assert (fourteen["method"], fourteen["effect"]) == ("its", -10.0)
    assert fourteen["reason"].startswith("the comparison games are checked")
    assert [h["method"] for h in explicit["horizons"].values()] == ["its", "its"]
    # a method asked for by name is served as it is, without the fallback
    no_estimate = asked_synth["horizons"]["14"]
    assert (no_estimate["method"], no_estimate["effect"], no_estimate["reason"]) == (
        "synth",
        None,
        NO_SYNTH_14,
    )
    assert list(aspect["horizons"]) == ["7"]
    assert (aspect["horizons"]["7"]["method"], aspect["horizons"]["7"]["reason"]) == (
        "its",
        NOT_RUN,
    )
    assert established["horizons"]["7"]["effect"] == -6.5
    json_able(served)


def test_decompose_change_matches_the_library() -> None:
    bundle = make_bundle()
    window = (PATCH_DAY, PATCH_DAY + timedelta(days=6))
    baseline = (PATCH_DAY - timedelta(days=28), PATCH_DAY - timedelta(days=1))

    def counts(first: date, last: date) -> dict[str, tuple[int, int]]:
        summed: dict[str, tuple[int, int]] = {}
        for b in game_data().buckets:
            if first <= b.date <= last:
                p, n = summed.get(b.bucket, (0, 0))
                summed[b.bucket] = (p + b.positives, n + b.totals)
        return summed

    expected = decompose(counts(*baseline), counts(*window))

    result = Tools(bundle, Trace()).decompose_change(GAME, window, baseline)

    for part in ("total", "composition", "within", "interaction"):
        assert result[part] == round(getattr(expected, part), 2)
        assert result["intervals"][part] == [round(v, 2) for v in expected.intervals[part]]
    assert (result["baseline_reviews"], result["window_reviews"]) == (
        expected.baseline_n,
        expected.window_n,
    )
    lt_2h = result["buckets"]["lt_2h"]
    window_counts, baseline_counts = counts(*window)["lt_2h"], counts(*baseline)["lt_2h"]
    assert lt_2h["window_reviews"] == window_counts[1]
    assert lt_2h["window_mix"] == round(100 * window_counts[1] / expected.window_n, 2)
    assert lt_2h["baseline_share"] == round(100 * baseline_counts[0] / baseline_counts[1], 2)
    assert "2h_10h" not in result["buckets"]  # no reviews in either window
    json_able(result)


def test_every_call_is_recorded_and_reruns_identically() -> None:
    trace = Trace(run="r7")
    tools = Tools(make_bundle(), trace)
    window = (PATCH_DAY, PATCH_DAY + timedelta(days=6))
    baseline = (PATCH_DAY - timedelta(days=28), PATCH_DAY - timedelta(days=1))

    results = [
        tools.get_game_overview(GAME),
        tools.list_patches(GAME, PATCH_DAY - timedelta(days=7), PATCH_DAY + timedelta(days=7)),
        tools.sentiment_timeseries(GAME, *baseline, "week"),
        tools.aspect_breakdown(GAME, window, baseline),
        tools.estimate_patch_effect("p1"),
        tools.decompose_change(GAME, window, baseline),
    ]

    assert [c.call_id for c in trace.calls] == ["r7.1", "r7.2", "r7.3", "r7.4", "r7.5", "r7.6"]
    assert [c.tool for c in trace.calls] == [
        "get_game_overview",
        "list_patches",
        "sentiment_timeseries",
        "aspect_breakdown",
        "estimate_patch_effect",
        "decompose_change",
    ]
    assert [c.result for c in trace.calls] == results
    assert trace.calls[3].args == {
        "appid": GAME,
        "window": ["2026-09-10", "2026-09-16"],
        "baseline_window": ["2026-08-13", "2026-09-09"],
    }
    for call in trace.calls:
        json.dumps(call.args)
        assert tools.rerun(call) == call.result
    assert len(trace.calls) == 6  # reruns aren't recorded
    assert trace.find("r7.5") is trace.calls[4]
    assert trace.find("r6.5") is None
    overview = results[0]
    assert (overview["name"], overview["genres"], overview["first_day"], overview["last_day"]) == (
        "Helldivers 2",
        ["Action", "Shooter"],
        "2026-08-01",
        "2026-10-10",
    )
    assert [p["patch_id"] for p in results[1]["patches"]] == ["p1", "e1", "p2"]
    assert results[1]["patches"][1] == {
        "patch_id": "e1",
        "day": "2026-09-11",
        "title": "Autumn event",
        "type": "event",
        "bundle_id": None,
    }


def test_tools_reject_untracked_games_and_unknown_patches() -> None:
    trace = Trace()
    tools = Tools(make_bundle(), trace)
    untracked = 1086940
    window = (PATCH_DAY, PATCH_DAY)

    errors = [
        tools.get_game_overview(untracked),
        tools.list_patches(untracked, PATCH_DAY, PATCH_DAY),
        tools.sentiment_timeseries(untracked, PATCH_DAY, PATCH_DAY),
        tools.aspect_breakdown(untracked, window, window),
        tools.decompose_change(untracked, window, window),
        tools.estimate_patch_effect("999"),
        tools.estimate_patch_effect("e1"),
        tools.sentiment_timeseries(GAME, PATCH_DAY, PATCH_DAY - timedelta(days=1)),
        tools.sentiment_timeseries(GAME, PATCH_DAY, PATCH_DAY, "month"),
        tools.decompose_change(OTHER_GAME, window, window),
    ]

    assert all(set(e) == {"error"} for e in errors)
    assert errors[0]["error"] == "game 1086940 isn't tracked"
    assert errors[5]["error"] == "no patch 999 in the data"
    assert (
        errors[6]["error"] == "patch e1 (event) isn't a treatment candidate, so it has no estimate"
    )
    assert errors[7]["error"] == "the range ends before it starts"
    assert errors[8]["error"] == "granularity must be day or week, not 'month'"
    assert errors[9]["error"] == "no reviews in the baseline window"
    assert [c.result for c in trace.calls] == errors  # errors are recorded like results
