"""The H1-H5 rules (plan Task 5, design D5): each hypothesis's status set by a fixed rule from the
core evidence, never by the model, with the rule's id and the values it read."""

from __future__ import annotations

import itertools
import json
from dataclasses import replace
from typing import Any

import pytest
from agent_fixtures import estimate

from patchpulse.agent.gate import causal_gate
from patchpulse.agent.rules import CoreEvidence, HypothesisResult, decide
from patchpulse.causal.config import CausalConfig
from patchpulse.causal.store import StoredEstimate

CONFIG = CausalConfig("its", "e2", "ab" * 32, validated=True)


def row(
    lo: float | None, hi: float | None, *, method: str = "its", p: float | None = None,
    flags: tuple[str, ...] = (),
) -> StoredEstimate:  # fmt: skip
    effect = None if lo is None or hi is None else (lo + hi) / 2
    return replace(estimate(7, method, effect), lo90=lo, hi90=hi, p_value=p, flags=frozenset(flags))


def period(
    reviews: int, new_players: int, *, sale: int = 0, discount: int = 0, unknown: int = 0
) -> dict[str, Any]:
    return {
        "reviews": reviews,
        "new_players": new_players,
        "sale_days": sale,
        "discount_days": discount,
        "price_unknown_days": unknown,
    }


def off_topic(change: float | None, lo: float | None, hi: float | None) -> dict[str, Any]:
    side = {"change": change, "lo90": lo, "hi90": hi}
    return {"aspects": {"off_topic": {"negative": side, "positive": side}}}


def split(
    total: float, total_iv: tuple[float, float], composition: float, comp_iv: tuple[float, float]
) -> dict[str, Any]:
    return {
        "total": total,
        "composition": composition,
        "intervals": {"total": list(total_iv), "composition": list(comp_iv)},
    }


QUIET = CoreEvidence(
    estimate=row(-1.0, 1.5),
    window=period(200, 40),
    baseline=period(800, 160),
    near_start=period(150, 30),
    aspects=off_topic(0.5, -3.0, 4.0),
    decomposition=split(0.4, (-1.5, 2.3), 0.1, (-0.4, 0.6)),
)


def status(evidence: CoreEvidence, code: str, config: CausalConfig = CONFIG) -> tuple[str, str]:
    (result,) = [r for r in decide(evidence, config) if r.code == code]
    return result.status, result.rule


def test_quiet_evidence_points_to_noise() -> None:
    results = decide(QUIET, CONFIG)

    assert [(r.code, r.status) for r in results] == [
        ("H1", "rejected"),
        ("H2", "rejected"),
        ("H3", "rejected"),
        ("H4", "rejected"),
        ("H5", "supported"),
    ]


def test_h1_the_patch() -> None:
    assert status(replace(QUIET, estimate=row(-14.0, -9.0)), "H1") == (
        "supported",
        "H1.gate_allows",
    )
    assert status(QUIET, "H1") == ("rejected", "H1.null_band")
    edge = replace(QUIET, estimate=row(-2.0, 2.0))  # the band's ends are inside it
    assert status(edge, "H1") == ("rejected", "H1.null_band")
    just_out = replace(QUIET, estimate=row(-2.5, 1.0))
    assert status(just_out, "H1") == ("inconclusive", "H1.wide_range")
    assert status(replace(QUIET, estimate=row(-6.0, 3.0)), "H1") == (
        "inconclusive",
        "H1.wide_range",
    )
    confounded = row(-14.0, -9.0, flags=("confounded",))
    assert status(replace(QUIET, estimate=confounded), "H1") == (
        "inconclusive",
        "H1.blocked.confounded",
    )
    assert status(replace(QUIET, estimate=None), "H1") == ("inconclusive", "H1.no_estimate")
    assert status(replace(QUIET, estimate=row(None, None)), "H1") == (
        "inconclusive",
        "H1.no_estimate",
    )


def test_h2_a_price_change_or_sale() -> None:
    sale = period(150, 30, sale=1)
    new_players_up = replace(QUIET, near_start=sale, window=period(200, 100))  # 20% -> 50%
    assert status(new_players_up, "H2") == ("supported", "H2.sale_and_mix")
    mix_moved = split(0.4, (-1.5, 2.3), -2.0, (-3.0, -1.0))
    assert status(replace(QUIET, near_start=sale, decomposition=mix_moved), "H2") == (
        "supported",
        "H2.sale_and_mix",
    )
    assert status(QUIET, "H2") == ("rejected", "H2.no_sale")
    unknown = period(150, 30, unknown=2)
    assert status(replace(QUIET, near_start=unknown), "H2") == (
        "inconclusive",
        "H2.discounts_unknown",
    )
    assert status(replace(QUIET, near_start=sale), "H2") == ("inconclusive", "H2.mix_steady")
    # 20% -> 24% on many reviews: its interval excludes 0, but it's under the 5-point bar
    small_rise = replace(
        QUIET,
        near_start=period(150, 30, discount=3),
        window=period(10_000, 2_400),
        baseline=period(10_000, 2_000),
    )
    assert status(small_rise, "H2") == ("inconclusive", "H2.mix_steady")


def test_h3_off_topic_reviews_or_a_review_bomb() -> None:
    spike = row(-1.0, 1.5, flags=("volume_spike",))
    risen = off_topic(6.0, 2.0, 10.0)
    assert status(replace(QUIET, estimate=spike, aspects=risen), "H3") == (
        "supported",
        "H3.spike_and_off_topic",
    )
    assert status(QUIET, "H3") == ("rejected", "H3.neither")
    assert status(replace(QUIET, estimate=spike), "H3") == ("inconclusive", "H3.one_moved")
    assert status(replace(QUIET, aspects=risen), "H3") == ("inconclusive", "H3.one_moved")
    assert status(replace(QUIET, estimate=None), "H3") == ("inconclusive", "H3.unknown")
    assert status(replace(QUIET, aspects=off_topic(None, None, None)), "H3") == (
        "inconclusive",
        "H3.unknown",
    )


def test_h4_an_audience_shift_not_explained_by_price() -> None:
    large = split(-4.0, (-6.0, -2.0), -3.0, (-4.5, -1.5))  # 75% of the change
    assert status(replace(QUIET, decomposition=large), "H4") == (
        "supported",
        "H4.composition_large",
    )
    assert status(QUIET, "H4") == ("rejected", "H4.composition_small")
    with_sale = replace(
        QUIET, decomposition=large, near_start=period(150, 30, sale=2), window=period(200, 100)
    )
    assert status(with_sale, "H4") == ("inconclusive", "H4.explained_by_price")
    minor_part = split(-8.0, (-10.0, -6.0), -1.0, (-1.8, -0.2))  # excludes 0, 12% of the change
    assert status(replace(QUIET, decomposition=minor_part), "H4") == (
        "inconclusive",
        "H4.otherwise",
    )
    unsure = split(2.0, (0.5, 3.5), 1.5, (-0.5, 3.5))  # includes 0 but 1.5 points
    assert status(replace(QUIET, decomposition=unsure), "H4") == ("inconclusive", "H4.otherwise")
    empty = {"error": "no reviews in the baseline window"}
    assert status(replace(QUIET, decomposition=empty), "H4") == (
        "inconclusive",
        "H4.no_decomposition",
    )


def test_h5_seasonality_or_noise() -> None:
    assert status(QUIET, "H5") == ("supported", "H5.no_change")
    moved = split(-4.0, (-6.0, -2.0), 0.1, (-0.4, 0.6))
    assert status(replace(QUIET, decomposition=moved), "H5") == (
        "rejected",
        "H5.change_established",
    )
    assert status(replace(QUIET, estimate=row(-14.0, -9.0)), "H5") == (
        "rejected",
        "H5.change_established",
    )
    sale_mix = replace(QUIET, near_start=period(150, 30, sale=1), window=period(200, 100))
    assert status(sale_mix, "H5") == ("inconclusive", "H5.other_explanation")
    empty = {"error": "no reviews in the window"}
    assert status(replace(QUIET, decomposition=empty), "H5") == ("inconclusive", "H5.unknown")
    # no patch estimate: the before/after difference alone decides
    assert status(replace(QUIET, estimate=None), "H5") == ("supported", "H5.no_change")


RANGES = [(-14.0, -9.0), (-4.0, 1.0), (None, None), (9.0, 14.0), (-1.0, 1.0)]
METHODS = [("its", None), ("synth", 0.05), ("synth", 0.5)]
FLAGS = [(), ("confounded",), ("volume_spike",), ("discounts_unknown",)]
CONFIGS = [
    CausalConfig("its", "e2", None, validated=v, spike_causal_ok=s)
    for v, s in itertools.product((True, False), repeat=2)
]


@pytest.mark.parametrize("config", CONFIGS)
def test_h1_is_supported_exactly_when_the_gate_allows(config: CausalConfig) -> None:
    for (lo, hi), (method, p), flags in itertools.product(RANGES, METHODS, FLAGS):
        served = row(lo, hi, method=method, p=p, flags=flags)
        h1 = decide(replace(QUIET, estimate=served), config)[0]
        assert (h1.status == "supported") == causal_gate(served, config).allowed, (served, config)


def test_rules_record_the_evidence_they_read() -> None:
    results = decide(QUIET, CONFIG)

    by_code = {r.code: r for r in results}
    assert by_code["H1"].evidence == {
        "effect": 0.25,
        "lo90": -1.0,
        "hi90": 1.5,
        "gate_allowed": False,
    }
    assert by_code["H2"].evidence["sale_days"] == 0
    assert by_code["H2"].evidence["new_players_change"] == 0.0
    assert by_code["H3"].evidence["volume_spike"] is False
    assert by_code["H3"].evidence["off_topic_change"] == 0.5
    assert by_code["H4"].evidence["composition"] == 0.1
    assert (by_code["H5"].evidence["total_lo90"], by_code["H5"].evidence["total_hi90"]) == (
        -1.5,
        2.3,
    )
    for result in results:
        assert result.rule.startswith(result.code + ".")
        assert all(isinstance(v, float | int | bool) or v is None for v in result.evidence.values())
        json.dumps(result.model_dump())
    assert HypothesisResult.model_validate(results[0].model_dump()) == results[0]
