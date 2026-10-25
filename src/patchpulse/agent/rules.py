"""The hypothesis rules (design D5, spec §10.3): H1-H5 set from the core evidence by fixed rules,
never by the model. Each result names the rule that set it and the values it read.

The thresholds are below, each with its reason; Seif reviews them before the first acceptance run,
and after Phase 6 has graded the golden set they change only through a ledgered ruling with a
re-run (the discipline of ADR-007). Intervals are 90%; changes are in percentage points, window
against baseline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from patchpulse.agent.gate import causal_gate
from patchpulse.agent.tools import newcombe
from patchpulse.causal.config import CausalConfig
from patchpulse.causal.store import StoredEstimate

# H1: a range inside +-2 points around 0 rules out an effect the site would call meaningful (the
# positive share's day-to-day noise on a game with tens of reviews a day is several points).
H1_NULL_BAND_PTS = 2.0
# H2: new players' share up by 5 points or more: a sale's audience, not week-to-week drift.
H2_NEW_PLAYERS_PTS = 5.0
# H4: the mix of reviewers explains at least half of the change in the positive share.
H4_COMPOSITION_SHARE = 0.5
# H4: a composition part under 1 point is too small to matter for the share.
H4_SMALL_PTS = 1.0

Status = Literal["supported", "rejected", "inconclusive"]
Evidence = dict[str, float | bool | None]
Result = dict[str, Any]


class HypothesisResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: Literal["H1", "H2", "H3", "H4", "H5"]
    status: Status
    rule: str  # "<code>.<rule id>": which branch of the rule set the status
    evidence: Evidence


@dataclass(frozen=True)
class CoreEvidence:
    """The core tool results (design D4, step 2) the rules read.

    `estimate` is the served positive-share row at the report's horizon (None without a patch or
    an estimate); `window`, `baseline` and `near_start` are sentiment_timeseries totals over the
    window, the baseline and the 7 days around the bundle's start (or the window's first day);
    `aspects` and `decomposition` are aspect_breakdown and decompose_change over the window
    against the baseline (decomposition may be an error result). `call_ids` names the trace's
    call behind each of them (keys: estimate, window, baseline, near_start, aspects,
    decomposition), so the basic report's claims can cite them."""

    estimate: StoredEstimate | None
    window: Result
    baseline: Result
    near_start: Result
    aspects: Result
    decomposition: Result
    call_ids: Mapping[str, str] = field(default_factory=dict)


def _excludes_zero(lo: float | None, hi: float | None) -> bool | None:
    if lo is None or hi is None:
        return None
    return not lo <= 0 <= hi


def _interval(result: Result, part: str) -> tuple[float | None, float | None]:
    if "error" in result:
        return None, None
    lo, hi = result["intervals"][part]
    return lo, hi


def _value(result: Result, part: str) -> float | None:
    return None if "error" in result else result[part]


def _h1(evidence: CoreEvidence, config: CausalConfig) -> HypothesisResult:
    row = evidence.estimate
    if row is None or row.effect is None or row.lo90 is None or row.hi90 is None:
        values: Evidence = {"effect": None, "lo90": None, "hi90": None, "gate_allowed": False}
        return HypothesisResult(
            code="H1", status="inconclusive", rule="H1.no_estimate", evidence=values
        )
    gate = causal_gate(row, config)
    values = {
        "effect": row.effect,
        "lo90": row.lo90,
        "hi90": row.hi90,
        "gate_allowed": gate.allowed,
    }
    if gate.allowed:
        status: Status = "supported"
        rule = "H1.gate_allows"
    elif -H1_NULL_BAND_PTS <= row.lo90 <= 0 <= row.hi90 <= H1_NULL_BAND_PTS:
        status, rule = "rejected", "H1.null_band"
    elif row.lo90 <= 0 <= row.hi90:
        status, rule = "inconclusive", "H1.wide_range"
    else:
        status, rule = "inconclusive", f"H1.blocked.{gate.blocked_by}"
    return HypothesisResult(code="H1", status=status, rule=rule, evidence=values)


def _new_players(evidence: CoreEvidence) -> tuple[float | None, float | None, float | None]:
    """The change in new players' share (points) and its interval."""
    w, b = evidence.window, evidence.baseline
    interval = newcombe(w["new_players"], w["reviews"], b["new_players"], b["reviews"])
    if interval is None:
        return None, None, None
    change = 100 * (w["new_players"] / w["reviews"] - b["new_players"] / b["reviews"])
    return round(change, 2), round(100 * interval[0], 2), round(100 * interval[1], 2)


def _h2(evidence: CoreEvidence) -> HypothesisResult:
    near = evidence.near_start
    change, lo, hi = _new_players(evidence)
    comp_lo, comp_hi = _interval(evidence.decomposition, "composition")
    values: Evidence = {
        "sale_days": near["sale_days"],
        "discount_days": near["discount_days"],
        "price_unknown_days": near["price_unknown_days"],
        "new_players_change": change,
        "new_players_lo90": lo,
        "new_players_hi90": hi,
        "composition_lo90": comp_lo,
        "composition_hi90": comp_hi,
    }
    on_sale = near["sale_days"] > 0 or near["discount_days"] > 0
    new_players_up = (
        change is not None and lo is not None and change >= H2_NEW_PLAYERS_PTS and lo > 0
    )
    mix_moved = new_players_up or _excludes_zero(comp_lo, comp_hi) is True
    status: Status
    if on_sale and mix_moved:
        status, rule = "supported", "H2.sale_and_mix"
    elif not on_sale and near["price_unknown_days"] == 0:
        status, rule = "rejected", "H2.no_sale"
    elif not on_sale:
        status, rule = "inconclusive", "H2.discounts_unknown"
    else:
        status, rule = "inconclusive", "H2.mix_steady"
    return HypothesisResult(code="H2", status=status, rule=rule, evidence=values)


def _h3(evidence: CoreEvidence) -> HypothesisResult:
    spike = None if evidence.estimate is None else "volume_spike" in evidence.estimate.flags
    off_topic = evidence.aspects["aspects"]["off_topic"]["negative"]
    lo, hi = off_topic["lo90"], off_topic["hi90"]
    risen = None if lo is None or hi is None else lo > 0
    values: Evidence = {
        "volume_spike": spike,
        "off_topic_change": off_topic["change"],
        "off_topic_lo90": lo,
        "off_topic_hi90": hi,
    }
    status: Status
    if spike is True and risen is True:
        status, rule = "supported", "H3.spike_and_off_topic"
    elif spike is False and risen is False:
        status, rule = "rejected", "H3.neither"
    elif spike is None or risen is None:
        status, rule = "inconclusive", "H3.unknown"
    else:
        status, rule = "inconclusive", "H3.one_moved"
    return HypothesisResult(code="H3", status=status, rule=rule, evidence=values)


def _h4(evidence: CoreEvidence, h2: HypothesisResult) -> HypothesisResult:
    composition = _value(evidence.decomposition, "composition")
    total = _value(evidence.decomposition, "total")
    lo, hi = _interval(evidence.decomposition, "composition")
    h2_supported = h2.status == "supported"
    values: Evidence = {
        "composition": composition,
        "composition_lo90": lo,
        "composition_hi90": hi,
        "total_change": total,
        "h2_supported": h2_supported,
    }
    status: Status
    if composition is None or total is None:
        status, rule = "inconclusive", "H4.no_decomposition"
    elif _excludes_zero(lo, hi) and abs(composition) >= H4_COMPOSITION_SHARE * abs(total):
        if h2_supported:
            status, rule = "inconclusive", "H4.explained_by_price"
        else:
            status, rule = "supported", "H4.composition_large"
    elif not _excludes_zero(lo, hi) and abs(composition) < H4_SMALL_PTS:
        status, rule = "rejected", "H4.composition_small"
    else:
        status, rule = "inconclusive", "H4.otherwise"
    return HypothesisResult(code="H4", status=status, rule=rule, evidence=values)


def _h5(evidence: CoreEvidence, others: list[HypothesisResult]) -> HypothesisResult:
    row = evidence.estimate
    effect_lo = None if row is None else row.lo90
    effect_hi = None if row is None else row.hi90
    total_lo, total_hi = _interval(evidence.decomposition, "total")
    other_supported = any(r.status == "supported" for r in others)
    values: Evidence = {
        "effect_lo90": effect_lo,
        "effect_hi90": effect_hi,
        "total_change": _value(evidence.decomposition, "total"),
        "total_lo90": total_lo,
        "total_hi90": total_hi,
        "other_supported": other_supported,
    }
    effect_moved = _excludes_zero(effect_lo, effect_hi)
    total_moved = _excludes_zero(total_lo, total_hi)
    status: Status
    if effect_moved or total_moved:
        status, rule = "rejected", "H5.change_established"
    elif total_moved is None:
        status, rule = "inconclusive", "H5.unknown"
    elif other_supported:
        status, rule = "inconclusive", "H5.other_explanation"
    else:
        status, rule = "supported", "H5.no_change"
    return HypothesisResult(code="H5", status=status, rule=rule, evidence=values)


def decide(evidence: CoreEvidence, config: CausalConfig) -> list[HypothesisResult]:
    """H1 to H5, in order."""
    h1 = _h1(evidence, config)
    h2 = _h2(evidence)
    h3 = _h3(evidence)
    h4 = _h4(evidence, h2)
    return [h1, h2, h3, h4, _h5(evidence, [h1, h2, h3, h4])]
