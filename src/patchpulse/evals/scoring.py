"""Scoring one run against one incident (Phase 6 design D3-D4).

A report states five explanations' statuses, set by rules, not a cause category. The category it
names is read by a fixed rule, frozen before the first graded run (SCORING_VERSION):

1. A refusal is `refused`.
2. Otherwise the first supported explanation in the order H1, H3, H2, H4 names it: H1 `patch`;
   H3 `off_topic`, or `platform_policy` when the platform-policy aspect rose more than the
   off-topic aspect among negative reviews (the run's own aspect breakdown); H2 `pricing_sale`;
   H4 `audience_shift`, which no incident has. H1 can't be supported together with H2 or H3 (the
   gate blocks it near a sale or under a review bomb), so the order only settles H2, H3 and H4
   ties, and a review bomb is the more specific explanation than a sale's mix.
3. With none supported, H5 supported is `no_change`; anything else is `undetermined`.

`launch_performance` is never predicted: no explanation names a launch (design, "What the code
already says").
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

from patchpulse.agent.graph import Refusal, RunRecord
from patchpulse.agent.report import Claim, Report
from patchpulse.agent.rules import HypothesisResult
from patchpulse.agent.tools import Trace
from patchpulse.agent.verifier import CAUSAL_LEXICON
from patchpulse.evals.incidents import Category, Incident

SCORING_VERSION = "s1"

Predicted = Category | Literal["audience_shift", "undetermined", "refused"]
_ORDER: tuple[tuple[str, Predicted], ...] = (
    ("H1", "patch"),
    ("H3", "off_topic"),
    ("H2", "pricing_sale"),
    ("H4", "audience_shift"),
)
MODEL_TOOLS = frozenset({"search_reviews", "get_patch_notes"})
NOT_RECORDED = "wasn't recorded"  # a replay miss, not a call the model got wrong
PATCH_WORDS = re.compile(r"\b(?:patch|update|hotfix)", re.IGNORECASE)


def _status(hypotheses: Sequence[HypothesisResult], code: str) -> str | None:
    return next((h.status for h in hypotheses if h.code == code), None)


def _negative_change(trace: Trace, aspect: str) -> float | None:
    """An aspect's change among negative reviews, from the run's first aspect breakdown."""
    call = next((c for c in trace.calls if c.tool == "aspect_breakdown"), None)
    try:
        change = call.result["aspects"][aspect]["negative"]["change"] if call else None
    except (KeyError, TypeError):
        return None
    return None if change is None else float(change)


def predicted_category(result: Report | Refusal, trace: Trace) -> Predicted:
    if isinstance(result, Refusal):
        return "refused"
    for code, category in _ORDER:
        if _status(result.hypotheses, code) != "supported":
            continue
        if code == "H3":
            policy = _negative_change(trace, "platform_policy")
            off_topic = _negative_change(trace, "off_topic")
            if policy is not None and (off_topic is None or policy > off_topic):
                return "platform_policy"
        return category
    return "no_change" if _status(result.hypotheses, "H5") == "supported" else "undetermined"


def window_iou(a: tuple[date, date], b: tuple[date, date]) -> float:
    """Intersection over union of two windows' days, both ends included."""
    overlap = (min(a[1], b[1]) - max(a[0], b[0])).days + 1
    if overlap <= 0:
        return 0.0
    union = (max(a[1], b[1]) - min(a[0], b[0])).days + 1
    return overlap / union


@dataclass(frozen=True)
class CaseScore:
    incident_id: str
    expected: Category
    predicted: Predicted
    correct: bool
    lenient: bool  # the primary or a secondary cause
    control: bool
    cause_alarm: bool  # a control with any of H1-H4 supported
    change_alarm: bool  # a control with H5 ruled out: a change established
    window_iou: float  # 0 for a refusal
    direction_ok: bool | None  # None for a control
    decided: int  # explanations not inconclusive, 0-5


def _direction_ok(incident: Incident, hypotheses: Sequence[HypothesisResult]) -> bool | None:
    if incident.control or incident.direction == "none":
        return None
    h5 = next((h for h in hypotheses if h.code == "H5"), None)
    change = None if h5 is None else h5.evidence.get("total_change")
    if not isinstance(change, int | float) or isinstance(change, bool):
        return False
    return change < 0 if incident.direction == "negative" else change > 0


def score_case(incident: Incident, record: RunRecord) -> CaseScore:
    predicted = predicted_category(record.result, record.trace)
    hypotheses = record.result.hypotheses if isinstance(record.result, Report) else []
    supported = {h.code for h in hypotheses if h.status == "supported"}
    window = (incident.window.start, incident.window.end)
    return CaseScore(
        incident_id=incident.id,
        expected=incident.primary_cause,
        predicted=predicted,
        correct=predicted == incident.primary_cause,
        lenient=predicted == incident.primary_cause or predicted in incident.secondary_causes,
        control=incident.control,
        cause_alarm=incident.control and bool(supported & {"H1", "H2", "H3", "H4"}),
        change_alarm=incident.control and _status(hypotheses, "H5") == "rejected",
        window_iou=0.0 if record.window is None else window_iou(record.window, window),
        direction_ok=_direction_ok(incident, hypotheses),
        decided=sum(h.status != "inconclusive" for h in hypotheses),
    )


@dataclass(frozen=True)
class ClaimStats:
    drafted: int
    shipped: int
    reworded: int
    withheld_by_rule: dict[str, int]


def _model_report(record: RunRecord) -> Report | None:
    """The report when the model drafted it: not a refusal, not a basic report."""
    result = record.result
    if isinstance(result, Refusal) or result.basic or record.basic:
        return None
    return result


def claim_stats(record: RunRecord) -> ClaimStats | None:
    report = _model_report(record)
    if report is None:
        return None
    drafted = (
        len(record.draft.claims)
        if record.draft is not None
        else len(report.claims) + len(report.withheld_claims)
    )
    return ClaimStats(
        drafted=drafted,
        shipped=len(report.claims),
        reworded=sum(c.reworded for c in report.claims),
        withheld_by_rule=dict(sorted(Counter(w.rule for w in report.withheld_claims).items())),
    )


@dataclass(frozen=True)
class ToolCallStats:
    planned: int  # the model's calls of search_reviews and get_patch_notes
    errors: int  # results with an error, replay misses left out
    repeats: int  # the same tool with the same arguments as an earlier call


def tool_call_stats(trace: Trace) -> ToolCallStats:
    calls = [c for c in trace.calls if c.tool in MODEL_TOOLS]
    seen: list[tuple[str, dict[str, object]]] = []
    repeats = 0
    for call in calls:
        if (call.tool, call.args) in seen:
            repeats += 1
        seen.append((call.tool, call.args))
    errors = sum("error" in c.result and NOT_RECORDED not in str(c.result["error"]) for c in calls)
    return ToolCallStats(planned=len(calls), errors=errors, repeats=repeats)


def _blames_a_patch(claim: Claim) -> bool:
    causal = claim.claim_type == "causal" or CAUSAL_LEXICON.search(claim.text) is not None
    return causal and PATCH_WORDS.search(claim.text) is not None


def patch_blame(record: RunRecord) -> tuple[int, int] | None:
    """A3's measure: the drafted and the shipped claims that blame a patch (causal by V4 and V5's
    test, and naming a patch, an update or a hotfix)."""
    report = _model_report(record)
    if report is None:
        return None
    drafted = record.draft.claims if record.draft is not None else []
    return (
        sum(_blames_a_patch(c) for c in drafted),
        sum(_blames_a_patch(c) for c in report.claims),
    )
