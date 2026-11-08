"""Hand-made reports, records and incidents for the eval tests (Phase 6 plan Tasks 3 and 5)."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from patchpulse.agent.graph import Refusal, RunRecord
from patchpulse.agent.report import (
    Claim,
    DateWindow,
    Draft,
    Report,
    Tolerance,
    WithheldClaim,
)
from patchpulse.agent.rules import HypothesisResult
from patchpulse.agent.tools import ToolCall, Trace
from patchpulse.evals.incidents import Incident

DAY = date(2025, 9, 2)
WINDOW = (DAY, DAY + timedelta(days=6))
CODES = ("H1", "H2", "H3", "H4", "H5")


def hypotheses(supported: tuple[str, ...] = (), *, h5: str | None = None,
               total_change: float | None = -3.1) -> list[HypothesisResult]:  # fmt: skip
    """H1-H5: the `supported` ones supported, H5 as given (else inconclusive), the rest
    inconclusive; H5 carries the before/after change."""
    results = []
    for code in CODES:
        status = "supported" if code in supported else "inconclusive"
        if code == "H5" and h5 is not None:
            status = h5
        evidence: dict[str, Any] = {"total_change": total_change} if code == "H5" else {}
        results.append(
            HypothesisResult.model_validate(
                {"code": code, "status": status, "rule": f"{code}.x", "evidence": evidence}
            )
        )
    return results


def claim(
    claim_id: str, text: str, *, claim_type: str = "descriptive", reworded: bool = False
) -> Claim:
    return Claim(
        claim_id=claim_id,
        text=text,
        claim_type=claim_type,  # type: ignore[arg-type]
        tolerance=Tolerance(relative=0.02),
        reworded=reworded,
    )


def report(hyps: list[HypothesisResult], *, claims: tuple[Claim, ...] = (),
           withheld: tuple[WithheldClaim, ...] = (), basic: bool = False) -> Report:  # fmt: skip
    return Report(
        report_id="q-1",
        appid=553850,
        scope="question",
        bundle_id=None,
        horizon=None,
        question="Why?",
        window=DateWindow(start=WINDOW[0], end=WINDOW[1]),
        data_version=60,
        model_ids={},
        prompt_version="p1",
        summary="",
        findings=[],
        hypotheses=hyps,
        claims=list(claims),
        withheld_claims=list(withheld),
        confidence="low",
        basic=basic,
        rule_pass_rates={},
    )


def aspects_call(policy: float | None, off_topic: float | None) -> ToolCall:
    result = {
        "aspects": {
            "platform_policy": {"negative": {"change": policy}},
            "off_topic": {"negative": {"change": off_topic}},
        }
    }
    return ToolCall("r.1", "aspect_breakdown", {}, result)


def record(
    result: Report | Refusal,
    *,
    calls: tuple[ToolCall, ...] = (),
    draft: Draft | None = None,
    window: tuple[date, date] | None = WINDOW,
) -> RunRecord:
    hyps = result.hypotheses if isinstance(result, Report) else []
    basic = isinstance(result, Report) and result.basic
    return RunRecord(result, window, hyps, draft, None, Trace(calls=list(calls)), basic)


def incident(
    primary: str = "patch",
    *,
    secondary: tuple[str, ...] = (),
    direction: str = "negative",
    window: tuple[date, date] = WINDOW,
    incident_id: str = "hd2-test",
) -> Incident:
    return Incident.model_validate({
        "id": incident_id, "game": "Helldivers 2", "appid": 553850,
        "window": {"start": window[0], "end": window[1]},
        "question": "Why?", "primary_cause": primary, "secondary_causes": list(secondary),
        "direction": direction, "key_facts": ["a fact"], "acceptable_evidence": ["evidence"],
        "sources": [{"url": "https://example.org/a", "published": DAY, "shows": "a page"}],
        "confidence": "high", "researched": date(2026, 11, 4),
    })  # fmt: skip
