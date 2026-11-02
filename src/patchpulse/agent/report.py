"""The report (design D6, spec §10.4) and its two copies.

The stored report (app.report) keeps everything: quote spans, review ids and the withheld claims
with their text. The public copy (the site, /reports and /ask) keeps none of those: a claim shows
how many reviews support it, withheld claims appear as counts by the rule that withheld them, and
findings keep only their verified claims. The model drafts findings and claims (DRAFT_SCHEMA) and
never the summary, the confidence or the hypotheses' statuses.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from patchpulse.agent.rules import HypothesisResult

ClaimType = Literal["descriptive", "comparative", "causal"]
Hypothesis = Literal["H1", "H2", "H3", "H4", "H5"]
Confidence = Literal["low", "medium", "high"]
SHARE_UNITS = frozenset({"pp", "%"})  # spec §10.4: absolute 0.5 points for shares
SHARE_TOLERANCE = 0.5
RELATIVE_TOLERANCE = 0.02  # spec §10.4: 2% otherwise
HIGH_CONFIDENCE_REVIEWS = 3  # spec §10.5
_SLACK = 1e-9  # a value exactly at the tolerance passes despite float rounding


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EvidenceRef(_Model):
    call_id: str
    json_path: str


class Tolerance(_Model):
    absolute: float | None = None
    relative: float | None = None

    @classmethod
    def for_unit(cls, unit: str | None) -> Tolerance:
        if unit in SHARE_UNITS:
            return cls(absolute=SHARE_TOLERANCE)
        return cls(relative=RELATIVE_TOLERANCE)

    def accepts(self, claimed: float, actual: float) -> bool:
        if self.absolute is not None:
            return abs(claimed - actual) <= self.absolute + _SLACK
        return abs(claimed - actual) <= (self.relative or 0.0) * abs(actual) + _SLACK


class QuoteSpan(_Model):
    review_id: int
    text: str


class Claim(_Model):
    claim_id: str
    text: str
    claim_type: ClaimType
    value: float | None = None
    unit: str | None = None
    evidence: EvidenceRef | None = None
    tolerance: Tolerance
    review_ids: list[int] = []
    quote_spans: list[QuoteSpan] = []
    reworded: bool = False  # rewritten by V4's hedge template

    @model_validator(mode="before")
    @classmethod
    def _default_tolerance(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("tolerance") is None:
            return {**data, "tolerance": Tolerance.for_unit(data.get("unit"))}
        return data

    @property
    def supporting_reviews(self) -> frozenset[int]:
        return frozenset(self.review_ids) | {q.review_id for q in self.quote_spans}


class Finding(_Model):
    title: str
    hypothesis: Hypothesis
    claim_ids: list[str]


class WithheldClaim(_Model):
    claim: Claim
    rule: str  # the verifier rule that withheld it: V1 to V7


class DateWindow(_Model):
    start: date
    end: date


class Draft(_Model):
    """The model's draft: findings and claims, nothing else."""

    findings: list[Finding]
    claims: list[Claim]


def renumbered(draft: Draft) -> Draft:
    """The draft with claim ids written by the code, c1, c2, ... in order, and the findings'
    references remapped (a repeated id maps to its first claim; an unknown one is dropped). A
    claim's id is shown with it, and the model could fill one with review text."""
    ids: dict[str, str] = {}
    claims = []
    for n, claim in enumerate(draft.claims, start=1):
        ids.setdefault(claim.claim_id, f"c{n}")
        claims.append(claim.model_copy(update={"claim_id": f"c{n}"}))
    findings = [
        f.model_copy(update={"claim_ids": [ids[i] for i in f.claim_ids if i in ids]})
        for f in draft.findings
    ]
    return Draft(findings=findings, claims=claims)


class ModelUse(_Model):
    """One model call: its step, model and tokens (design D13: in the stored report, never in
    the public copy)."""

    step: str
    model_id: str
    input_tokens: int
    output_tokens: int
    cached: bool


class Report(_Model):
    report_id: str
    appid: int
    scope: Literal["bundle", "question"]
    bundle_id: str | None
    horizon: int | None
    question: str | None
    window: DateWindow
    data_version: int
    model_ids: dict[str, str]  # step -> the model that answered it
    prompt_version: str
    summary: str
    findings: list[Finding]
    hypotheses: list[HypothesisResult]
    claims: list[Claim]  # verified only
    withheld_claims: list[WithheldClaim]
    confidence: Confidence
    basic: bool  # written without a model (design D8)
    rule_pass_rates: dict[str, float]
    usage: list[ModelUse] = []  # every model call, in order (stored only)
    seconds: float | None = None  # the run's wall time (stored only)


def confidence_for(report: Report) -> Confidence:
    """Spec §10.5: high with a verified causal claim (one that passed the gate, so not reworded)
    and at least 3 supporting reviews across the verified claims; medium with any other verified
    claim, a causal one with fewer reviews included (Seif, 26 Oct: stronger evidence never reads
    lower than descriptive claims alone); low with none."""
    causal = [c for c in report.claims if c.claim_type == "causal" and not c.reworded]
    reviews = frozenset().union(*(c.supporting_reviews for c in report.claims))
    if causal and len(reviews) >= HIGH_CONFIDENCE_REVIEWS:
        return "high"
    return "medium" if report.claims else "low"


def public_copy(report: Report) -> dict[str, Any]:
    """What visitors see: no quote spans, no review ids, no withheld text."""
    verified = {c.claim_id for c in report.claims}
    findings = []
    for finding in report.findings:
        shown = [i for i in finding.claim_ids if i in verified]
        if shown:
            findings.append({**finding.model_dump(), "claim_ids": shown})
    withheld: dict[str, int] = {}
    for w in report.withheld_claims:
        withheld[w.rule] = withheld.get(w.rule, 0) + 1
    return {
        "report_id": report.report_id,
        "appid": report.appid,
        "scope": report.scope,
        "bundle_id": report.bundle_id,
        "horizon": report.horizon,
        "question": report.question,
        "window": report.window.model_dump(mode="json"),
        "data_version": report.data_version,
        "model_ids": dict(report.model_ids),
        "prompt_version": report.prompt_version,
        "summary": report.summary,
        "confidence": report.confidence,
        "basic": report.basic,
        "hypotheses": [h.model_dump(mode="json") for h in report.hypotheses],
        "findings": findings,
        "claims": [
            {
                "claim_id": c.claim_id,
                "text": c.text,
                "claim_type": c.claim_type,
                "value": c.value,
                "unit": c.unit,
                "reworded": c.reworded,
                "supported_by": len(c.supporting_reviews),
            }
            for c in report.claims
        ],
        "withheld": dict(sorted(withheld.items())),
        "rule_pass_rates": dict(report.rule_pass_rates),
    }


_STRING = {"type": "string"}
_HYPOTHESES = ["H1", "H2", "H3", "H4", "H5"]

# The draft's JSON schema, in the plain form Gemini's JSON mode takes (Phase 2's style): optional
# fields are left out of `required` rather than nullable.
DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": _STRING,
                    "hypothesis": {"type": "string", "enum": _HYPOTHESES},
                    "claim_ids": {"type": "array", "items": _STRING},
                },
                "required": ["title", "hypothesis", "claim_ids"],
            },
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": _STRING,
                    "text": _STRING,
                    "claim_type": {
                        "type": "string",
                        "enum": ["descriptive", "comparative", "causal"],
                    },
                    "value": {"type": "number"},
                    "unit": _STRING,
                    "evidence": {
                        "type": "object",
                        "properties": {"call_id": _STRING, "json_path": _STRING},
                        "required": ["call_id", "json_path"],
                    },
                    "review_ids": {"type": "array", "items": {"type": "integer"}},
                    "quote_spans": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"review_id": {"type": "integer"}, "text": _STRING},
                            "required": ["review_id", "text"],
                        },
                    },
                },
                "required": ["claim_id", "text", "claim_type"],
            },
        },
    },
    "required": ["findings", "claims"],
}
