"""The report schema and its two copies (plan Task 6, design D6, spec §10.4): the stored report
keeps quote spans, review ids and the withheld claims' text; the public copy keeps none of them."""

from __future__ import annotations

import json
import string
from datetime import date

import numpy as np
import pytest

from patchpulse.agent.report import (
    DRAFT_SCHEMA,
    Claim,
    DateWindow,
    Draft,
    EvidenceRef,
    Finding,
    QuoteSpan,
    Report,
    Tolerance,
    WithheldClaim,
    confidence_for,
    public_copy,
)
from patchpulse.agent.rules import HypothesisResult


def claim(
    claim_id: str,
    claim_type: str = "descriptive",
    *,
    review_ids: tuple[int, ...] = (),
    quotes: tuple[tuple[int, str], ...] = (),
    text: str = "Negative reviews mentioning performance rose from 12% to 31%.",
    unit: str | None = "pp",
    reworded: bool = False,
) -> Claim:
    return Claim.model_validate(
        {
            "claim_id": claim_id,
            "text": text,
            "claim_type": claim_type,
            "value": 19.0,
            "unit": unit,
            "evidence": {"call_id": "r1.4", "json_path": "aspects.performance.negative.change"},
            "review_ids": list(review_ids),
            "quote_spans": [{"review_id": r, "text": t} for r, t in quotes],
            "reworded": reworded,
        }
    )


HYPOTHESES = [
    HypothesisResult(code=code, status="inconclusive", rule=f"{code}.x", evidence={"v": 1.0})
    for code in ("H1", "H2", "H3", "H4", "H5")
]


def report(
    claims: list[Claim], withheld: list[WithheldClaim] | None = None, **values: object
) -> Report:
    fields: dict[str, object] = {
        "report_id": "p1-7d-52",
        "appid": 553850,
        "scope": "bundle",
        "bundle_id": "p1",
        "horizon": 7,
        "question": None,
        "window": {"start": "2026-09-10", "end": "2026-09-16"},
        "data_version": 52,
        "model_ids": {"draft": "gemini-3.5-flash-lite"},
        "prompt_version": "abc123",
        "summary": "Performance complaints rose after the patch.",
        "findings": [{"title": "Performance", "hypothesis": "H1", "claim_ids": ["c1"]}],
        "hypotheses": [h.model_dump() for h in HYPOTHESES],
        "claims": [c.model_dump() for c in claims],
        "withheld_claims": [w.model_dump() for w in withheld or []],
        "confidence": "medium",
        "basic": False,
        "rule_pass_rates": {"V1": 1.0, "V2": 0.75},
    }
    fields.update(values)
    return Report.model_validate(fields)


def test_tolerance_follows_the_unit() -> None:
    share = claim("c1", unit="pp")
    count = claim("c2", unit="reviews")

    assert share.tolerance == Tolerance(absolute=0.5, relative=None)
    assert count.tolerance == Tolerance(absolute=None, relative=0.02)
    assert share.tolerance.accepts(19.5, 19.0)
    assert not share.tolerance.accepts(19.6, 19.0)
    assert count.tolerance.accepts(102, 100)
    assert not count.tolerance.accepts(103, 100)
    assert claim("c3", unit="%").tolerance == Tolerance(absolute=0.5, relative=None)


def words(rng: np.random.Generator, n: int) -> str:
    letters = list(string.ascii_lowercase)
    return " ".join("".join(rng.choice(letters, size=int(rng.integers(3, 10)))) for _ in range(n))


@pytest.mark.parametrize("seed", range(20))
def test_public_copy_drops_quotes_ids_and_withheld_text(seed: int) -> None:
    rng = np.random.default_rng(seed)
    ids = sorted({int(i) for i in rng.integers(1_000_000_000, 10_000_000_000, size=12)})
    assert len(ids) == 12
    spans = [(rid, words(rng, int(rng.integers(2, 13)))) for rid in ids[:6]]
    verified = [
        claim(f"c{i}", str(rng.choice(["descriptive", "comparative", "causal"])),
              review_ids=tuple(ids[6 + i : 8 + i]), quotes=(spans[i],))
        for i in range(4)
    ]  # fmt: skip
    hidden = words(rng, 15)
    withheld = [
        WithheldClaim(claim=claim("w1", text=hidden, quotes=(spans[4], spans[5])), rule="V3"),
        WithheldClaim(claim=claim("w2", text=hidden + " again"), rule="V7"),
        WithheldClaim(claim=claim("w3", text="caused it"), rule="V3"),
    ]
    stored = report(verified, withheld)

    copy = json.dumps(public_copy(stored))

    for rid, text in spans:
        assert text not in copy
        assert str(rid) not in copy
    for rid in ids:
        assert str(rid) not in copy
    assert hidden not in copy
    assert "caused it" not in copy
    shown = public_copy(stored)
    assert shown["withheld"] == {"V3": 2, "V7": 1}
    assert [c["supported_by"] for c in shown["claims"]] == [3, 3, 3, 3]  # 2 cited + 1 quoted
    assert "quote_spans" in stored.claims[0].model_dump()  # the stored copy keeps them


def test_public_copy_keeps_reworded_and_drops_findings_without_verified_claims() -> None:
    stored = report(
        [claim("c1", "comparative", reworded=True)],
        [WithheldClaim(claim=claim("c2", "causal"), rule="V4")],
        findings=[
            {"title": "The patch", "hypothesis": "H1", "claim_ids": ["c1", "c2"]},
            {"title": "Only withheld", "hypothesis": "H3", "claim_ids": ["c2"]},
        ],
    )

    shown = public_copy(stored)

    assert shown["claims"][0]["reworded"] is True
    assert shown["findings"] == [{"title": "The patch", "hypothesis": "H1", "claim_ids": ["c1"]}]
    assert shown["withheld"] == {"V4": 1}
    assert [h["code"] for h in shown["hypotheses"]] == ["H1", "H2", "H3", "H4", "H5"]
    assert shown["window"] == {"start": "2026-09-10", "end": "2026-09-16"}


def test_confidence_rule() -> None:
    causal = claim("c1", "causal", review_ids=(1, 2))
    quoted = claim("c2", review_ids=(3,), quotes=((2, "runs badly"),))

    assert confidence_for(report([causal, quoted])) == "high"  # reviews 1, 2 and 3
    assert confidence_for(report([causal])) == "low"  # a causal claim, 2 reviews
    assert confidence_for(report([quoted, claim("c3", "comparative")])) == "medium"
    reworded = claim("c1", "causal", review_ids=(1, 2, 3), reworded=True)
    assert confidence_for(report([reworded])) == "medium"  # a hedged claim isn't causal
    assert confidence_for(report([])) == "low"


def test_report_round_trips() -> None:
    stored = report(
        [claim("c1", review_ids=(11,), quotes=((11, "frame drops"),))],
        [WithheldClaim(claim=claim("c2", "causal"), rule="V4")],
        question="Why did reviews drop after the patch?",
        scope="question",
    )

    again = Report.model_validate_json(stored.model_dump_json())

    assert again == stored
    assert again.window == DateWindow(start=date(2026, 9, 10), end=date(2026, 9, 16))
    assert again.claims[0].evidence == EvidenceRef(
        call_id="r1.4", json_path="aspects.performance.negative.change"
    )
    assert again.claims[0].quote_spans == [QuoteSpan(review_id=11, text="frame drops")]


def test_the_draft_schema_matches_the_draft_model() -> None:
    claim_schema = DRAFT_SCHEMA["properties"]["claims"]["items"]
    finding_schema = DRAFT_SCHEMA["properties"]["findings"]["items"]
    written = {"claim_id", "text", "claim_type", "value", "unit", "evidence", "review_ids",
               "quote_spans"}  # fmt: skip

    assert set(claim_schema["properties"]) == written
    assert written < set(Claim.model_fields)  # tolerance and reworded aren't the model's
    assert set(finding_schema["properties"]) == set(Finding.model_fields)
    assert "summary" not in DRAFT_SCHEMA["properties"]
    draft = Draft.model_validate(
        {
            "findings": [{"title": "Performance", "hypothesis": "H1", "claim_ids": ["c1"]}],
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "Performance complaints rose.",
                    "claim_type": "descriptive",
                }
            ],
        }
    )
    assert draft.claims[0].tolerance == Tolerance(absolute=None, relative=0.02)
