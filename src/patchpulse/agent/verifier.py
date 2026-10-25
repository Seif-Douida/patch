"""The claim verifier (design D7, spec §10.5): fixed rules, run in order on every claim before
anyone sees it. The first rule a claim fails withholds it, except V4, which rewords a causal claim
it can't allow when there's an estimate to state.

- V1 evidence exists: the cited call is in this run's trace (a claim citing only reviews needs
  no call, but one with a value does).
- V2 the value matches: the cited call re-executed on the same bundle (the model's network tools
  are read from the trace) gives a number at the JSON path within the claim's tolerance.
- V3 quotes hold: every cited review was returned by search_reviews in this run, and every quote
  is part of its text once whitespace is normalised.
- V4 the causal gate: a claim typed causal, or using causal words, needs an estimate_patch_effect
  row the gate allows (its caveat is appended); otherwise it's reworded by a fixed hedge when the
  row has an estimate, and withheld when it hasn't.
- V5 type consistency: a descriptive or comparative claim using causal words is withheld.
- V7 no review text: a claim whose text or unit repeats LEAK_WORDS words in a row from a review
  retrieved in this run is withheld (casefolded, punctuation and spacing ignored).
V6 checks the summary, written after verification, in `verify_summary`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from patchpulse.agent.gate import (
    CONFOUNDED,
    NOT_VALIDATED,
    PLACEBO_FAILED,
    RANGE_INCLUDES_ZERO,
    VOLUME_SPIKE,
    Gate,
    causal_gate,
)
from patchpulse.agent.report import Claim, Finding, WithheldClaim
from patchpulse.agent.rules import CoreEvidence, HypothesisResult
from patchpulse.agent.tools import ToolCall, Tools, Trace
from patchpulse.causal.config import CausalConfig
from patchpulse.causal.its import POSITIVE_SHARE, VOLUME
from patchpulse.causal.store import StoredEstimate

LEAK_WORDS = 8
SUMMARY_MAX_WORDS = 120  # spec §10.4
NETWORK_TOOLS = frozenset({"search_reviews", "get_patch_notes"})  # V2 reads their recorded result

# Spec §10.5's causal words and their common forms.
_CAUSAL_PHRASES = (
    r"caus(?:e|es|ed|ing)",
    r"(?:lead|leads|led|leading) to",
    r"due to",
    r"because",
    r"as a result of",
    r"result(?:s|ed|ing)? in",
    r"drove",
    r"driven by",
    r"triggered",
    r"owing to",
    r"thanks to",
    r"responsible for",
    r"attribut(?:able|ed) to",
    r"brought about",
    r"impacted",
)
CAUSAL_LEXICON = re.compile(
    r"\b(?:" + "|".join(p.replace(" ", r"\s+") for p in _CAUSAL_PHRASES) + r")\b", re.IGNORECASE
)

HYPOTHESIS_TITLES = {
    "H1": "The patch",
    "H2": "A price change or sale",
    "H3": "Off-topic reviews or a review bomb",
    "H4": "A change in who reviewed",
    "H5": "Noise or seasonality",
}
# The site's aspect labels (web/lib/aspects.ts), in lower case.
ASPECT_WORDS = {
    "performance": "performance",
    "stability_bugs": "stability and bugs",
    "gameplay_balance": "gameplay and balance",
    "content": "content",
    "monetization": "monetization",
    "online_servers": "online and servers",
    "story_world": "story and world",
    "ux_controls": "UX and controls",
    "platform_policy": "platform and policy",
    "off_topic": "off-topic matters",
}
METHOD_WORDS = {"its": "interrupted time series", "synth": "synthetic control"}
# Why a hedged claim stops short of crediting the patch, by the gate's blocker.
HEDGE_REASONS = {
    RANGE_INCLUDES_ZERO: "its 90% range includes no change at all",
    PLACEBO_FAILED: "the method's placebo check didn't pass",
    NOT_VALIDATED: "the method's benchmark doesn't cover this estimate",
    CONFOUNDED: "a sale or discount ran within 3 days of the patch",
    VOLUME_SPIKE: "reviews came in a burst that the method can't tell apart from the patch",
}

_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMBER = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_EXEMPT_AFTER = re.compile(
    r"\s*-?\s*(?:days?\b|weeks?\b|% (?:range|interval)\b|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Verified:
    claims: list[Claim]
    withheld: list[WithheldClaim]
    pass_rates: dict[str, float]  # per rule: claims that passed / claims it applied to


# Helpers shared by the rules.


def _segments(path: str) -> list[str]:
    parts: list[str] = []
    for piece in path.strip().lstrip("$").strip(".").split("."):
        name, *indexes = piece.split("[")
        if name:
            parts.append(name)
        parts += [i.rstrip("]") for i in indexes]
    return parts


def extract(result: Any, path: str) -> Any:
    """The value at a dotted path (`aspects.performance.negative.change`, `points[2].share` or
    `points.2.share`); KeyError, IndexError or ValueError when it isn't there."""
    node = result
    for part in _segments(path):
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict):
            node = node[part]
        else:
            raise KeyError(path)
    return node


def _spaces(text: str) -> str:
    return " ".join(text.split())


def _words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.casefold()).split()


def _runs(text: str) -> set[tuple[str, ...]]:
    words = _words(text)
    return {tuple(words[i : i + LEAK_WORDS]) for i in range(len(words) - LEAK_WORDS + 1)}


def retrieved_runs(texts: Iterable[str]) -> set[tuple[str, ...]]:
    return set().union(*(_runs(t) for t in texts))


def leaks(text: str, runs: set[tuple[str, ...]]) -> bool:
    return not _runs(text).isdisjoint(runs)


def _outcome_words(row: StoredEstimate) -> str:
    if row.outcome == POSITIVE_SHARE:
        words = "the positive share"
    elif row.outcome == VOLUME:
        words = "review volume"
    elif row.outcome.startswith("aspect:"):
        code = row.outcome.removeprefix("aspect:")
        words = f"the share of negative reviews mentioning {ASPECT_WORDS.get(code, code)}"
    else:
        words = row.outcome
    if row.sample == "established":
        words += " among players with 10 or more hours"
    return words


def _amount(row: StoredEstimate, value: float) -> str:
    return f"{value:+.1f}%" if row.outcome == VOLUME else f"{value:+.1f} points"


def hedge(row: StoredEstimate, gate: Gate) -> str:
    """V4's fixed rewording of a causal claim the gate blocks, stating the estimate without
    crediting the patch."""
    if row.effect is None or row.lo90 is None or row.hi90 is None:
        raise ValueError("a hedge states an estimate: this row has none")
    outcome = _outcome_words(row)
    reason = HEDGE_REASONS.get(gate.blocked_by or "", "the causal checks didn't pass")
    return (
        f"Over the {row.horizon} days from the patch, {outcome} was {_amount(row, row.effect)} "
        f"against the {METHOD_WORDS.get(row.method, row.method)} estimate of what it would have "
        f"been without the patch (90% range {row.lo90:+.1f} to {row.hi90:+.1f}). That isn't "
        f"enough to credit the patch: {reason}."
    )


# The rules.


class _Tally:
    def __init__(self) -> None:
        self.applied: dict[str, int] = {}
        self.passed: dict[str, int] = {}

    def check(self, rule: str, ok: bool) -> bool:
        self.applied[rule] = self.applied.get(rule, 0) + 1
        self.passed[rule] = self.passed.get(rule, 0) + int(ok)
        return ok

    def rates(self) -> dict[str, float]:
        return {rule: self.passed[rule] / n for rule, n in sorted(self.applied.items())}


def _value_matches(claim: Claim, call: ToolCall | None, tools: Tools) -> bool:
    if claim.value is None or claim.evidence is None or call is None:
        return False
    result = call.result if call.tool in NETWORK_TOOLS else tools.rerun(call)
    try:
        actual = extract(result, claim.evidence.json_path)
    except (KeyError, IndexError, ValueError):
        return False
    if isinstance(actual, bool) or not isinstance(actual, int | float):
        return False
    return claim.tolerance.accepts(claim.value, float(actual))


def _quotes_hold(claim: Claim, retrieved: Mapping[int, str]) -> bool:
    if any(rid not in retrieved for rid in claim.review_ids):
        return False
    for quote in claim.quote_spans:
        text = retrieved.get(quote.review_id)
        span = _spaces(quote.text)
        if text is None or not span or span not in _spaces(text):
            return False
    return True


def _cited_row(claim: Claim, call: ToolCall | None, tools: Tools) -> StoredEstimate | None:
    """The estimate row a claim cites: the horizon its path names, or the only one served."""
    if call is None or claim.evidence is None or call.tool != "estimate_patch_effect":
        return None
    rows = tools.estimate_rows(call)
    parts = _segments(claim.evidence.json_path)
    if len(parts) >= 2 and parts[0] == "horizons":
        return next((r for r in rows if str(r.horizon) == parts[1]), None)
    return rows[0] if len(rows) == 1 else None


def _is_causal(claim: Claim) -> bool:
    return claim.claim_type == "causal" or CAUSAL_LEXICON.search(claim.text) is not None


def _verify(
    claim: Claim,
    trace: Trace,
    tools: Tools,
    config: CausalConfig,
    runs: set[tuple[str, ...]],
    tally: _Tally,
) -> tuple[Claim | None, str | None]:
    call = None if claim.evidence is None else trace.find(claim.evidence.call_id)
    v1 = call is not None if claim.evidence is not None else (
        claim.value is None and bool(claim.supporting_reviews)
    )  # fmt: skip
    if not tally.check("V1", v1):
        return None, "V1"
    if claim.value is not None and not tally.check("V2", _value_matches(claim, call, tools)):
        return None, "V2"
    if claim.supporting_reviews and not tally.check("V3", _quotes_hold(claim, trace.retrieved)):
        return None, "V3"
    if _is_causal(claim):
        row = _cited_row(claim, call, tools)
        gate = None if row is None else causal_gate(row, config)
        tally.check("V4", gate is not None and gate.allowed)
        if row is None or gate is None:
            return None, "V4"
        if gate.allowed:
            if gate.caveat:
                claim = claim.model_copy(update={"text": f"{claim.text} {gate.caveat}"})
        elif row.effect is None or row.lo90 is None or row.hi90 is None:
            return None, "V4"
        else:
            claim = claim.model_copy(
                update={"text": hedge(row, gate), "claim_type": "comparative", "reworded": True}
            )
    if claim.claim_type != "causal" and not tally.check(
        "V5", CAUSAL_LEXICON.search(claim.text) is None
    ):
        return None, "V5"
    leaked = leaks(claim.text, runs) or (claim.unit is not None and leaks(claim.unit, runs))
    if not tally.check("V7", not leaked):
        return None, "V7"
    return claim, None


def verify_claims(
    claims: Sequence[Claim], trace: Trace, tools: Tools, config: CausalConfig
) -> Verified:
    """Every claim through V1-V5 and V7, in order; the stored claim as drafted when withheld."""
    runs = retrieved_runs(trace.retrieved.values())
    tally = _Tally()
    kept: list[Claim] = []
    withheld: list[WithheldClaim] = []
    for claim in claims:
        verified, rule = _verify(claim, trace, tools, config, runs, tally)
        if verified is not None:
            kept.append(verified)
        else:
            withheld.append(WithheldClaim(claim=claim, rule=rule or "V1"))
    return Verified(claims=kept, withheld=withheld, pass_rates=tally.rates())


def safe_findings(
    findings: Sequence[Finding], verified: Sequence[Claim], retrieved: Mapping[int, str]
) -> list[Finding]:
    """Findings whose title repeats review text, or uses causal words without a verified causal
    claim under it, get their explanation's plain title instead."""
    runs = retrieved_runs(retrieved.values())
    causal = {c.claim_id for c in verified if c.claim_type == "causal" and not c.reworded}
    safe = []
    for finding in findings:
        overclaims = CAUSAL_LEXICON.search(finding.title) and not causal & set(finding.claim_ids)
        if leaks(finding.title, runs) or overclaims:
            finding = finding.model_copy(update={"title": HYPOTHESIS_TITLES[finding.hypothesis]})
        safe.append(finding)
    return safe


def _numbers(summary: str) -> list[float]:
    """The numbers a summary states as values: not dates, years, durations or the 90% range."""
    text = _ISO_DATE.sub(" ", summary)
    values = []
    for match in _NUMBER.finditer(text):
        token = match.group()
        if _EXEMPT_AFTER.match(text, match.end()):
            continue
        value = float(token.replace(",", ""))
        if "." not in token and 1990 <= value <= 2100:  # a year
            continue
        values.append(value)
    return values


def verify_summary(
    summary: str, verified: Sequence[Claim], retrieved: Mapping[int, str]
) -> str | None:
    """V6 and V7 on the summary: at most 120 words; every number in it is a verified claim's
    value within that claim's tolerance (sign stated in words); causal words only with a verified
    causal claim; no run of review text. None means "use the template"."""
    if len(summary.split()) > SUMMARY_MAX_WORDS:
        return None
    if leaks(summary, retrieved_runs(retrieved.values())):
        return None
    if CAUSAL_LEXICON.search(summary) and not any(
        c.claim_type == "causal" and not c.reworded for c in verified
    ):
        return None
    valued = [c for c in verified if c.value is not None]
    for number in _numbers(summary):
        if not any(c.tolerance.accepts(number, abs(c.value or 0.0)) for c in valued):
            return None
    return summary


def template_summary(verified: Sequence[Claim], hypotheses: Sequence[HypothesisResult]) -> str:
    """The fixed summary used when the model's fails V6 or V7, or there's no model."""
    groups: dict[str, list[str]] = {"supported": [], "rejected": [], "inconclusive": []}
    for h in hypotheses:
        groups[h.status].append(HYPOTHESIS_TITLES[h.code].lower())
    sentences = [
        f"{label}: {', '.join(groups[status])}."
        for status, label in (
            ("supported", "Supported"),
            ("rejected", "Ruled out"),
            ("inconclusive", "Inconclusive"),
        )
        if groups[status]
    ]
    sentences.append(
        "The claims below passed the verifier." if verified else "No claim passed the verifier."
    )
    return " ".join(sentences)


def basic_claims(evidence: CoreEvidence) -> list[Claim]:
    """The basic report's fixed sentences (design D8), each citing its core tool call."""
    ids = evidence.call_ids
    drafts: list[dict[str, Any]] = []
    window, baseline = evidence.window, evidence.baseline
    if "window" in ids and window.get("share") is not None and baseline.get("share") is not None:
        drafts.append(
            {
                "text": f"The positive share went from {baseline['share']:.1f}% in the baseline "
                f"to {window['share']:.1f}% in the window.",
                "claim_type": "comparative",
                "value": window["share"],
                "unit": "%",
                "evidence": {"call_id": ids["window"], "json_path": "total.share"},
            }
        )
    if "window" in ids and window.get("new_players_share") is not None:
        drafts.append(
            {
                "text": f"Players with under 2 hours wrote {window['new_players_share']:.1f}% of "
                "the window's reviews.",
                "claim_type": "descriptive",
                "value": window["new_players_share"],
                "unit": "%",
                "evidence": {"call_id": ids["window"], "json_path": "total.new_players_share"},
            }
        )
    row = evidence.estimate
    if (
        "estimate" in ids
        and row is not None
        and row.effect is not None
        and row.lo90 is not None
        and row.hi90 is not None
    ):
        drafts.append(
            {
                "text": f"Over the {row.horizon} days from the patch, the positive share was "
                f"{row.effect:+.1f} points against the {METHOD_WORDS.get(row.method, row.method)} "
                f"estimate of what it would have been without the patch (90% range "
                f"{row.lo90:+.1f} to {row.hi90:+.1f}).",
                "claim_type": "comparative",
                "value": round(row.effect, 2),
                "unit": "pp",
                "evidence": {
                    "call_id": ids["estimate"],
                    "json_path": f"horizons.{row.horizon}.effect",
                },
            }
        )
    split = evidence.decomposition
    if "decomposition" in ids and "error" not in split:
        drafts.append(
            {
                "text": f"Of the {split['total']:+.1f}-point change in the positive share, "
                f"{split['composition']:+.1f} points came with a change in who reviewed "
                "(their playtime at review).",
                "claim_type": "comparative",
                "value": split["composition"],
                "unit": "pp",
                "evidence": {"call_id": ids["decomposition"], "json_path": "composition"},
            }
        )
    moved = []  # aspects whose share among negative reviews changed, its interval excluding 0
    for code, sides in evidence.aspects.get("aspects", {}).items():
        side = sides["negative"]
        if side["change"] is not None and not side["lo90"] <= 0 <= side["hi90"]:
            moved.append((abs(side["change"]), code, side))
    if "aspects" in ids and moved:
        _, code, side = max(moved, key=lambda m: (m[0], m[1]))
        drafts.append(
            {
                "text": f"Among negative reviews, mentions of {ASPECT_WORDS.get(code, code)} went "
                f"from {side['baseline_share']:.1f}% to {side['window_share']:.1f}% "
                f"({side['change']:+.1f} points).",
                "claim_type": "comparative",
                "value": side["change"],
                "unit": "pp",
                "evidence": {
                    "call_id": ids["aspects"],
                    "json_path": f"aspects.{code}.negative.change",
                },
            }
        )
    return [Claim.model_validate({**d, "claim_id": f"b{i}"}) for i, d in enumerate(drafts, 1)]
