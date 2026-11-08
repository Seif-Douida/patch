"""Metrics over the golden set (Phase 6 design D4 and D8).

Every rate comes with a 95% percentile bootstrap interval over incidents (2,000 resamples, a fixed
seed). Claim and tool-call rates pool each resample's incidents (a cluster bootstrap): resampling
claims would treat a report's claims as independent and give intervals that are too narrow.
Alarms count controls only. Ablations are compared with paired differences over the incidents
both arms ran.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from patchpulse.agent.report import Report
from patchpulse.evals.incidents import Incident
from patchpulse.evals.records import StoredCase
from patchpulse.evals.scoring import (
    CaseScore,
    claim_stats,
    patch_blame,
    score_case,
    tool_call_stats,
)
from patchpulse.models.metrics import bootstrap_ci, paired_bootstrap_ci

EVAL_SEED = 20261108
SAMPLES = 2000
WITHHELD_RULES = ("V1", "V2", "V3", "V4", "V5", "V7")  # V6 checks the summary, never a claim
_PLACES = 6


@dataclass(frozen=True)
class Estimate:
    value: float | None
    lo: float | None = None
    hi: float | None = None
    n: int = 0


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), _PLACES)


def rate(flags: Sequence[bool]) -> Estimate:
    """The share of true flags, with its interval over items."""
    if not flags:
        return Estimate(None)
    values = np.asarray(flags, dtype=float)
    lo, hi = bootstrap_ci(
        lambda ix: float(values[ix].mean()), len(values), samples=SAMPLES, seed=EVAL_SEED
    )
    return Estimate(_round(values.mean()), _round(lo), _round(hi), len(values))


def mean(values: Sequence[float]) -> Estimate:
    if not values:
        return Estimate(None)
    array = np.asarray(values, dtype=float)
    lo, hi = bootstrap_ci(
        lambda ix: float(array[ix].mean()), len(array), samples=SAMPLES, seed=EVAL_SEED
    )
    return Estimate(_round(array.mean()), _round(lo), _round(hi), len(array))


def count(true: int, n: int) -> Estimate:
    """A count of cases (no interval: the gate compares counts)."""
    return Estimate(float(true), None, None, n)


def ratio(numerators: Sequence[float], denominators: Sequence[float]) -> Estimate:
    """Pooled numerators over pooled denominators, one pair per incident, with a cluster
    bootstrap. The draws are bootstrap_ci's; a resample whose denominators are all 0 has no ratio
    and is left out of the percentiles."""
    num = np.asarray(numerators, dtype=float)
    den = np.asarray(denominators, dtype=float)
    if len(den) == 0 or den.sum() == 0:
        return Estimate(None, n=len(den))
    draws = np.random.default_rng(EVAL_SEED).integers(0, len(den), size=(SAMPLES, len(den)))
    values = np.array([num[ix].sum() / den[ix].sum() if den[ix].sum() > 0 else np.nan
                       for ix in draws])  # fmt: skip
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return Estimate(_round(num.sum() / den.sum()), _round(lo), _round(hi), len(den))


def golden_metrics(
    incidents: Sequence[Incident], stored: Mapping[str, StoredCase]
) -> dict[str, Estimate]:
    """The golden-set metrics (design D4) of one arm's stored runs."""
    runs = [(i, stored[i.id].record()) for i in incidents if i.id in stored]
    scores = [score_case(i, r) for i, r in runs]
    out: dict[str, Estimate] = {
        "root_cause_accuracy": rate([s.correct for s in scores]),
        "root_cause_correct": count(sum(s.correct for s in scores), len(scores)),
        "lenient_accuracy": rate([s.lenient for s in scores]),
        "refusals": count(sum(s.predicted == "refused" for s in scores), len(scores)),
    }
    controls = [s for s in scores if s.control]
    out["cause_alarm_rate"] = rate([s.cause_alarm for s in controls])
    out["cause_alarms"] = count(sum(s.cause_alarm for s in controls), len(controls))
    out["change_alarm_rate"] = rate([s.change_alarm for s in controls])
    out["window_iou"] = mean([s.window_iou for s in scores])
    out["window_hit_rate"] = rate([s.window_iou >= 0.5 for s in scores])
    out["direction_accuracy"] = rate([s.direction_ok for s in scores if s.direction_ok is not None])
    out["decided_share"] = mean([s.decided / 5 for s in scores])

    stats = [st for st in (claim_stats(r) for _, r in runs) if st is not None]
    drafted = [st.drafted for st in stats]
    out["pre_verifier_pass_rate"] = ratio([st.shipped for st in stats], drafted)
    for rule in WITHHELD_RULES:
        out[f"withheld.{rule}"] = ratio([st.withheld_by_rule.get(rule, 0) for st in stats], drafted)
    out["reworded"] = ratio([st.reworded for st in stats], [st.shipped for st in stats])

    tools = [tool_call_stats(r.trace) for _, r in runs]
    planned = [t.planned for t in tools]
    out["tool_error_rate"] = ratio([t.errors for t in tools], planned)
    out["tool_repeat_rate"] = ratio([t.repeats for t in tools], planned)

    reports = [r.result for _, r in runs if isinstance(r.result, Report)]
    seconds = [rep.seconds for rep in reports if rep.seconds is not None]
    out["seconds_median"] = Estimate(_round(float(np.median(seconds))) if seconds else None,
                                     n=len(seconds))  # fmt: skip
    out["seconds_p95"] = Estimate(
        _round(float(np.percentile(seconds, 95))) if seconds else None, n=len(seconds)
    )
    out["tokens_in"] = mean([sum(u.input_tokens for u in rep.usage) for rep in reports])
    out["tokens_out"] = mean([sum(u.output_tokens for u in rep.usage) for rep in reports])
    out["model_calls"] = mean([len(rep.usage) for rep in reports])

    blames = [b for b in (patch_blame(r) for i, r in runs if i.primary_cause != "patch") if b]
    out["patch_blame_before"] = rate([before > 0 for before, _ in blames])
    out["patch_blame_after"] = rate([after > 0 for _, after in blames])
    return out


def per_category(scores: Sequence[CaseScore]) -> dict[str, tuple[int, int]]:
    """Correct and total cases per expected category."""
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for s in scores:
        tally[s.expected][0] += int(s.correct)
        tally[s.expected][1] += 1
    return {category: (c, n) for category, (c, n) in sorted(tally.items())}


def confusion(scores: Sequence[CaseScore]) -> dict[str, int]:
    """Cases per `expected -> predicted` pair."""
    pairs = Counter(f"{s.expected} -> {s.predicted}" for s in scores)
    return dict(sorted(pairs.items()))


def paired_difference[T](
    base: Mapping[str, T], other: Mapping[str, T], metric: Callable[[T], float | None]
) -> Estimate:
    """The mean of `metric` in `other` minus in `base`, over the cases both have a value for,
    with a paired bootstrap interval."""
    pairs = [(metric(other[key]), metric(base[key])) for key in sorted(set(base) & set(other))]
    kept = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not kept:
        return Estimate(None)
    a = np.asarray([x for x, _ in kept], dtype=float)
    b = np.asarray([y for _, y in kept], dtype=float)
    lo, hi = paired_bootstrap_ci(
        lambda ix: float(a[ix].mean()),
        lambda ix: float(b[ix].mean()),
        len(kept),
        samples=SAMPLES,
        seed=EVAL_SEED,
    )
    return Estimate(_round(a.mean() - b.mean()), _round(lo), _round(hi), len(kept))


def a1(
    stored: Mapping[str, StoredCase], support: Mapping[str, Mapping[str, bool | None]]
) -> dict[str, Estimate]:
    """A1, the unsupported-claim rate with the verifier off and on (design D7). Off: the claims
    the verifier withheld plus the shipped ones the judge found unsupported, over the drafted
    claims judged or withheld. On: the shipped claims judged unsupported, over the shipped claims
    judged. A claim the judge hasn't scored counts in neither."""
    off_num: list[float] = []
    off_den: list[float] = []
    on_num: list[float] = []
    on_den: list[float] = []
    for case_id, case in sorted(stored.items()):
        record = case.record()
        if claim_stats(record) is None or not isinstance(record.result, Report):
            continue
        verdicts = support.get(case_id, {})
        judged = [verdicts.get(c.claim_id) for c in record.result.claims]
        known = [v for v in judged if v is not None]
        unsupported = sum(v is False for v in known)
        withheld = len(record.result.withheld_claims)
        off_num.append(withheld + unsupported)
        off_den.append(withheld + len(known))
        on_num.append(unsupported)
        on_den.append(len(known))
    return {"unsupported_off": ratio(off_num, off_den), "unsupported_on": ratio(on_num, on_den)}
