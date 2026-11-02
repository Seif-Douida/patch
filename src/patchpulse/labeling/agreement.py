"""Does Claude label like Seif? The gold set's acceptance check (design D2, ADR-019).

Seif labels 150 test reviews blind; this compares his labels with Claude's on the same reviews,
aspect by aspect, with Cohen's kappa. The rule was fixed before any comparison:

* an aspect is assessable if at least 8 of the 150 are positive for either annotator (below that,
  kappa swings on one or two reviews and is reported as "too rare to judge");
* the gold set passes if the mean kappa over assessable aspects is at least 0.70 and every
  assessable aspect reaches 0.60.

Seif's reliability is reported beside it (ADR-019): his blind re-label of 30 of the 150, at
least 14 days later, against his own audit answers, with no pass rule. With 30 reviews an aspect
with few positives swings on one review, so the positives stand beside each kappa, and one kappa
over every review-aspect pair sums it up.

The rendered report holds counts and kappas only: no review ids, no text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from patchpulse.labeling.audit import RELABEL_MIN_DAYS, RELABEL_SIZE
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.metrics import cohen_kappa

MIN_POSITIVES = 8
MACRO_PASS = 0.70
ASPECT_PASS = 0.60


@dataclass(frozen=True)
class AspectAgreement:
    code: str
    positives_claude: int
    positives_seif: int
    kappa: float | None
    assessable: bool

    @property
    def passes(self) -> bool | None:
        if not self.assessable or self.kappa is None:
            return None
        return self.kappa >= ASPECT_PASS


@dataclass(frozen=True)
class AgreementReport:
    n_reviews: int
    aspects: list[AspectAgreement]
    macro_kappa: float | None

    @property
    def failing(self) -> list[str]:
        return [row.code for row in self.aspects if row.passes is False]

    @property
    def passed(self) -> bool:
        return self.macro_kappa is not None and self.macro_kappa >= MACRO_PASS and not self.failing


def agreement(
    claude: Mapping[int, frozenset[str]], seif: Mapping[int, frozenset[str]], ids: Sequence[int]
) -> AgreementReport:
    missing = [i for i in ids if i not in claude or i not in seif]
    if missing:
        raise ValueError(f"{len(missing)} audited reviews lack a label from one annotator")
    rows = []
    for code in ASPECTS:
        a = [code in claude[i] for i in ids]
        b = [code in seif[i] for i in ids]
        rows.append(
            AspectAgreement(
                code=code,
                positives_claude=sum(a),
                positives_seif=sum(b),
                kappa=cohen_kappa(a, b),
                assessable=max(sum(a), sum(b)) >= MIN_POSITIVES,
            )
        )
    kappas = [row.kappa for row in rows if row.assessable and row.kappa is not None]
    macro = sum(kappas) / len(kappas) if kappas else None
    return AgreementReport(n_reviews=len(ids), aspects=rows, macro_kappa=macro)


@dataclass(frozen=True)
class RelabelAspect:
    code: str
    positives_first: int  # in Seif's audit
    positives_second: int  # in his re-label
    kappa: float | None


@dataclass(frozen=True)
class SelfAgreement:
    n_reviews: int
    aspects: list[RelabelAspect]
    pooled_kappa: float | None  # over every review-aspect pair
    identical: int  # reviews given exactly the same aspects both times


def self_agreement(
    first: Mapping[int, frozenset[str]], second: Mapping[int, frozenset[str]], ids: Sequence[int]
) -> SelfAgreement:
    """Seif's re-label against his audit answers on the re-labeled reviews."""
    missing = [i for i in ids if i not in first or i not in second]
    if missing:
        raise ValueError(f"{len(missing)} re-labeled review(s) lack an answer from one of the two")
    rows = []
    pooled_first: list[bool] = []
    pooled_second: list[bool] = []
    for code in ASPECTS:
        a = [code in first[i] for i in ids]
        b = [code in second[i] for i in ids]
        rows.append(RelabelAspect(code, sum(a), sum(b), cohen_kappa(a, b)))
        pooled_first += a
        pooled_second += b
    return SelfAgreement(
        n_reviews=len(ids),
        aspects=rows,
        pooled_kappa=cohen_kappa(pooled_first, pooled_second),
        identical=sum(first[i] == second[i] for i in ids),
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_agreement(
    report: AgreementReport, reliability: SelfAgreement | None = None, *, relabeled: int = 0
) -> str:
    """The report; `relabeled` counts the re-label's answers so far while it's unfinished."""
    return _render_audit(report) + _render_reliability(reliability, relabeled, report.n_reviews)


def _render_reliability(reliability: SelfAgreement | None, relabeled: int, audited: int) -> str:
    lines = ["", "## Seif's re-label (intra-annotator agreement)", ""]
    if reliability is None:
        lines.append(
            f"Pending: Seif re-labels {RELABEL_SIZE} of the {audited}, blind, at least "
            f"{RELABEL_MIN_DAYS} days after the audit ({relabeled} of {RELABEL_SIZE} so far)."
        )
        return "\n".join(lines) + "\n"
    lines += [
        f"Seif re-labeled {reliability.n_reviews} of the {audited}, blind, at least "
        f"{RELABEL_MIN_DAYS} days after the audit, and is compared here with his own audit "
        "answers (ADR-019). There is no pass rule. With so few reviews an aspect with few "
        "positives swings on one review, so the positives stand beside each kappa.",
        "",
        f"Over all {reliability.n_reviews * len(reliability.aspects)} review-aspect pairs: kappa "
        f"{_fmt(reliability.pooled_kappa)}. {reliability.identical} of {reliability.n_reviews} "
        "reviews got exactly the same aspects both times.",
        "",
        "| Aspect | Audit positives | Re-label positives | Kappa |",
        "|---|---|---|---|",
    ]
    for row in reliability.aspects:
        lines.append(
            f"| {row.code} | {row.positives_first} | {row.positives_second} | {_fmt(row.kappa)} |"
        )
    return "\n".join(lines) + "\n"


def _render_audit(report: AgreementReport) -> str:
    verdict = "**passed**" if report.passed else "**failed**"
    lines = [
        "# Gold-set agreement: Claude vs Seif (blind audit)",
        "",
        f"{report.n_reviews} test reviews, labeled independently by both annotators. Rule fixed "
        f"before the comparison (ADR-019): an aspect is assessable with at least {MIN_POSITIVES} "
        f"positives for either annotator; the gold set passes if the mean kappa over assessable "
        f"aspects is at least {MACRO_PASS:.2f} and every assessable aspect is at least "
        f"{ASPECT_PASS:.2f}.",
        "",
        f"Result: {verdict}. Mean kappa over assessable aspects: {_fmt(report.macro_kappa)}.",
        "",
        "| Aspect | Claude positives | Seif positives | Kappa | Verdict |",
        "|---|---|---|---|---|",
    ]
    for row in report.aspects:
        verdict_cell = {True: "pass", False: "**fail**", None: "too rare to judge"}[row.passes]
        if row.assessable and row.kappa is None:
            verdict_cell = "undefined"
        lines.append(
            f"| {row.code} | {row.positives_claude} | {row.positives_seif} | "
            f"{_fmt(row.kappa)} | {verdict_cell} |"
        )
    return "\n".join(lines) + "\n"
