"""Does Claude label like Seif? The gold set's acceptance check (design D2, ADR-019).

Seif labels 150 test reviews blind; this compares his labels with Claude's on the same reviews,
aspect by aspect, with Cohen's kappa. The rule was fixed before any comparison:

* an aspect is assessable if at least 8 of the 150 are positive for either annotator (below that,
  kappa swings on one or two reviews and is reported as "too rare to judge");
* the gold set passes if the mean kappa over assessable aspects is at least 0.70 and every
  assessable aspect reaches 0.60.

The rendered report holds counts and kappas only: no review ids, no text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_agreement(report: AgreementReport) -> str:
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
