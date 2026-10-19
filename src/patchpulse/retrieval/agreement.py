"""Does Seif grade relevance like the judges? The eval's acceptance check (design D7, ADR-019
extended).

Seif grades 300 pool pairs blind; this compares his grades with the judges' on the same pairs with
Cohen's kappa, quadratic-weighted (a 0-vs-2 disagreement costs four times a one-step one). The
rule was fixed before any comparison: the judgments pass if kappa is at least 0.70. A binary kappa
at "relevant" (grade 1 or more) is reported beside it, outside the rule, and so is kappa per judge
model (two models judged the pool: Opus, then Sonnet after a usage limit).

The rendered report holds counts and kappas only: no query or review ids, no text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from patchpulse.models.metrics import cohen_kappa

PASS = 0.70
GRADES = (0, 1, 2)
# The guidelines the pool was judged under when Seif's audit was drawn and graded. Pairs judged
# under a later version were re-judged after the first check, with guidelines revised on this
# audit's disagreements (D7), so a check that counts them is optimistic.
AUDITED_GUIDELINES = "v2"

Pair = tuple[str, int]


def weighted_kappa(a: Sequence[int], b: Sequence[int], *, categories: int = 3) -> float | None:
    """Quadratic-weighted kappa for two graders' ordinal grades (0 to `categories` - 1).

    kappa = 1 - sum(w * observed) / sum(w * expected), with w = (i - j)^2 / (k - 1)^2, observed the
    joint distribution of the grades and expected the product of the two graders' marginals.
    Undefined (None) when the expected weighted disagreement is 0: both gave one same grade to all.
    """
    x, y = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    if x.shape != y.shape or x.ndim != 1 or x.size == 0:
        raise ValueError("weighted_kappa needs two equal-length, non-empty 1-D grade sequences")
    if x.min() < 0 or y.min() < 0 or x.max() >= categories or y.max() >= categories:
        raise ValueError(f"grades must be integers from 0 to {categories - 1}")
    observed = np.zeros((categories, categories))
    np.add.at(observed, (x, y), 1.0)
    observed /= x.size
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    steps = np.arange(categories)
    weights = (steps[:, None] - steps[None, :]) ** 2 / (categories - 1) ** 2
    chance = float((weights * expected).sum())
    if chance == 0.0:
        return None
    return 1.0 - float((weights * observed).sum()) / chance


@dataclass(frozen=True)
class RelevanceAgreement:
    n: int
    kappa: float | None
    binary_kappa: float | None
    table: list[list[int]]  # rows: the judges' grade 0..2; columns: Seif's
    by_judge: dict[str, tuple[int, float | None]]  # judge model -> (pairs, kappa)

    @property
    def passed(self) -> bool:
        return self.kappa is not None and self.kappa >= PASS


def relevance_agreement(
    claude: Mapping[Pair, int],
    seif: Mapping[Pair, int],
    audit: Sequence[Pair],
    *,
    judges: Mapping[Pair, str],
) -> RelevanceAgreement:
    """Agreement on the audit pairs; refuses until every one has both grades."""
    missing = [pair for pair in audit if pair not in seif or pair not in claude]
    if missing:
        raise ValueError(
            f"{len(missing)} of the {len(audit)} audit pairs lack a grade from Seif or the judges"
        )
    a = [int(claude[pair]) for pair in audit]
    b = [int(seif[pair]) for pair in audit]
    table = [[0] * len(GRADES) for _ in GRADES]
    for x, y in zip(a, b, strict=True):
        table[x][y] += 1
    by_judge: dict[str, tuple[int, float | None]] = {}
    for judge in sorted({judges[pair] for pair in audit}):
        mine = [i for i, pair in enumerate(audit) if judges[pair] == judge]
        by_judge[judge] = (len(mine), weighted_kappa([a[i] for i in mine], [b[i] for i in mine]))
    return RelevanceAgreement(
        n=len(audit),
        kappa=weighted_kappa(a, b),
        binary_kappa=cohen_kappa([x >= 1 for x in a], [y >= 1 for y in b]),
        table=table,
        by_judge=by_judge,
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def version_key(version: str) -> tuple[int, str]:
    """Guidelines versions in order: v2 before v10."""
    number = re.fullmatch(r"v(\d+)", version)
    return (int(number.group(1)) if number else -1, version)


def render_relevance_agreement(
    result: RelevanceAgreement,
    *,
    overall: RelevanceAgreement | None = None,
    guidelines: Mapping[str, int] | None = None,
) -> str:
    """`result` is the gate; `overall`, when given, is every audit pair, unread ones included;
    `guidelines` counts the gate's pairs by the guidelines version they were judged under."""
    verdict = "**passed**" if result.passed else "**failed**"
    total = result.n if overall is None else overall.n
    lines = [
        "# Relevance judgments: Claude vs Seif (blind audit)",
        "",
        f"{total} pool pairs drawn at random and graded independently: 0 not relevant, "
        "1 partly relevant, 2 relevant. Rule fixed before the comparison (ADR-019 extended): the "
        f"judgments pass if the quadratic-weighted kappa is at least {PASS:.2f}.",
        "",
    ]
    if guidelines:
        versions = [f"{v} ({guidelines[v]} pairs)" for v in sorted(guidelines, key=version_key)]
        listed = (
            ", ".join(versions[:-1]) + " and " + versions[-1] if len(versions) > 1 else versions[0]
        )
        lines += [f"The judges' grades were given under guidelines {listed}.", ""]
        if set(guidelines) - {AUDITED_GUIDELINES}:
            lines += [
                "This check is optimistic (design D7): the first check failed, the guidelines were "
                "revised on this audit's disagreements, and pairs were re-judged under them. It "
                "measures the revision on the pairs it was tuned on.",
                "",
            ]
    if overall is not None:
        lines += [
            f"The gate counts the {result.n} pairs Seif read. The other {overall.n - result.n} "
            "pairs are reviews in languages other than English and French that he couldn't read "
            "and graded 1, unsure, by default: not a relevance judgment, so they were left out "
            "(decided before any comparison). The judges' grades on other languages are audited "
            "only on the pairs he translated.",
            "",
        ]
    lines += [
        f"Result: {verdict}. Quadratic-weighted kappa: {_fmt(result.kappa)}.",
        "",
    ]
    if overall is not None:
        lines += [
            f"Reported, not gated: kappa over all {overall.n} pairs: {_fmt(overall.kappa)}.",
            "",
        ]
    lines += [
        f"Reported outside the rule, on the gate's pairs: kappa at relevant (grade 1 or more) "
        f"{_fmt(result.binary_kappa)}; per judge model:",
        "",
        "| Judge | Pairs | Weighted kappa |",
        "|---|---|---|",
    ]
    for judge, (n, kappa) in result.by_judge.items():
        lines.append(f"| {judge} | {n} | {_fmt(kappa)} |")
    lines += [
        "",
        "| Judges \\ Seif | 0 | 1 | 2 |",
        "|---|---|---|---|",
    ]
    for grade, row in zip(GRADES, result.table, strict=True):
        lines.append(f"| {grade} | " + " | ".join(str(count) for count in row) + " |")
    return "\n".join(lines) + "\n"
