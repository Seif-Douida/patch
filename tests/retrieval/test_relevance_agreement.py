"""Does Seif grade relevance like the judges? Quadratic-weighted kappa on his blind audit of 300
pool pairs (plan Task 11, design D7, ADR-019 extended); the pass rule, 0.70, was fixed before any
comparison."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from patchpulse.retrieval.agreement import (
    PASS,
    relevance_agreement,
    render_relevance_agreement,
    weighted_kappa,
)


def test_weighted_kappa_matches_a_hand_computed_table() -> None:
    # Four pairs: three agreements and one 0-vs-1. Weights (i-j)^2/4: 0.25 for one step, 1 for two.
    # Observed weighted disagreement 0.25 * 1/4 = 0.0625. Marginals a = (.5, .25, .25),
    # b = (.25, .5, .25); expected = 0.25*(.25+.0625) + 1*(.125+.0625) + 0.25*(.0625+.125)
    # = 0.3125. Kappa = 1 - 0.0625 / 0.3125 = 0.8.
    assert weighted_kappa([0, 1, 2, 0], [0, 1, 2, 1]) == pytest.approx(0.8)
    assert weighted_kappa([0, 1, 2, 2], [0, 1, 2, 2]) == pytest.approx(1.0)


def test_weighted_kappa_matches_scikit_learn() -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        a = rng.integers(0, 3, size=300)
        noise = rng.integers(-1, 2, size=300) * (rng.random(300) < 0.3)
        b = np.clip(a + noise, 0, 2)
        expected = cohen_kappa_score(a, b, weights="quadratic", labels=[0, 1, 2])
        assert weighted_kappa(a.tolist(), b.tolist()) == pytest.approx(expected)


def test_weighted_kappa_is_none_when_undefined() -> None:
    # Both graders gave every pair the same grade: chance agreement is total, kappa undefined.
    assert weighted_kappa([2, 2, 2], [2, 2, 2]) is None
    assert weighted_kappa([0, 0], [2, 2]) == pytest.approx(0.0)  # defined: always apart
    with pytest.raises(ValueError, match="equal-length"):
        weighted_kappa([0, 1], [0])
    with pytest.raises(ValueError, match="equal-length"):
        weighted_kappa([], [])
    with pytest.raises(ValueError, match="0 to 2"):
        weighted_kappa([0, 3], [0, 1])


def pairs(n: int) -> list[tuple[str, int]]:
    return [(f"q{i % 7:03d}", 1000 + i) for i in range(n)]


def test_agreement_refuses_until_all_300_are_graded() -> None:
    audit = pairs(300)
    claude = {pair: (i % 3) for i, pair in enumerate(audit)}
    judges = {
        pair: "claude-opus-5-5" if i < 100 else "claude-sonnet-5-5" for i, pair in enumerate(audit)
    }
    seif = dict(list(claude.items())[:299])

    with pytest.raises(ValueError, match="1 of the 300 audit pairs"):
        relevance_agreement(claude, seif, audit, judges=judges)

    seif[audit[299]] = claude[audit[299]]
    for pair in audit[:30]:  # 30 one-step disagreements
        seif[pair] = 1 if claude[pair] != 1 else 2
    result = relevance_agreement(claude, seif, audit, judges=judges)

    assert result.n == 300
    assert sum(map(sum, result.table)) == 300
    assert result.table[0][1] == 10  # Claude 0, Seif 1
    assert result.kappa == pytest.approx(
        cohen_kappa_score([claude[p] for p in audit], [seif[p] for p in audit], weights="quadratic")
    )
    assert result.binary_kappa is not None  # relevant = grade >= 1, reported apart from the rule
    assert set(result.by_judge) == {"claude-opus-5-5", "claude-sonnet-5-5"}
    assert result.by_judge["claude-sonnet-5-5"][0] == 200  # pairs per judge
    assert result.passed == (result.kappa is not None and result.kappa >= PASS)
    assert PASS == 0.70


def test_report_holds_counts_and_kappas_only() -> None:
    audit = pairs(300)
    claude = {pair: (i % 3) for i, pair in enumerate(audit)}
    judges = {pair: "claude-sonnet-5-5" for pair in audit}
    result = relevance_agreement(claude, dict(claude), audit, judges=judges)

    text = render_relevance_agreement(result)

    assert "**passed**" in text
    assert "1.00" in text
    assert "q00" not in text  # no query ids
    assert "1000" not in text  # no review ids
