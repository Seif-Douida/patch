# Relevance judgments: Claude vs Seif (blind audit)

300 pool pairs drawn at random and graded independently: 0 not relevant, 1 partly relevant, 2 relevant. Rule fixed before the comparison (ADR-019 extended): the judgments pass if the quadratic-weighted kappa is at least 0.70.

The judges' grades were given under guidelines v2 (130 pairs) and v3 (137 pairs).

This check is optimistic (design D7): the first check failed, the guidelines were revised on this audit's disagreements, and pairs were re-judged under them. It measures the revision on the pairs it was tuned on.

The gate counts the 267 pairs Seif read. The other 33 pairs are reviews in languages other than English and French that he couldn't read and graded 1, unsure, by default: not a relevance judgment, so they were left out (decided before any comparison). The judges' grades on other languages are audited only on the pairs he translated.

Result: **failed**. Quadratic-weighted kappa: 0.67.

Reported, not gated: kappa over all 300 pairs: 0.63.

Reported outside the rule, on the gate's pairs: kappa at relevant (grade 1 or more) 0.61; per judge model:

| Judge | Pairs | Weighted kappa |
|---|---|---|
| claude-opus-5-5 | 55 | 0.00 |
| claude-sonnet-5-5 | 212 | 0.64 |

| Judges \ Seif | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 30 | 8 | 10 |
| 1 | 8 | 14 | 23 |
| 2 | 3 | 12 | 159 |
