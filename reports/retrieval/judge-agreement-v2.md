# Relevance judgments: Claude vs Seif (blind audit)

300 pool pairs drawn at random and graded independently: 0 not relevant, 1 partly relevant, 2 relevant. Rule fixed before the comparison (ADR-019 extended): the judgments pass if the quadratic-weighted kappa is at least 0.70.

The judges' grades were given under guidelines v2 (267 pairs).

The gate counts the 267 pairs Seif read. The other 33 pairs are reviews in languages other than English and French that he couldn't read and graded 1, unsure, by default: not a relevance judgment, so they were left out (decided before any comparison). The judges' grades on other languages are audited only on the pairs he translated.

Result: **failed**. Quadratic-weighted kappa: 0.62.

Reported, not gated: kappa over all 300 pairs: 0.59.

Reported outside the rule, on the gate's pairs: kappa at relevant (grade 1 or more) 0.60; per judge model:

| Judge | Pairs | Weighted kappa |
|---|---|---|
| claude-opus-5-5 | 102 | 0.53 |
| claude-sonnet-5-5 | 165 | 0.67 |

| Judges \ Seif | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 33 | 10 | 14 |
| 1 | 8 | 21 | 51 |
| 2 | 0 | 3 | 127 |
