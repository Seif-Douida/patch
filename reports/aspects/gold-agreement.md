# Gold-set agreement: Claude vs Seif (blind audit)

150 test reviews, labeled independently by both annotators. Rule fixed before the comparison (ADR-019): an aspect is assessable with at least 8 positives for either annotator; the gold set passes if the mean kappa over assessable aspects is at least 0.70 and every assessable aspect is at least 0.60.

Result: **passed**. Mean kappa over assessable aspects: 0.99.

| Aspect | Claude positives | Seif positives | Kappa | Verdict |
|---|---|---|---|---|
| performance | 16 | 16 | 1.00 | pass |
| stability_bugs | 13 | 13 | 1.00 | pass |
| gameplay_balance | 46 | 45 | 0.98 | pass |
| content | 28 | 29 | 0.98 | pass |
| monetization | 6 | 6 | 1.00 | too rare to judge |
| online_servers | 4 | 4 | 1.00 | too rare to judge |
| story_world | 33 | 31 | 0.96 | pass |
| ux_controls | 12 | 12 | 1.00 | pass |
| platform_policy | 0 | 0 | n/a | too rare to judge |
| off_topic | 9 | 9 | 1.00 | pass |

## Seif's re-label (intra-annotator agreement)

Seif re-labeled 30 of the 150, blind, at least 14 days after the audit, and is compared here with his own audit answers (ADR-019). There is no pass rule. With so few reviews an aspect with few positives swings on one review, so the positives stand beside each kappa.

Over all 300 review-aspect pairs: kappa 0.72. 18 of 30 reviews got exactly the same aspects both times.

| Aspect | Audit positives | Re-label positives | Kappa |
|---|---|---|---|
| performance | 1 | 1 | 1.00 |
| stability_bugs | 3 | 3 | 1.00 |
| gameplay_balance | 9 | 3 | 0.22 |
| content | 5 | 4 | 0.87 |
| monetization | 0 | 1 | 0.00 |
| online_servers | 1 | 1 | 1.00 |
| story_world | 8 | 8 | 1.00 |
| ux_controls | 2 | 1 | 0.65 |
| platform_policy | 0 | 0 | n/a |
| off_topic | 1 | 2 | -0.05 |
