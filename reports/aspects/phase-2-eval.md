# Phase 2 evaluation: aspect students against the teacher

Plan Task 14, design D6. The champion was chosen on gold dev before gold test was read; gold test was then used once, for this report. Numbers only: no review text.

## Champion: s0

S0 has the best dev macro-F1 of the models fast enough.

Rules fixed before any results: at least 70 reviews a second on one CPU; then the best dev macro-F1, with each dev review judged by thresholds tuned on the other folds; a statistical tie with a cheaper model (paired 95% interval containing 0) goes to the cheaper one (S0, then S1a, then S1b).

| Student | Version | Reviews a second (1 CPU) | Dev macro-F1 | Eligible |
|---|---|---|---|---|
| s0 | `20261012T223218Z` | 4,134.4 | 0.647 | yes |
| s1a | `20261012T205209Z` | 9.0 | 0.581 | no |
| s1b | `20261012T215556Z` | 8.8 | 0.650 | no |

## Gold test (Claude's labels, audited by Seif), 450 reviews

| Model | Macro-F1 (95% CI) | Micro-F1 | English (n=298) | French (n=152) | ECE | Brier |
|---|---|---|---|---|---|---|
| teacher | 0.818 (0.752 to 0.868) | 0.895 | 0.844 | 0.759 | n/a | n/a |
| s0 | 0.614 (0.538 to 0.676) | 0.738 | 0.674 | 0.464 | 0.053 | 0.043 |
| s1a | 0.634 (0.551 to 0.702) | 0.715 | 0.665 | 0.544 | 0.103 | 0.080 |
| s1b | 0.657 (0.586 to 0.715) | 0.786 | 0.692 | 0.576 | 0.109 | 0.074 |

F1 per aspect:

| Aspect | Positives | teacher | s0 | s1a | s1b |
|---|---|---|---|---|---|
| performance | 52 | 0.914 | 0.865 | 0.699 | 0.816 |
| stability_bugs | 46 | 0.901 | 0.658 | 0.688 | 0.810 |
| gameplay_balance | 117 | 0.911 | 0.819 | 0.778 | 0.840 |
| content | 88 | 0.877 | 0.719 | 0.722 | 0.740 |
| monetization | 19 | 0.905 | 0.556 | 0.686 | 0.667 |
| online_servers | 6 | 0.769 | 0.400 | 0.571 | 0.364 |
| story_world | 105 | 0.938 | 0.766 | 0.794 | 0.887 |
| ux_controls | 27 | 0.828 | 0.476 | 0.406 | 0.468 |
| platform_policy | 1 | 0.286 | 0.182 | 0.333 | 0.200 |
| off_topic | 26 | 0.846 | 0.698 | 0.667 | 0.783 |

Paired differences in macro-F1:

| Comparison | Difference (95% CI) | Tied |
|---|---|---|
| teacher - s0 | +0.204 (+0.145 to +0.267) | no |
| teacher - s1a | +0.183 (+0.125 to +0.242) | no |
| teacher - s1b | +0.160 (+0.100 to +0.222) | no |
| s0 - s1a | -0.020 (-0.091 to +0.055) | yes |
| s0 - s1b | -0.043 (-0.080 to -0.005) | no |
| s1a - s1b | -0.023 (-0.092 to +0.045) | yes |

## Seif's 150 (his own labels; part of gold test), 150 reviews

| Model | Macro-F1 (95% CI) | Micro-F1 | English (n=100) | French (n=50) | ECE | Brier |
|---|---|---|---|---|---|---|
| teacher | 0.791 (0.709 to 0.838) | 0.893 | 0.835 | 0.681 | n/a | n/a |
| s0 | 0.568 (0.495 to 0.623) | 0.741 | 0.606 | 0.370 | 0.057 | 0.044 |
| s1a | 0.611 (0.511 to 0.679) | 0.719 | 0.655 | 0.435 | 0.097 | 0.074 |
| s1b | 0.595 (0.520 to 0.648) | 0.782 | 0.629 | 0.409 | 0.108 | 0.073 |

F1 per aspect:

| Aspect | Positives | teacher | s0 | s1a | s1b |
|---|---|---|---|---|---|
| performance | 16 | 0.938 | 0.824 | 0.667 | 0.824 |
| stability_bugs | 13 | 0.923 | 0.815 | 0.759 | 0.815 |
| gameplay_balance | 45 | 0.882 | 0.814 | 0.753 | 0.822 |
| content | 29 | 0.881 | 0.781 | 0.688 | 0.724 |
| monetization | 6 | 0.923 | 0.571 | 0.667 | 0.769 |
| online_servers | 4 | 0.750 | 0.000 | 0.500 | 0.000 |
| story_world | 31 | 0.923 | 0.716 | 0.807 | 0.954 |
| ux_controls | 12 | 0.917 | 0.444 | 0.696 | 0.375 |
| platform_policy | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| off_topic | 9 | 0.778 | 0.714 | 0.571 | 0.667 |

Paired differences in macro-F1:

| Comparison | Difference (95% CI) | Tied |
|---|---|---|
| teacher - s0 | +0.223 (+0.141 to +0.296) | no |
| teacher - s1a | +0.181 (+0.102 to +0.267) | no |
| teacher - s1b | +0.196 (+0.115 to +0.270) | no |
| s0 - s1a | -0.043 (-0.102 to +0.035) | yes |
| s0 - s1b | -0.027 (-0.081 to +0.025) | yes |
| s1a - s1b | +0.016 (-0.061 to +0.073) | yes |

## Agreement with the teacher (silver holdout, 3,000 reviews)

Every language the silver set holds; this is the measure Phase 7 tracks online.

| Student | Macro-F1 against the teacher | Micro-F1 |
|---|---|---|
| s0 | 0.552 | 0.659 |
| s1a | 0.591 | 0.671 |
| s1b | 0.683 | 0.749 |

## Size and speed

| Student | Version | Size (MB) | Reviews a second (1 CPU) | Dev macro-F1 |
|---|---|---|---|---|
| s0 | `20261012T223218Z` | 37.6 | 4,134.4 | 0.647 |
| s1a | `20261012T205209Z` | 135.2 | 9.0 | 0.581 |
| s1b | `20261012T215556Z` | 135.7 | 8.8 | 0.650 |

## Method and limitations

- The teacher is Gemma 4 31B on a self-hosted server (ADR-020): it labeled the 30,000 silver reviews the students learned from. Its scores here have no thresholds or probabilities.
- Each student's thresholds were tuned on gold dev; gold test played no part in any choice.
- Per-language quality is measured for English and French only. Other languages are covered by agreement with the teacher on the silver holdout.
- Gold test labels are Claude's, audited blind by Seif on 150 reviews (mean kappa 0.99). Seif's 150 are scored against his own labels.
- Macro-F1 weighs all 10 aspects equally. online_servers and platform_policy have few positives in gold, so their F1 moves a lot with a review or two.
- Students are trained with balanced class weights, so their probabilities lean high for rare aspects: ECE and Brier describe that, the thresholds correct for it.
- Intervals are 95% percentile bootstraps over reviews (2,000 resamples); paired intervals resample both models on the same reviews.
