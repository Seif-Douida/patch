# Drift thresholds, calibrated on past weeks

156 weeks measured, 0 skipped (fewer than 200 reviews). Each threshold is its statistic's 99th percentile over the weeks measured (Phase 7 design D2).

| Rule | Statistic | p50 | p90 | p95 | p99 | Threshold | Weeks above it | Weeks measured |
|---|---|---|---|---|---|---|---|---|
| drift_language | language_psi | 0.063 | 0.277 | 0.316 | 0.356 | 0.3557 | 2 | 156 |
| drift_length | length_psi | 0.013 | 0.220 | 0.282 | 0.354 | 0.3538 | 2 | 156 |
| drift_embedding | embedding_auc | 0.681 | 0.880 | 0.892 | 0.901 | 0.9009 | 2 | 156 |
| drift_aspects | aspect_max_change | 8.971 | 18.505 | 21.785 | 26.228 | 26.23 | 2 | 156 |
| drift_confidence | uncertain_change | 2.354 | 5.577 | 5.882 | 6.369 | 6.369 | 2 | 156 |

The embedding statistic needs each week's e5 vectors from the review index, which holds each game's 40,000 most recent reviews and its patch and launch windows (ADR-022): it is measured on the weeks the index covers, the other statistics on every week.
