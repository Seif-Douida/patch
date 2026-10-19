# Phase 3 retrieval eval

100 frozen queries (76 aspect, 19 event, 5 crosslingual; 73 in english, 27 in french). Each system's top 20 per query, base and filtered, pooled into 5,930 judged pairs; a result nobody judged counts as not relevant. Relevant means grade 1 or more (0 not relevant, 1 partly, 2 relevant). Means over queries with 95% bootstrap intervals (2,000 resamples of the queries). 0 queries have no relevant review in their pool: left out of the means, never scored.

## The judgments and their agreement with Seif

Judged by claude-sonnet-5-5 (4,744 pairs), claude-opus-5-5 (1,186 pairs); guidelines v2 (3,217 pairs), v3 (2,713 pairs).

Quadratic-weighted kappa against Seif's blind audit: 0.670 on the 267 audit pairs he read (rule fixed in advance: at least 0.70): **failed**. 33 audit pairs in languages he couldn't read are left out.

| Query kind | Audit pairs | Weighted kappa |
|---|---|---|
| aspect | 185 | 0.749 |
| event | 73 | 0.469 |
| crosslingual | 9 | 0.833 |

The judgments failed the agreement rule and were accepted anyway (`--accept-failed-agreement`): Seif chose not to grade further after two failed checks (reports/retrieval/judge-agreement-v2.md, then -v3.md, which is optimistic). Every number below rests on these judgments, the event queries' least of all.

## Every system

| System | Variant | Recall@10 | Recall@20 | NDCG@10 | MRR@10 |
|---|---|---|---|---|---|
| bm25 | base | 0.185 [0.173, 0.199] | 0.360 [0.341, 0.379] | 0.865 [0.814, 0.907] | 0.963 [0.927, 0.990] |
| bm25 | filtered | 0.194 [0.183, 0.206] | 0.368 [0.351, 0.384] | 0.902 [0.866, 0.932] | 0.990 [0.975, 1.000] |
| dense:e5-small | base | 0.184 [0.172, 0.197] | 0.357 [0.338, 0.375] | 0.850 [0.806, 0.889] | 0.958 [0.924, 0.985] |
| dense:e5-small | filtered | 0.190 [0.180, 0.202] | 0.366 [0.351, 0.382] | 0.876 [0.839, 0.910] | 0.965 [0.933, 0.990] |
| hybrid:e5-small | base | 0.190 [0.177, 0.206] | 0.364 [0.343, 0.384] | 0.878 [0.834, 0.914] | 0.963 [0.930, 0.990] |
| hybrid:e5-small | filtered | 0.195 [0.184, 0.209] | 0.381 [0.364, 0.399] | 0.901 [0.866, 0.932] | 0.978 [0.955, 0.995] |
| dense:minilm-l12 | base | 0.180 [0.169, 0.192] | 0.351 [0.331, 0.371] | 0.832 [0.786, 0.873] | 0.966 [0.932, 0.991] |
| dense:minilm-l12 | filtered | 0.187 [0.176, 0.197] | 0.365 [0.347, 0.382] | 0.875 [0.834, 0.911] | 0.977 [0.948, 1.000] |
| hybrid:minilm-l12 | base | 0.185 [0.172, 0.200] | 0.361 [0.342, 0.381] | 0.858 [0.815, 0.896] | 0.966 [0.937, 0.990] |
| hybrid:minilm-l12 | filtered | 0.193 [0.183, 0.206] | 0.378 [0.363, 0.395] | 0.901 [0.868, 0.929] | 0.988 [0.970, 1.000] |

Recall has a ceiling: queries have a median of 51 relevant reviews in their pools, more than ten or twenty results can hold. The best possible Recall@10 is 0.219 and Recall@20 is 0.424, so recall compares the systems; it doesn't measure coverage.

## Paired differences (base variant)

| Comparison | Metric | Difference | 95% interval | Queries | Tie |
|---|---|---|---|---|---|
| hybrid:e5-small - dense:e5-small | NDCG@10 | +0.028 | +0.012 to +0.044 | 100 | no |
| hybrid:e5-small - dense:e5-small | Recall@20 | +0.007 | -0.001 to +0.015 | 100 | yes |
| hybrid:e5-small - bm25 | NDCG@10 | +0.013 | -0.003 to +0.031 | 100 | yes |
| hybrid:e5-small - bm25 | Recall@20 | +0.004 | -0.005 to +0.013 | 100 | yes |
| dense:e5-small - bm25 | NDCG@10 | -0.016 | -0.042 to +0.015 | 100 | yes |
| dense:e5-small - bm25 | Recall@20 | -0.003 | -0.015 to +0.010 | 100 | yes |
| hybrid:minilm-l12 - dense:minilm-l12 | NDCG@10 | +0.027 | +0.009 to +0.046 | 100 | no |
| hybrid:minilm-l12 - dense:minilm-l12 | Recall@20 | +0.009 | +0.000 to +0.018 | 100 | no |
| hybrid:minilm-l12 - bm25 | NDCG@10 | -0.007 | -0.025 to +0.011 | 100 | yes |
| hybrid:minilm-l12 - bm25 | Recall@20 | +0.001 | -0.010 to +0.012 | 100 | yes |
| dense:minilm-l12 - bm25 | NDCG@10 | -0.034 | -0.067 to -0.001 | 100 | no |
| dense:minilm-l12 - bm25 | Recall@20 | -0.009 | -0.026 to +0.009 | 100 | yes |

## By query language and kind (NDCG@10, base variant)

| System | English (68) | French (27) | Cross-lingual (5) | Aspect (76) | Event (19) | All but events (81) |
|---|---|---|---|---|---|---|
| bm25 | 0.873 [0.818, 0.923] | 0.847 [0.752, 0.933] | 0.860 [0.784, 0.939] | 0.913 [0.873, 0.952] | 0.674 [0.531, 0.804] | 0.910 [0.871, 0.946] |
| dense:e5-small | 0.882 [0.838, 0.922] | 0.770 [0.667, 0.865] | 0.838 [0.747, 0.920] | 0.888 [0.848, 0.925] | 0.699 [0.567, 0.817] | 0.885 [0.846, 0.920] |
| hybrid:e5-small | 0.894 [0.849, 0.935] | 0.835 [0.739, 0.918] | 0.889 [0.811, 0.968] | 0.917 [0.882, 0.950] | 0.718 [0.589, 0.834] | 0.915 [0.882, 0.946] |
| dense:minilm-l12 | 0.845 [0.791, 0.893] | 0.792 [0.692, 0.881] | 0.867 [0.734, 0.983] | 0.868 [0.824, 0.908] | 0.677 [0.538, 0.811] | 0.868 [0.826, 0.905] |
| hybrid:minilm-l12 | 0.864 [0.812, 0.911] | 0.840 [0.757, 0.917] | 0.879 [0.818, 0.954] | 0.899 [0.862, 0.935] | 0.689 [0.552, 0.821] | 0.898 [0.862, 0.931] |

## The embedding model

Rules fixed before any result (design D7): the model must pass the speed floor; then the best hybrid NDCG@10 (base variant); a statistical tie with the best (its paired interval contains 0) goes to the lower measured CPU cost.

| Model | Passes the floor | Passages a second (1 vCPU) | Hybrid NDCG@10 | Difference to the best [95%] |
|---|---|---|---|---|
| e5-small (chosen) | yes | 5.7 | 0.878 | - |
| minilm-l12 | yes | 15.8 | 0.858 | +0.020 [+0.001, +0.039] |

Chosen: **e5-small**. e5-small: the best hybrid NDCG@10 (0.878) of the candidates that pass the speed floor.

Without the event queries, whose judgments agree least with Seif, the same rules choose e5-small: the choice holds.

## ANN recall@10 (local Qdrant)

Dense search through the HNSW index against exact search, base filters, every query.

| Model | ANN recall@10 |
|---|---|
| e5-small | 0.991 |
| minilm-l12 | 1.000 |

## Limitations

- Pooled recall: a relevant review that none of the 5 systems ranked in its top 20 is invisible, so recall is relative to the pool.
- Pool depth 20: metrics at 10 are sound; deeper rankings weren't judged.
- The judges are language models, checked against one person's blind audit: the check failed overall, and is weakest on event queries, whose grades hinge on when an update came out (neither the judges nor Seif were shown the date).
