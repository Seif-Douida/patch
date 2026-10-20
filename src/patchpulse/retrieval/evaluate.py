"""Phase 3's retrieval eval (design D7, plan Task 12): every system's metrics per query, means with
bootstrap intervals over queries, paired differences, the embedding model's choice by the rules
fixed before any result, ANN recall, and the report.

The report holds numbers only: no query text, no review id, no review text.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from patchpulse.models.metrics import bootstrap_ci, paired_bootstrap_ci
from patchpulse.retrieval.agreement import PASS, RelevanceAgreement
from patchpulse.retrieval.benchmark import EmbedderSpeed, passes_floor
from patchpulse.retrieval.metrics import Grades, mrr_at, ndcg_at, recall_at
from patchpulse.retrieval.queries import Query
from patchpulse.retrieval.search import QueryEmbedder, search

EVAL_SEED = 20261019
SAMPLES = 2000
DEPTH = 20
Metric = Callable[[Sequence[int], Grades, int], float | None]
METRICS: dict[str, tuple[Metric, int]] = {
    "Recall@10": (recall_at, 10),
    "Recall@20": (recall_at, 20),
    "NDCG@10": (ndcg_at, 10),
    "MRR@10": (mrr_at, 10),
}
CHOICE_METRIC = "NDCG@10"  # on hybrid search, base variant (design D7)
PAIRED_METRICS = ("NDCG@10", "Recall@20")
Values = list[float | None]  # one per query, in the frozen file's order; None: no relevant review
SystemKey = tuple[str, str]  # (system, variant)

# Query slices reported apart. A cross-lingual query is English text against French reviews.
SLICES: dict[str, Callable[[Query], bool]] = {
    "English": lambda q: q.language == "english" and q.kind != "crosslingual",
    "French": lambda q: q.language == "french" and q.kind != "crosslingual",
    "Cross-lingual": lambda q: q.kind == "crosslingual",
    "Aspect": lambda q: q.kind == "aspect",
    "Event": lambda q: q.kind == "event",
    "All but events": lambda q: q.kind != "event",
}


def score_run(
    run: pd.DataFrame, grades: Mapping[str, Grades], query_ids: Sequence[str]
) -> dict[str, Values]:
    """Each metric for each query of `query_ids`; a query the run never answered ranks nothing."""
    ranked = {
        str(query_id): [int(r) for r in rows.sort_values("rank")["review_id"]]
        for query_id, rows in run.groupby("query_id")
    }
    return {
        name: [metric(ranked.get(q, []), grades.get(q, {}), k) for q in query_ids]
        for name, (metric, k) in METRICS.items()
    }


@dataclass(frozen=True)
class Summary:
    mean: float | None
    low: float | None
    high: float | None
    counted: int
    left_out: int  # queries without a relevant review


def summarize(values: Values, *, seed: int, samples: int = SAMPLES) -> Summary:
    """The mean over queries with a 95% bootstrap interval, resampling queries."""
    scored = np.array([value for value in values if value is not None], dtype=np.float64)
    if scored.size == 0:
        return Summary(None, None, None, 0, len(values))
    low, high = bootstrap_ci(
        lambda ix: float(scored[ix].mean()), scored.size, samples=samples, seed=seed
    )
    return Summary(float(scored.mean()), low, high, int(scored.size), len(values) - scored.size)


@dataclass(frozen=True)
class Difference:
    mean: float
    low: float
    high: float
    n: int  # queries both systems are scored on

    @property
    def tie(self) -> bool:
        return self.low <= 0.0 <= self.high


def paired(a: Values, b: Values, *, seed: int, samples: int = SAMPLES) -> Difference:
    """a - b over the queries both are scored on, with a paired bootstrap interval."""
    both = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not both:
        raise ValueError("no query is scored for both systems")
    x = np.array([pair[0] for pair in both], dtype=np.float64)
    y = np.array([pair[1] for pair in both], dtype=np.float64)
    low, high = paired_bootstrap_ci(
        lambda ix: float(x[ix].mean()),
        lambda ix: float(y[ix].mean()),
        len(both),
        samples=samples,
        seed=seed,
    )
    return Difference(float((x - y).mean()), low, high, len(both))


@dataclass(frozen=True)
class Choice:
    name: str
    reason: str
    table: list[dict[str, Any]]  # one row per candidate, JSON-ready

    def to_json(self) -> dict[str, Any]:
        return {"chosen": self.name, "reason": self.reason, "candidates": self.table}


def _or_lowest(value: float | None) -> float:
    return -1.0 if value is None else value


def choose_embedder(
    ndcg: Mapping[str, Values], speeds: Mapping[str, EmbedderSpeed], *, seed: int
) -> Choice:
    """Design D7's rules, fixed before any result: (1) the candidate must pass the D3 speed floor;
    (2) then the best hybrid NDCG@10; (3) a statistical tie with the best (the paired interval
    contains 0) goes to the model with the lower measured CPU cost (more passages a second)."""
    passing = [name for name in ndcg if name in speeds and passes_floor(speeds[name])]
    if not passing:
        raise ValueError("no candidate passes the speed floor")
    means = {name: summarize(values, seed=seed).mean for name, values in ndcg.items()}
    best = max(passing, key=lambda name: _or_lowest(means[name]))
    against = {name: paired(ndcg[best], ndcg[name], seed=seed) for name in passing if name != best}
    tied = [best, *(name for name, difference in against.items() if difference.tie)]
    chosen = max(tied, key=lambda name: speeds[name].passages_per_s)
    if chosen == best:
        reason = (
            f"{best}: the best hybrid NDCG@10 ({means[best]:.3f}) of the candidates that pass "
            "the speed floor"
            + ("" if len(tied) == 1 else ", and the cheapest of those it ties with")
        )
    else:
        gap = against[chosen]
        reason = (
            f"{chosen}: a tie with {best} on hybrid NDCG@10 (difference {gap.mean:+.3f}, 95% "
            f"interval {gap.low:+.3f} to {gap.high:+.3f}) goes to the lower CPU cost "
            f"({speeds[chosen].passages_per_s:.1f} passages a second against "
            f"{speeds[best].passages_per_s:.1f})"
        )
    table = []
    for name in ndcg:
        difference = against.get(name)
        table.append(
            {
                "name": name,
                "passes_floor": bool(name in speeds and passes_floor(speeds[name])),
                "passages_per_s": speeds[name].passages_per_s if name in speeds else None,
                "hybrid_ndcg10": means[name],
                "difference_to_best": (
                    None
                    if difference is None
                    else [difference.mean, difference.low, difference.high]
                ),
                "chosen": name == chosen,
            }
        )
    return Choice(chosen, reason, table)


def _top_ids(
    client: Any, collection: str, embedder: QueryEmbedder, query: Query, k: int, *, exact: bool
) -> set[int]:
    hits = search(
        client,
        collection,
        embedder,
        query.text,
        appid=query.appid,
        language=query.language,
        filters=query.base_filters,
        mode="dense",
        k=k,
        exact=exact,
    )
    return {hit.review_id for hit in hits}


def ann_recall(
    client: Any, collection: str, embedder: QueryEmbedder, queries: Sequence[Query], k: int = 10
) -> float:
    """How much of the exact dense top k the HNSW search finds, averaged over queries."""
    overlaps = []
    for query in queries:
        found = _top_ids(client, collection, embedder, query, k, exact=False)
        exact = _top_ids(client, collection, embedder, query, k, exact=True)
        if exact:
            overlaps.append(len(found & exact) / len(exact))
    return float(np.mean(overlaps)) if overlaps else 0.0


@dataclass(frozen=True)
class Comparison:
    better: str  # system expected ahead (hybrid over dense over BM25)
    worse: str
    metric: str
    difference: Difference


@dataclass(frozen=True)
class Evaluation:
    queries: list[Query]
    per_query: dict[SystemKey, dict[str, Values]]
    summaries: dict[SystemKey, dict[str, Summary]]
    comparisons: list[Comparison]
    slices: dict[str, tuple[int, dict[str, Summary]]]  # slice -> (queries, system -> NDCG@10)
    pool_pairs: int
    no_relevant: int  # queries without a relevant review in the pool
    # The best Recall@k any ranking could reach: a query with more relevant reviews than k caps it.
    recall_ceiling: dict[int, float | None]
    relevant_per_query: float | None  # the median, over queries with any


def _subset(values: Values, mask: Sequence[bool]) -> Values:
    return [value for value, keep in zip(values, mask, strict=True) if keep]


def _candidates(keys: Sequence[SystemKey]) -> list[str]:
    names: list[str] = []
    for system, _ in keys:
        name = system.partition(":")[2]
        if name and name not in names:
            names.append(name)
    return names


def evaluate_runs(
    runs: Mapping[SystemKey, pd.DataFrame],
    grades: Mapping[str, Grades],
    queries: Sequence[Query],
    *,
    seed: int = EVAL_SEED,
) -> Evaluation:
    query_ids = [query.query_id for query in queries]
    per_query = {key: score_run(run, grades, query_ids) for key, run in runs.items()}
    summaries = {
        key: {metric: summarize(values, seed=seed) for metric, values in metrics.items()}
        for key, metrics in per_query.items()
    }
    comparisons = []
    for name in _candidates(list(runs)):
        for better, worse in (
            (f"hybrid:{name}", f"dense:{name}"),
            (f"hybrid:{name}", "bm25"),
            (f"dense:{name}", "bm25"),
        ):
            for metric in PAIRED_METRICS:
                difference = paired(
                    per_query[(better, "base")][metric],
                    per_query[(worse, "base")][metric],
                    seed=seed,
                )
                comparisons.append(Comparison(better, worse, metric, difference))
    slices = {}
    for label, belongs in SLICES.items():
        mask = [belongs(query) for query in queries]
        if not any(mask):
            continue
        slices[label] = (
            sum(mask),
            {
                system: summarize(
                    _subset(per_query[(system, variant)][CHOICE_METRIC], mask), seed=seed
                )
                for system, variant in runs
                if variant == "base"
            },
        )
    any_values = next(iter(per_query.values()))[CHOICE_METRIC] if per_query else []
    relevant = [
        n for n in (sum(g >= 1 for g in grades.get(q, {}).values()) for q in query_ids) if n > 0
    ]
    ceiling = {
        k: float(np.mean([min(k, n) / n for n in relevant])) if relevant else None
        for k in sorted({k for metric, k in METRICS.values() if metric is recall_at})
    }
    return Evaluation(
        queries=list(queries),
        per_query=per_query,
        summaries=summaries,
        comparisons=comparisons,
        slices=slices,
        pool_pairs=sum(len(judged) for judged in grades.values()),
        no_relevant=sum(value is None for value in any_values),
        recall_ceiling=ceiling,
        relevant_per_query=float(np.median(relevant)) if relevant else None,
    )


@dataclass(frozen=True)
class Report:
    evaluation: Evaluation
    choice: Choice
    choice_without_events: Choice
    ann_recall: Mapping[str, float]
    agreement: RelevanceAgreement  # the gate's pairs, as in `retrieval agreement`
    agreement_by_kind: Mapping[str, tuple[int, float | None]]
    judged_by: Mapping[str, int]  # judge model -> pairs
    guidelines: Mapping[str, int]  # guidelines version -> pairs
    accepted_failed_agreement: bool
    unread: int = 0  # audit pairs left out of the gate: text Seif couldn't read
    notes: list[str] = field(default_factory=list)


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _cell(summary: Summary) -> str:
    if summary.mean is None:
        return "n/a"
    return f"{summary.mean:.3f} [{summary.low:.3f}, {summary.high:.3f}]"


def render_report(report: Report) -> str:
    ev = report.evaluation
    kinds = pd.Series([q.kind for q in ev.queries]).value_counts()
    languages = pd.Series([q.language for q in ev.queries]).value_counts()
    systems = list(dict.fromkeys(system for system, _ in ev.summaries))
    agreement = report.agreement
    lines = [
        "# Phase 3 retrieval eval",
        "",
        f"{len(ev.queries)} frozen queries ("
        + ", ".join(f"{n} {kind}" for kind, n in kinds.items())
        + "; "
        + ", ".join(f"{n} in {language}" for language, n in languages.items())
        + f"). Each system's top {DEPTH} per query, base and filtered, pooled into "
        f"{ev.pool_pairs:,} judged pairs; a result nobody judged counts as not relevant. "
        "Relevant means grade 1 or more (0 not relevant, 1 partly, 2 relevant). Means over "
        f"queries with 95% bootstrap intervals ({SAMPLES:,} resamples of the queries). "
        f"{ev.no_relevant} queries have no relevant review in their pool: left out of the means, "
        "never scored.",
        "",
        "## The judgments and their agreement with Seif",
        "",
        "Judged by "
        + ", ".join(f"{model} ({n:,} pairs)" for model, n in report.judged_by.items())
        + "; guidelines "
        + ", ".join(f"{version} ({n:,} pairs)" for version, n in sorted(report.guidelines.items()))
        + ".",
        "",
        f"Quadratic-weighted kappa against Seif's blind audit: {_num(agreement.kappa)} on the "
        f"{agreement.n} audit pairs he read (rule fixed in advance: at least {PASS:.2f}): "
        + ("**passed**." if agreement.passed else "**failed**.")
        + (
            f" {report.unread} audit pairs in languages he couldn't read are left out."
            if report.unread
            else ""
        ),
        "",
        "| Query kind | Audit pairs | Weighted kappa |",
        "|---|---|---|",
        *(
            f"| {kind} | {n} | {_num(kappa)} |"
            for kind, (n, kappa) in report.agreement_by_kind.items()
        ),
        "",
    ]
    if not agreement.passed and report.accepted_failed_agreement:
        lines += [
            "The judgments failed the agreement rule and were accepted anyway "
            "(`--accept-failed-agreement`): Seif chose not to grade further after two failed "
            "checks (reports/retrieval/judge-agreement-v2.md, then -v3.md, which is optimistic). "
            "Every number below rests on these judgments, the event queries' least of all.",
            "",
        ]
    lines += [
        "## Every system",
        "",
        "| System | Variant | " + " | ".join(METRICS) + " |",
        "|---|---|" + "---|" * len(METRICS),
    ]
    for (system, variant), metrics in ev.summaries.items():
        lines.append(
            f"| {system} | {variant} | " + " | ".join(_cell(metrics[m]) for m in METRICS) + " |"
        )
    if ev.relevant_per_query is not None:
        lines += [
            "",
            "Recall has a ceiling: queries have a median of "
            f"{ev.relevant_per_query:.0f} relevant reviews in their pools, more than ten or twenty "
            "results can hold. The best possible "
            + " and ".join(f"Recall@{k} is {_num(value)}" for k, value in ev.recall_ceiling.items())
            + ", so recall compares the systems; it doesn't measure coverage.",
        ]
    lines += [
        "",
        "## Paired differences (base variant)",
        "",
        "| Comparison | Metric | Difference | 95% interval | Queries | Tie |",
        "|---|---|---|---|---|---|",
    ]
    for c in ev.comparisons:
        d = c.difference
        lines.append(
            f"| {c.better} - {c.worse} | {c.metric} | {d.mean:+.3f} | {d.low:+.3f} to "
            f"{d.high:+.3f} | {d.n} | {'yes' if d.tie else 'no'} |"
        )
    labels = list(ev.slices)
    lines += [
        "",
        "## By query language and kind (NDCG@10, base variant)",
        "",
        "| System | " + " | ".join(f"{label} ({ev.slices[label][0]})" for label in labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    for system in systems:
        lines.append(
            f"| {system} | "
            + " | ".join(_cell(ev.slices[label][1][system]) for label in labels)
            + " |"
        )
    lines += [
        "",
        "## The embedding model",
        "",
        "Rules fixed before any result (design D7): the model must pass the speed floor; then the "
        "best hybrid NDCG@10 (base variant); a statistical tie with the best (its paired interval "
        "contains 0) goes to the lower measured CPU cost.",
        "",
        "| Model | Passes the floor | Passages a second (1 vCPU) | Hybrid NDCG@10 | "
        "Difference to the best [95%] |",
        "|---|---|---|---|---|",
    ]
    for row in report.choice.table:
        gap = row["difference_to_best"]
        lines.append(
            f"| {row['name']}{' (chosen)' if row['chosen'] else ''} | "
            f"{'yes' if row['passes_floor'] else 'no'} | {row['passages_per_s']:.1f} | "
            f"{_num(row['hybrid_ndcg10'])} | "
            + ("-" if gap is None else f"{gap[0]:+.3f} [{gap[1]:+.3f}, {gap[2]:+.3f}]")
            + " |"
        )
    same = report.choice_without_events.name == report.choice.name
    lines += [
        "",
        f"Chosen: **{report.choice.name}**. {report.choice.reason}.",
        "",
        "Without the event queries, whose judgments agree least with Seif, the same rules choose "
        f"{report.choice_without_events.name}"
        + (": the choice holds." if same else ": **the choice changes**.")
        + "",
        "",
        "## ANN recall@10 (local Qdrant)",
        "",
        "Dense search through the HNSW index against exact search, base filters, every query.",
        "",
        "| Model | ANN recall@10 |",
        "|---|---|",
        *(f"| {name} | {value:.3f} |" for name, value in report.ann_recall.items()),
        "",
        "## Limitations",
        "",
        f"- Pooled recall: a relevant review that none of the {len(systems)} systems ranked in "
        f"its top {DEPTH} is invisible, so recall is relative to the pool.",
        f"- Pool depth {DEPTH}: metrics at 10 are sound; deeper rankings weren't judged.",
        "- The judges are language models, checked against one person's blind audit: the check "
        "failed overall, and is weakest on event queries, whose grades hinge on when an update "
        "came out (neither the judges nor Seif were shown the date)."
        if not agreement.passed
        else "- The judges are language models, checked against one person's blind audit.",
        *(f"- {note}" for note in report.notes),
    ]
    return "\n".join(lines) + "\n"
