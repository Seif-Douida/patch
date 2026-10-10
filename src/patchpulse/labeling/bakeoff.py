"""The teacher bake-off (plan Task 9, design D4): which LLM labels the silver set.

Each candidate labels the gold dev reviews with the current prompt; labels are cached, so a rerun
costs no requests. They're scored against Claude's gold labels: per-aspect precision, recall and
F1, macro-F1 with a bootstrap interval, and a paired interval for every pair of candidates.
Throughput is what the free tier sustains at `use_fraction`, from the tokens Google counted: a
candidate's reviews a day.

The rule was fixed before the run: the best macro-F1 wins, but a candidate whose paired interval
with the best includes 0 is tied with it, and among tied candidates the one with more reviews a day
wins. The report and the MLflow run hold ids, counts and scores only, never review text.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from numpy.typing import NDArray

from patchpulse.labeling.cache import LabelCache
from patchpulse.labeling.gemini import GeminiResponse
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.labeling.teacher import (
    PROMPT_VERSION,
    ReviewIn,
    TextModel,
    label_reviews,
    prompt_hash,
)
from patchpulse.labeling.teachers import ModelConfig
from patchpulse.models.metrics import (
    IndexMetric,
    bootstrap_ci,
    macro_f1,
    paired_bootstrap_ci,
    prf,
)
from patchpulse.models.text import CLEANING_VERSION
from patchpulse.models.tracking import log_ids_hash, log_report, tracking_run

BAKEOFF_SEED = 20261010
EXPERIMENT = "teacher-bakeoff"
REPORT_PATH = Path("reports/aspects/teacher-bakeoff.md")
_MINUTES_A_DAY = 1440


@dataclass(frozen=True)
class CandidateResult:
    name: str
    model_id: str
    f1: dict[str, float]
    support: dict[str, int]  # gold positives per aspect
    macro_f1: float
    macro_f1_ci: tuple[float, float]
    reviews_per_day: float | None  # None when every label came from the cache
    calls: int
    labeled: int
    missing: int  # dev reviews without a label: scored as "no aspect"
    precision: dict[str, float] = field(default_factory=dict)
    recall: dict[str, float] = field(default_factory=dict)
    stopped_for_quota: bool = False
    stopped_for_outage: bool = False


@dataclass(frozen=True)
class PairedDifference:
    a: str
    b: str
    difference: float  # macro-F1 of a minus b
    low: float
    high: float

    @property
    def tied(self) -> bool:
        return self.low <= 0 <= self.high


@dataclass(frozen=True)
class BakeoffResult:
    n_reviews: int
    candidates: tuple[CandidateResult, ...]
    paired: tuple[PairedDifference, ...]
    prompt_version: str
    prompt_hash: str
    review_ids: tuple[int, ...] = ()


class _Meter:
    """Counts a model's requests and tokens, for the throughput estimate."""

    def __init__(self, model: TextModel) -> None:
        self.model = model
        self.calls = 0
        self.tokens = 0

    def generate(
        self,
        system: str,
        user: str,
        *,
        json_schema: Mapping[str, Any] | None,
        expected_output_tokens: int = 512,
        max_output_tokens: int = 4096,
    ) -> GeminiResponse:
        answer = self.model.generate(
            system,
            user,
            json_schema=json_schema,
            expected_output_tokens=expected_output_tokens,
            max_output_tokens=max_output_tokens,
        )
        self.calls += 1
        self.tokens += answer.input_tokens + answer.output_tokens
        return answer


def _matrix(labels: Sequence[frozenset[str]]) -> NDArray[np.bool_]:
    return np.array([[code in row for code in ASPECTS] for row in labels], dtype=bool)


def _scorer(truth: NDArray[np.bool_], predicted: NDArray[np.bool_]) -> IndexMetric:
    """Macro-F1 on the resampled rows a bootstrap draws."""

    def score(rows: NDArray[np.intp]) -> float:
        return macro_f1(truth[rows], predicted[rows])

    return score


def _reviews_per_day(config: ModelConfig, meter: _Meter, labeled: int) -> float | None:
    if meter.calls == 0:
        return None
    tokens_per_call = meter.tokens / meter.calls
    calls_a_minute = min(config.share(config.rpm), config.share(config.tpm) // tokens_per_call)
    calls_a_day = min(config.share(config.rpd), calls_a_minute * _MINUTES_A_DAY)
    return float(calls_a_day * labeled / meter.calls)


def run_bakeoff(
    candidates: Mapping[str, ModelConfig],
    reviews: Sequence[ReviewIn],
    gold: Mapping[int, frozenset[str]],
    *,
    make_client: Callable[[str, ModelConfig], TextModel],
    cache: LabelCache,
    seed: int = BAKEOFF_SEED,
    samples: int = 2000,
) -> BakeoffResult:
    ids = [review.review_id for review in reviews]
    truth = _matrix([gold[i] for i in ids])
    digest = prompt_hash()
    results: list[CandidateResult] = []
    predictions: dict[str, NDArray[np.bool_]] = {}
    for name, config in candidates.items():
        meter = _Meter(make_client(name, config))
        run = label_reviews(meter, cache, reviews, model=config.id)
        labels = cache.get_many(ids, model=config.id, prompt_hash=digest)
        predicted = _matrix([labels.get(i, frozenset()) for i in ids])
        predictions[name] = predicted
        scores = prf(truth, predicted)
        results.append(
            CandidateResult(
                name=name,
                model_id=config.id,
                f1={c: float(v) for c, v in zip(ASPECTS, scores.f1, strict=True)},
                support={c: int(v) for c, v in zip(ASPECTS, scores.support, strict=True)},
                macro_f1=macro_f1(truth, predicted),
                macro_f1_ci=bootstrap_ci(
                    _scorer(truth, predicted), len(ids), samples=samples, seed=seed
                ),
                reviews_per_day=_reviews_per_day(config, meter, run.labeled),
                calls=meter.calls,
                labeled=len(labels),
                missing=len(ids) - len(labels),
                precision={c: float(v) for c, v in zip(ASPECTS, scores.precision, strict=True)},
                recall={c: float(v) for c, v in zip(ASPECTS, scores.recall, strict=True)},
                stopped_for_quota=run.stopped_for_quota,
                stopped_for_outage=run.stopped_for_outage,
            )
        )
    paired = []
    for first, second in combinations(results, 2):
        low, high = paired_bootstrap_ci(
            _scorer(truth, predictions[first.name]),
            _scorer(truth, predictions[second.name]),
            len(ids),
            samples=samples,
            seed=seed,
        )
        paired.append(
            PairedDifference(first.name, second.name, first.macro_f1 - second.macro_f1, low, high)
        )
    return BakeoffResult(
        n_reviews=len(ids),
        candidates=tuple(results),
        paired=tuple(paired),
        prompt_version=PROMPT_VERSION,
        prompt_hash=digest,
        review_ids=tuple(ids),
    )


def _pair(result: BakeoffResult, a: str, b: str) -> PairedDifference:
    for pair in result.paired:
        if {pair.a, pair.b} == {a, b}:
            return pair
    raise KeyError(f"no paired interval for {a} and {b}")


def choose_teacher(result: BakeoffResult) -> str:
    best = max(result.candidates, key=lambda c: c.macro_f1)
    tied = [c for c in result.candidates if c is best or _pair(result, best.name, c.name).tied]
    return max(tied, key=lambda c: (c.reviews_per_day or 0.0, c.macro_f1)).name


def _per_day(value: float | None) -> str:
    return "n/a (all cached)" if value is None else f"{value:,.0f}"


def render_report(result: BakeoffResult, winner: str) -> str:
    names = [c.name for c in result.candidates]
    lines = [
        f"# Teacher bake-off: gold dev ({result.n_reviews} reviews)",
        "",
        f"Prompt {result.prompt_version} (hash `{result.prompt_hash[:12]}`), cleaning "
        f"{CLEANING_VERSION}. Truth: Claude's gold labels. Macro-F1 averages all "
        f"{len(ASPECTS)} aspects; an aspect with no gold positives in dev scores 0 for every "
        "candidate.",
        "",
        "Rule fixed before the run (plan Task 9): the best macro-F1 wins, unless its paired 95% "
        "interval with another candidate includes 0. Tied candidates are separated by the reviews "
        "a day the free tier sustains at half its limits.",
        "",
        "| Candidate | Model | Macro-F1 (95% CI) | Reviews a day | Requests | Unlabeled |",
        "|---|---|---|---|---|---|",
    ]
    for c in result.candidates:
        low, high = c.macro_f1_ci
        lines.append(
            f"| {c.name} | `{c.model_id}` | {c.macro_f1:.3f} ({low:.3f} to {high:.3f}) | "
            f"{_per_day(c.reviews_per_day)} | {c.calls} | {c.missing} |"
        )
    lines += [
        "",
        "## Per-aspect F1",
        "",
        "| Aspect | Gold positives | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    support = result.candidates[0].support if result.candidates else {}
    for code in ASPECTS:
        cells = " | ".join(f"{c.f1.get(code, 0.0):.2f}" for c in result.candidates)
        lines.append(f"| {code} | {support.get(code, 0)} | {cells} |")
    if result.paired:
        lines += [
            "",
            "## Paired differences in macro-F1",
            "",
            "| Pair | Difference | 95% CI | Verdict |",
            "|---|---|---|---|",
        ]
        for pair in result.paired:
            verdict = "tied" if pair.tied else f"{pair.a if pair.difference > 0 else pair.b} better"
            lines.append(
                f"| {pair.a} vs {pair.b} | {pair.difference:+.3f} | "
                f"{pair.low:+.3f} to {pair.high:+.3f} | {verdict} |"
            )
    lines += ["", f"**Teacher: {winner}**", ""]
    return "\n".join(lines)


def log_bakeoff(result: BakeoffResult, winner: str, report: Path) -> str:
    """One MLflow run in `teacher-bakeoff` with every candidate's scores; returns its run id."""
    params = {
        "prompt_version": result.prompt_version,
        "prompt_hash": result.prompt_hash,
        "cleaning_version": CLEANING_VERSION,
        "candidates": ",".join(c.model_id for c in result.candidates),
    }
    with tracking_run(EXPERIMENT, params=params, tags={"winner": winner}) as run:
        log_ids_hash("dev", result.review_ids)
        for c in result.candidates:
            metrics = {
                f"{c.name}.macro_f1": c.macro_f1,
                f"{c.name}.macro_f1_low": c.macro_f1_ci[0],
                f"{c.name}.macro_f1_high": c.macro_f1_ci[1],
                f"{c.name}.missing": float(c.missing),
            }
            if c.reviews_per_day is not None:
                metrics[f"{c.name}.reviews_per_day"] = c.reviews_per_day
            metrics.update({f"{c.name}.f1.{code}": value for code, value in c.f1.items()})
            mlflow.log_metrics(metrics)
        log_report(report)
        return str(run.info.run_id)
