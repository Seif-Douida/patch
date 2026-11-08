"""The eval report (Phase 6 design D9): `reports/evals/<date>.md` and `.json`, and an MLflow run in
the `agent-evals` experiment.

The renderer reads EvalResults only: numbers, case ids, categories, rule ids and model ids. It
never sees a stored case, so no review, claim, summary or judge text can reach a committed file or
MLflow by construction (spec §4, C4).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow

from patchpulse.agent.report import Report
from patchpulse.evals.incidents import Incident
from patchpulse.evals.judge import citation_share, claim_support, judge_metrics
from patchpulse.evals.metrics import (
    Estimate,
    a1,
    confusion,
    golden_metrics,
    paired_difference,
    per_category,
)
from patchpulse.evals.records import StoredCase
from patchpulse.evals.scoring import claim_stats, patch_blame, score_case
from patchpulse.models.tracking import log_report, tracking_run

REPORTS_DIR = Path("reports") / "evals"
EXPERIMENT = "agent-evals"
OTHER_REPORTS = (
    ("Classification (Phase 2)", "reports/aspects/phase-2-eval.md"),
    ("Retrieval (Phase 3)", "reports/retrieval/phase-3-eval.md"),
    ("Causal benchmark (Phase 4)", "reports/causal/benchmark.md"),
)
HEADLINE = (
    ("root_cause_accuracy", "Root-cause accuracy"),
    ("lenient_accuracy", "Root cause, a secondary cause counted"),
    ("cause_alarm_rate", "False alarms on controls (a cause supported)"),
    ("change_alarm_rate", "Change alarms on controls (H5 ruled out)"),
    ("window_iou", "Window overlap (IoU)"),
    ("window_hit_rate", "Windows overlapping at 0.5 or more"),
    ("direction_accuracy", "Direction of the change"),
    ("decided_share", "Explanations decided (not inconclusive)"),
    ("pre_verifier_pass_rate", "Drafted claims the verifier passed"),
    ("reworded", "Shipped claims reworded by V4"),
    ("citation_support", "Cited reviews supporting their claim (judge)"),
    ("figure_faithfulness", "Cited figures described correctly (judge)"),
    ("tool_error_rate", "Model tool calls with an error"),
    ("tool_repeat_rate", "Model tool calls repeating an earlier one"),
    ("patch_blame_before", "Non-patch incidents whose draft blames a patch"),
    ("patch_blame_after", "Non-patch incidents whose report blames a patch"),
    ("model_calls", "Model calls per run"),
    ("tokens_in", "Input tokens per run"),
    ("tokens_out", "Output tokens per run"),
    ("seconds_median", "Seconds per run (median)"),
    ("seconds_p95", "Seconds per run (p95)"),
)
COUNTS = ("root_cause_correct", "cause_alarms", "refusals")
SHARE_KEYS = frozenset(
    {"reworded", "pre_verifier_pass_rate", "citation_support", "figure_faithfulness"}
)


@dataclass(frozen=True)
class RunInfo:
    date: str  # YYYY-MM-DD
    data_version: int
    scoring_version: str
    prompt_version: str
    rubric_version: str | None
    chains: Mapping[str, tuple[str, ...]]  # arm -> its single-model chain
    judge_chain: tuple[str, ...] | None
    commit: str


@dataclass(frozen=True)
class CaseRow:
    incident_id: str
    expected: str
    predicted: str
    supported: tuple[str, ...]  # the rule ids of the supported explanations
    refused: bool
    indexed_reviews: int | None


@dataclass(frozen=True)
class EvalResults:
    info: RunInfo
    cases: tuple[CaseRow, ...]  # the main arm's
    arms: Mapping[str, Mapping[str, Estimate]]
    per_category: Mapping[str, tuple[int, int]]
    confusion: Mapping[str, int]
    ablations: Mapping[str, Mapping[str, Estimate]]  # arm -> metric -> arm minus main
    a1: Mapping[str, Estimate] | None = None
    injection: Mapping[str, Estimate] | None = None
    judge: Mapping[str, Estimate] | None = None  # the judge's agreement with Seif
    ci_suites: Mapping[str, float] | None = None


Run = tuple[Incident, StoredCase]


def _correct(run: Run) -> float:
    return float(score_case(run[0], run[1].record()).correct)


def _pass_rate(run: Run) -> float | None:
    stats = claim_stats(run[1].record())
    return None if stats is None or stats.drafted == 0 else stats.shipped / stats.drafted


def _blame(run: Run) -> float | None:
    if run[0].primary_cause == "patch":
        return None
    blame = patch_blame(run[1].record())
    return None if blame is None else float(blame[0] > 0)


def _tokens(run: Run) -> float | None:
    report = run[1].report
    return None if report is None else float(sum(u.input_tokens + u.output_tokens
                                                 for u in report.usage))  # fmt: skip


ABLATION_METRICS: dict[str, Callable[[Run], float | None]] = {
    "root_cause_accuracy": _correct,
    "pre_verifier_pass_rate": _pass_rate,
    "patch_blame_before": _blame,
    "tokens": _tokens,
}


def build_results(
    incidents: Sequence[Incident],
    arms: Mapping[str, Mapping[str, StoredCase]],
    info: RunInfo,
    *,
    verdicts: Mapping[str, bool] | None = None,
    injection: Mapping[str, Estimate] | None = None,
    judge: Mapping[str, Estimate] | None = None,
    ci_suites: Mapping[str, float] | None = None,
) -> EvalResults:
    """Everything the report shows, from the stored runs; the main arm must be there.
    `verdicts` are the judge's, by item id: they add citation support and figure
    faithfulness to each arm, A1, and the ablations' citation-support difference."""
    if "main" not in arms:
        raise ValueError("no stored main run: run `patchpulse-evals run --arm main` first")
    main = arms["main"]
    by_id = {i.id: i for i in incidents}
    scores = [score_case(i, main[i.id].record()) for i in incidents if i.id in main]
    rows = []
    for score in scores:
        stored = main[score.incident_id]
        report = stored.report
        supported = (
            tuple(h.rule for h in report.hypotheses if h.status == "supported")
            if isinstance(report, Report)
            else ()
        )
        rows.append(
            CaseRow(
                incident_id=score.incident_id,
                expected=score.expected,
                predicted=score.predicted,
                supported=supported,
                refused=score.predicted == "refused",
                indexed_reviews=stored.indexed_reviews,
            )
        )

    def runs(stored: Mapping[str, StoredCase]) -> dict[str, Run]:
        return {key: (by_id[key], case) for key, case in stored.items() if key in by_id}

    metric_fns = dict(ABLATION_METRICS)
    if verdicts is not None:
        judged = verdicts
        metric_fns["citation_support"] = lambda run: citation_share(run[1], judged)
    ablations = {
        name: {
            metric: paired_difference(runs(main), runs(stored), fn)
            for metric, fn in metric_fns.items()
        }
        for name, stored in sorted(arms.items())
        if name not in ("main", "injection")
    }

    def arm_metrics(stored: Mapping[str, StoredCase]) -> dict[str, Estimate]:
        metrics = golden_metrics(incidents, stored)
        if verdicts is not None:
            metrics |= judge_metrics({k: v for k, v in stored.items() if k in by_id}, verdicts)
        return metrics

    support = (
        None
        if verdicts is None
        else {case_id: claim_support(case, verdicts) for case_id, case in main.items()}
    )
    return EvalResults(
        info=info,
        cases=tuple(rows),
        arms={
            name: arm_metrics(stored)
            for name, stored in sorted(arms.items())
            if name != "injection"
        },
        per_category=per_category(scores),
        confusion=confusion(scores),
        ablations=ablations,
        a1=None if support is None else a1(main, support),
        injection=injection,
        judge=judge,
        ci_suites=ci_suites,
    )


def _is_share(key: str) -> bool:
    """Shares show as percentages and their differences in points; the rest as numbers."""
    return (
        key.endswith(("_rate", "_accuracy", "_share"))
        or key.startswith(("withheld.", "patch_blame", "unsupported_"))
        or key in SHARE_KEYS
    )


def _value(key: str, value: float | None) -> str:
    if value is None:
        return "-"
    if _is_share(key):
        return f"{100 * value:.1f}%"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def _difference(key: str, value: float | None) -> str:
    if value is None:
        return "-"
    if _is_share(key):
        return f"{100 * value:+.1f} pts"
    return f"{value:+,.0f}" if float(value).is_integer() else f"{value:+,.2f}"


def _interval(key: str, estimate: Estimate, show: Callable[[str, float | None], str]) -> str:
    if estimate.lo is None or estimate.hi is None:
        return "-"
    return f"{show(key, estimate.lo)} to {show(key, estimate.hi)}"


def _estimate_rows(
    estimates: Mapping[str, Estimate], names: Sequence[tuple[str, str]]
) -> list[str]:
    lines = ["| Metric | Value | 95% interval | n |", "|---|---|---|---|"]
    for key, label in names:
        estimate = estimates.get(key)
        if estimate is not None:
            lines.append(f"| {label} | {_value(key, estimate.value)} | "
                         f"{_interval(key, estimate, _value)} | {estimate.n} |")  # fmt: skip
    return lines


def render_report(results: EvalResults) -> str:
    info = results.info
    main = results.arms.get("main", {})
    chains = "; ".join(f"{arm}: {', '.join(chain)}" for arm, chain in sorted(info.chains.items()))
    decided = (
        _value("decided_share", main["decided_share"].value) if "decided_share" in main else "-"
    )
    lines = [
        f"# Agent evaluation, {info.date}",
        "",
        "The agent answered each golden incident's question as a visitor would, on one night's "
        "frozen data. Numbers only: no review, claim or summary text leaves the machine.",
        "",
        "## The run",
        "",
        f"Data version {info.data_version}, scoring {info.scoring_version}, prompt "
        f"{info.prompt_version}, rubric {info.rubric_version or 'none yet'}, commit "
        f"{info.commit}.",
        f"Models: {chains}. Judge: {', '.join(info.judge_chain or ()) or 'not run yet'}.",
        "",
        "## Headline (main arm)",
        "",
        *_estimate_rows(main, HEADLINE),
        "",
        "Counts: "
        + ", ".join(
            f"{key.replace('_', ' ')} {int(main[key].value or 0)} of {main[key].n}"
            for key in COUNTS
            if key in main
        )
        + ".",
        "",
        "With 32 incidents, an accuracy's 95% interval spans about 17 points either side, so "
        "compare arms with the paired differences below, not the headline intervals.",
        "",
        "Hypothesis coverage: 5 of 5 by construction (the rules set every explanation's "
        f"status); the decided share is {decided}.",
        "",
        "## By category",
        "",
        "| Expected | Correct | Cases |",
        "|---|---|---|",
        *(f"| {cat} | {c} | {n} |" for cat, (c, n) in results.per_category.items()),
        "",
        "## Confusion",
        "",
        "| Expected -> predicted | Cases |",
        "|---|---|",
        *(f"| {pair} | {n} |" for pair, n in results.confusion.items()),
        "",
        "## Cases",
        "",
        "| Incident | Expected | Predicted | Supported (rule) | Indexed reviews |",
        "|---|---|---|---|---|",
        *(
            f"| {r.incident_id} | {r.expected} | {r.predicted} | "
            f"{', '.join(r.supported) or '-'} | "
            f"{'-' if r.indexed_reviews is None else r.indexed_reviews} |"
            for r in results.cases
        ),
        "",
    ]
    empty = [r.incident_id for r in results.cases if r.indexed_reviews == 0]
    lines.append(
        f"No indexed reviews in the scoped window: {', '.join(empty)}."
        if empty
        else "Every scoped window had indexed reviews, where counted."
    )
    lines += ["", "## Ablations (each arm minus the main arm, paired over incidents)", ""]
    if results.ablations:
        lines += ["| Arm | Metric | Difference | 95% interval | n |", "|---|---|---|---|---|"]
        for arm, diffs in results.ablations.items():
            for metric, estimate in diffs.items():
                shown = _difference(metric, estimate.value)
                interval = _interval(metric, estimate, _difference)
                lines.append(f"| {arm} | {metric} | {shown} | {interval} | {estimate.n} |")
    else:
        lines.append("No ablation arm stored yet.")
    lines += ["", "## A1: unsupported claims with the verifier off and on", ""]
    lines += (
        _estimate_rows(
            results.a1, (("unsupported_off", "Verifier off"), ("unsupported_on", "Verifier on"))
        )
        if results.a1
        else ["Needs the judge's verdicts: not run yet."]
    )
    lines += ["", "## Injection set", ""]
    lines += (
        _estimate_rows(results.injection, [(k, k) for k in results.injection])
        if results.injection
        else ["Not run yet."]
    )
    lines += ["", "## Judge agreement with Seif", ""]
    lines += (
        _estimate_rows(results.judge, [(k, k) for k in results.judge])
        if results.judge
        else ["Not run yet."]
    )
    lines += ["", "## Other evaluations", ""]
    lines += [f"- {label}: [{path}](../../{path})" for label, path in OTHER_REPORTS]
    if results.ci_suites:
        lines += ["", "CI's suites (they guard changes; the retrieval pool is the judged reviews "
                  "only, so these aren't Phase 3's full-index numbers):", ""]  # fmt: skip
        lines += [f"- {name}: {value:.3f}" for name, value in results.ci_suites.items()]
    return "\n".join(lines) + "\n"


def _plain(value: Any) -> Any:
    if isinstance(value, Estimate):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return {name: _plain(getattr(value, name)) for name in value.__dataclass_fields__}
    return value


def results_json(results: EvalResults) -> dict[str, Any]:
    plain: dict[str, Any] = _plain(results)
    return plain


def write_report(results: EvalResults, directory: Path = REPORTS_DIR) -> tuple[Path, Path]:
    """Writes `<date>.md` and `<date>.json`; returns their paths."""
    directory.mkdir(parents=True, exist_ok=True)
    markdown = directory / f"{results.info.date}.md"
    document = directory / f"{results.info.date}.json"
    markdown.write_text(render_report(results), encoding="utf-8", newline="\n")
    document.write_text(
        json.dumps(results_json(results), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return markdown, document


def log_eval_run(results: EvalResults, report_path: Path) -> str:
    """One MLflow run: the run info as params, the main arm's estimates and bounds as metrics,
    the report as the artifact. Returns the run id."""
    info = results.info
    params = {
        "date": info.date,
        "data_version": info.data_version,
        "scoring_version": info.scoring_version,
        "prompt_version": info.prompt_version,
        "rubric_version": info.rubric_version or "none",
        "commit": info.commit,
        "judge_chain": ",".join(info.judge_chain or ()) or "none",
        **{f"chain.{arm}": ",".join(chain) for arm, chain in sorted(info.chains.items())},
    }
    metrics: dict[str, float] = {}
    for key, estimate in results.arms.get("main", {}).items():
        for suffix, value in (("", estimate.value), ("_lo", estimate.lo), ("_hi", estimate.hi)):
            if value is not None:
                metrics[f"{key}{suffix}"] = float(value)
    with tracking_run(EXPERIMENT, params=params, run_name=f"evals-{info.date}") as run:
        mlflow.log_metrics(metrics)
        log_report(report_path)
        return str(run.info.run_id)
