"""Phase 2's evaluation (plan Task 14, design D6), in two steps and in this order.

1. **The champion, from gold dev only** (`choose_champion`), by rules fixed before any results:
   - eligible: at least 70 reviews a second on one CPU (`benchmark_cpu`, in a 1-CPU, 2 GiB
     container), so a 20k-review night scores in under 5 minutes;
   - then the best out-of-fold dev macro-F1 (each review judged by thresholds tuned on the other
     folds);
   - a statistical tie with a cheaper eligible model (the paired 95% interval of the difference
     contains 0) goes to the cheaper one: S0, then S1a, then S1b.
2. **The report, from gold test, once** (`evaluate_models`, `render_eval_report`): the teacher
   and every student on gold test (Claude's labels) and on Seif's 150 (his labels), per aspect and
   language, with calibration and bootstrap intervals; and each student's agreement with the
   teacher on the silver holdout. Nothing from test feeds back into the choice.

The report holds numbers only, never review text (C4).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.metrics import (
    IndexMetric,
    bootstrap_ci,
    brier_score,
    expected_calibration_error,
    macro_f1,
    micro_f1,
    paired_bootstrap_ci,
    prf,
)

MIN_REVIEWS_PER_S = 70.0
COST_ORDER: tuple[str, ...] = ("s0", "s1a", "s1b")  # cheapest first: a tie goes to the earlier
SAMPLES = 2_000
TEACHER = "teacher"
LANGUAGES = ("english", "french")
BENCH_IMAGE = "python:3.12-slim"
SRC_DIR = Path(__file__).resolve().parents[2]  # the directory holding the `patchpulse` package

Run = Callable[..., "subprocess.CompletedProcess[str]"]


def _scorer(truth: NDArray[np.bool_], predicted: NDArray[np.bool_]) -> IndexMetric:
    def score(rows: NDArray[np.intp]) -> float:
        return macro_f1(truth[rows], predicted[rows])

    return score


# --- 1. The champion, from dev ---------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    family: str
    version: str
    reviews_per_s: float  # in the 1-CPU container
    dev_predicted: NDArray[np.bool_]  # out-of-fold decisions on gold dev


@dataclass(frozen=True)
class Comparison:
    """Dev macro-F1 of the best eligible model minus a cheaper one's, with its paired interval."""

    best: str
    other: str
    difference: float
    low: float
    high: float

    @property
    def tied(self) -> bool:
        return self.low <= 0.0 <= self.high


@dataclass(frozen=True)
class Choice:
    champion: str | None
    best: str | None  # the best dev macro-F1 among the eligible
    eligible: tuple[str, ...]
    reviews_per_s: dict[str, float]
    dev_macro_f1: dict[str, float]
    versions: dict[str, str]
    comparisons: tuple[Comparison, ...]
    reason: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Choice:
        raw: dict[str, Any] = json.loads(text)
        return cls(
            champion=raw["champion"],
            best=raw["best"],
            eligible=tuple(raw["eligible"]),
            reviews_per_s={k: float(v) for k, v in raw["reviews_per_s"].items()},
            dev_macro_f1={k: float(v) for k, v in raw["dev_macro_f1"].items()},
            versions=dict(raw["versions"]),
            comparisons=tuple(Comparison(**c) for c in raw["comparisons"]),
            reason=raw["reason"],
        )


def _cost(family: str) -> int:
    if family not in COST_ORDER:
        raise ValueError(f"unknown student family {family!r}")
    return COST_ORDER.index(family)


def choose_champion(
    candidates: Sequence[Candidate],
    dev_truth: NDArray[np.bool_],
    *,
    min_reviews_per_s: float = MIN_REVIEWS_PER_S,
    seed: int,
    samples: int = SAMPLES,
) -> Choice:
    truth = np.asarray(dev_truth, dtype=bool)
    ordered = sorted(candidates, key=lambda c: _cost(c.family))
    macro = {c.family: macro_f1(truth, c.dev_predicted) for c in ordered}
    rates = {c.family: c.reviews_per_s for c in ordered}
    versions = {c.family: c.version for c in ordered}
    eligible = [c for c in ordered if c.reviews_per_s >= min_reviews_per_s]
    if not eligible:
        return Choice(
            None,
            None,
            (),
            rates,
            macro,
            versions,
            (),
            f"no model reaches {min_reviews_per_s:g} reviews a second on one CPU",
        )
    best = max(eligible, key=lambda c: macro[c.family])  # equal scores: the cheaper comes first
    comparisons = []
    for other in eligible:
        if _cost(other.family) < _cost(best.family):
            low, high = paired_bootstrap_ci(
                _scorer(truth, best.dev_predicted),
                _scorer(truth, other.dev_predicted),
                truth.shape[0],
                samples=samples,
                seed=seed,
            )
            difference = macro[best.family] - macro[other.family]
            comparisons.append(Comparison(best.family, other.family, difference, low, high))
    tied = [c for c in comparisons if c.tied]  # in cost order: the first is the cheapest
    if tied:
        champion = tied[0].other
        reason = (
            f"{best.family} has the best dev macro-F1, but {champion} is statistically tied with "
            f"it (paired 95% interval {tied[0].low:+.3f} to {tied[0].high:+.3f}) and cheaper"
        )
    else:
        champion = best.family
        reason = f"{best.family} has the best dev macro-F1 of the models fast enough"
        if comparisons:
            reason += ", and every cheaper one is clearly behind"
    return Choice(
        champion,
        best.family,
        tuple(c.family for c in eligible),
        rates,
        macro,
        versions,
        tuple(comparisons),
        reason,
    )


# --- CPU throughput ---------------------------------------------------------------------------


def scoring_requirements() -> str:
    """The `scoring` group's pinned packages, as the jobs image would install them."""
    command = [
        "uv",
        "export",
        "--frozen",
        "--only-group",
        "scoring",
        "--no-hashes",
        "--no-emit-project",
        "--no-header",
    ]
    exported = subprocess.run(  # noqa: S603 - fixed executable, arguments from code
        command, capture_output=True, text=True, check=True
    )
    return exported.stdout


def _host(path: Path) -> str:
    return path.resolve().as_posix()


def benchmark_cpu(
    model_dirs: Sequence[Path],
    texts: Sequence[str],
    *,
    image: str = BENCH_IMAGE,
    run: Run = subprocess.run,
    requirements: str | None = None,
) -> dict[str, float]:
    """Reviews a second for each model, scored inside `docker run --cpus=1 --memory=2g`.

    One container installs the `scoring` group once and times every model in turn
    (`patchpulse.models.benchmark`). Everything is mounted read-only; the texts stay local.
    """
    pinned = requirements if requirements is not None else scoring_requirements()
    with tempfile.TemporaryDirectory(prefix="pp-bench-") as work_dir:
        work = Path(work_dir)
        (work / "texts.json").write_text(json.dumps(list(texts)), encoding="utf-8")
        (work / "requirements.txt").write_text(pinned, encoding="utf-8")
        mounts = ["-v", f"{_host(SRC_DIR)}:/app/src:ro", "-v", f"{_host(work)}:/work:ro"]
        targets = []
        for index, model_dir in enumerate(model_dirs):
            target = f"/models/{index}"
            mounts += ["-v", f"{_host(model_dir)}:{target}:ro"]
            targets.append(target)
        script = (
            "pip install --quiet --no-cache-dir --disable-pip-version-check "
            "--root-user-action=ignore -r /work/requirements.txt && PYTHONPATH=/app/src "
            f"python -m patchpulse.models.benchmark /work/texts.json {' '.join(targets)}"
        )
        command = [
            "docker",
            "run",
            "--rm",
            "--cpus=1",
            "--cpuset-cpus=0",
            "--memory=2g",
            *mounts,
            image,
            "sh",
            "-c",
            script,
        ]
        result = run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"the benchmark container failed (exit {result.returncode}): "
            f"{result.stderr.strip()[-2000:]}"
        )
    rates: dict[str, float] = {}
    for line in result.stdout.splitlines():
        family, tab, rate = line.partition("\t")
        if tab and family in COST_ORDER:
            rates[family] = float(rate)
    return rates


# --- 2. The report, from gold test -----------------------------------------------------------


@dataclass(frozen=True)
class GoldSet:
    name: str
    ids: list[int]
    truth: NDArray[np.bool_]  # (reviews, aspects)
    languages: list[str]


@dataclass(frozen=True)
class Outputs:
    """One model's decisions on one review set, and its probabilities if it has any."""

    labels: NDArray[np.bool_]
    probs: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class Scored:
    set_name: str
    model: str
    n: int
    macro_f1: float
    macro_f1_ci: tuple[float, float]
    micro_f1: float
    precision: dict[str, float]
    recall: dict[str, float]
    f1: dict[str, float]
    support: dict[str, int]
    by_language: dict[str, float]  # macro-F1 on each language's reviews
    language_n: dict[str, int]
    ece: float | None  # only for models with probabilities
    brier: float | None


@dataclass(frozen=True)
class Paired:
    set_name: str
    a: str
    b: str
    difference: float  # macro-F1 of a minus b, on the same reviews
    low: float
    high: float


@dataclass(frozen=True)
class EvalResult:
    scores: tuple[Scored, ...]
    paired: tuple[Paired, ...]
    agreement: dict[str, float]  # macro-F1 against the teacher's labels on the silver holdout
    agreement_micro: dict[str, float]
    holdout_n: int

    def score(self, set_name: str, model: str) -> Scored:
        return next(s for s in self.scores if s.set_name == set_name and s.model == model)

    @property
    def models(self) -> list[str]:
        return list(dict.fromkeys(s.model for s in self.scores))

    @property
    def sets(self) -> list[str]:
        return list(dict.fromkeys(s.set_name for s in self.scores))


def _score(gold: GoldSet, model: str, out: Outputs, *, seed: int, samples: int) -> Scored:
    truth, labels = gold.truth, np.asarray(out.labels, dtype=bool)
    if labels.shape != truth.shape:
        raise ValueError(f"{model} on {gold.name}: labels {labels.shape} vs truth {truth.shape}")
    per = prf(truth, labels)
    languages = np.array(gold.languages)
    by_language, language_n = {}, {}
    for language in LANGUAGES:
        rows = languages == language
        if rows.any():
            by_language[language] = macro_f1(truth[rows], labels[rows])
            language_n[language] = int(rows.sum())
    return Scored(
        set_name=gold.name,
        model=model,
        n=truth.shape[0],
        macro_f1=macro_f1(truth, labels),
        macro_f1_ci=bootstrap_ci(
            _scorer(truth, labels), truth.shape[0], samples=samples, seed=seed
        ),
        micro_f1=micro_f1(truth, labels),
        precision=dict(zip(ASPECTS, map(float, per.precision), strict=True)),
        recall=dict(zip(ASPECTS, map(float, per.recall), strict=True)),
        f1=dict(zip(ASPECTS, map(float, per.f1), strict=True)),
        support=dict(zip(ASPECTS, map(int, per.support), strict=True)),
        by_language=by_language,
        language_n=language_n,
        ece=None if out.probs is None else expected_calibration_error(truth, out.probs),
        brier=None if out.probs is None else brier_score(truth, out.probs),
    )


def evaluate_models(
    sets: Sequence[GoldSet],
    outputs: Mapping[str, Mapping[str, Outputs]],
    *,
    holdout_teacher: NDArray[np.bool_],
    holdout_students: Mapping[str, NDArray[np.bool_]],
    seed: int,
    samples: int = SAMPLES,
) -> EvalResult:
    """`outputs[model][set name]` for every model (the teacher first) on every set."""
    scores = []
    paired = []
    for gold in sets:
        for model, by_set in outputs.items():
            scores.append(_score(gold, model, by_set[gold.name], seed=seed, samples=samples))
        for a, b in combinations(outputs, 2):
            first, second = outputs[a][gold.name].labels, outputs[b][gold.name].labels
            low, high = paired_bootstrap_ci(
                _scorer(gold.truth, first),
                _scorer(gold.truth, second),
                gold.truth.shape[0],
                samples=samples,
                seed=seed,
            )
            difference = macro_f1(gold.truth, first) - macro_f1(gold.truth, second)
            paired.append(Paired(gold.name, a, b, difference, low, high))
    teacher = np.asarray(holdout_teacher, dtype=bool)
    return EvalResult(
        scores=tuple(scores),
        paired=tuple(paired),
        agreement={m: macro_f1(teacher, labels) for m, labels in holdout_students.items()},
        agreement_micro={m: micro_f1(teacher, labels) for m, labels in holdout_students.items()},
        holdout_n=teacher.shape[0],
    )


@dataclass(frozen=True)
class ModelFacts:
    family: str
    version: str
    size_mb: float
    reviews_per_s: float
    dev_cv_macro_f1: float


def _f(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def render_eval_report(result: EvalResult, choice: Choice, facts: Mapping[str, ModelFacts]) -> str:
    lines = [
        "# Phase 2 evaluation: aspect students against the teacher",
        "",
        "Plan Task 14, design D6. The champion was chosen on gold dev before gold test was read; "
        "gold test was then used once, for this report. Numbers only: no review text.",
        "",
        f"## Champion: {choice.champion or 'none'}",
        "",
        f"{choice.reason[0].upper()}{choice.reason[1:]}.",
        "",
        "Rules fixed before any results: at least 70 reviews a second on one CPU; then the best "
        "dev macro-F1, with each dev review judged by thresholds tuned on the other folds; a "
        "statistical tie with a cheaper model (paired 95% interval containing 0) goes to the "
        "cheaper one (S0, then S1a, then S1b).",
        "",
    ]
    rows = [
        [
            family,
            f"`{choice.versions.get(family, '')}`",
            f"{choice.reviews_per_s[family]:,.1f}",
            f"{choice.dev_macro_f1[family]:.3f}",
            "yes" if family in choice.eligible else "no",
        ]
        for family in choice.dev_macro_f1
    ]
    header = ["Student", "Version", "Reviews a second (1 CPU)", "Dev macro-F1", "Eligible"]
    lines += _table(header, rows)
    if choice.comparisons:
        lines += ["", "Paired dev differences (best minus cheaper):", ""]
        lines += _table(
            ["Comparison", "Difference (95% interval)", "Tied"],
            [
                [
                    f"{c.best} - {c.other}",
                    f"{c.difference:+.3f} ({c.low:+.3f} to {c.high:+.3f})",
                    "yes" if c.tied else "no",
                ]
                for c in choice.comparisons
            ],
        )
    titles = {
        "test": "Gold test (Claude's labels, audited by Seif)",
        "audit": "Seif's 150 (his own labels; part of gold test)",
    }
    for set_name in result.sets:
        first = result.score(set_name, result.models[0])
        lines += [
            "",
            f"## {titles.get(set_name, set_name)}, {first.n} reviews",
            "",
        ]
        language_header = [
            f"{lang.title()} (n={first.language_n[lang]})" for lang in first.language_n
        ]
        rows = []
        for model in result.models:
            s = result.score(set_name, model)
            low, high = s.macro_f1_ci
            rows.append(
                [
                    model,
                    f"{s.macro_f1:.3f} ({low:.3f} to {high:.3f})",
                    f"{s.micro_f1:.3f}",
                    *(f"{s.by_language[lang]:.3f}" for lang in first.language_n),
                    _f(s.ece),
                    _f(s.brier),
                ]
            )
        header = ["Model", "Macro-F1 (95% CI)", "Micro-F1", *language_header, "ECE", "Brier"]
        lines += _table(header, rows)
        lines += ["", "F1 per aspect:", ""]
        rows = [
            [
                code,
                str(first.support[code]),
                *(f"{result.score(set_name, m).f1[code]:.3f}" for m in result.models),
            ]
            for code in ASPECTS
        ]
        lines += _table(["Aspect", "Positives", *result.models], rows)
        lines += ["", "Paired differences in macro-F1:", ""]
        rows = [
            [
                f"{p.a} - {p.b}",
                f"{p.difference:+.3f} ({p.low:+.3f} to {p.high:+.3f})",
                "yes" if p.low <= 0 <= p.high else "no",
            ]
            for p in result.paired
            if p.set_name == set_name
        ]
        lines += _table(["Comparison", "Difference (95% CI)", "Tied"], rows)
    lines += [
        "",
        f"## Agreement with the teacher (silver holdout, {result.holdout_n:,} reviews)",
        "",
        "Every language the silver set holds; this is the measure Phase 7 tracks online.",
        "",
    ]
    lines += _table(
        ["Student", "Macro-F1 against the teacher", "Micro-F1"],
        [
            [m, f"{result.agreement[m]:.3f}", f"{result.agreement_micro[m]:.3f}"]
            for m in result.agreement
        ],
    )
    lines += ["", "## Size and speed", ""]
    lines += _table(
        ["Student", "Version", "Size (MB)", "Reviews a second (1 CPU)", "Dev macro-F1"],
        [
            [
                f.family,
                f"`{f.version}`",
                f"{f.size_mb:.1f}",
                f"{f.reviews_per_s:,.1f}",
                f"{f.dev_cv_macro_f1:.3f}",
            ]
            for f in facts.values()
        ],
    )
    lines += [
        "",
        "## Method and limitations",
        "",
        "- The teacher is Gemma 4 31B on a self-hosted server (ADR-020): it labeled the 30,000 "
        "silver reviews the students learned from. Its scores here have no thresholds or "
        "probabilities.",
        "- Each student's thresholds were tuned on gold dev; gold test played no part in any "
        "choice.",
        "- Per-language quality is measured for English and French only. Other languages are "
        "covered by agreement with the teacher on the silver holdout.",
        "- Gold test labels are Claude's, audited blind by Seif on 150 reviews (mean kappa 0.99). "
        "Seif's 150 are scored against his own labels.",
        "- Macro-F1 weighs all 10 aspects equally. online_servers and platform_policy have few "
        "positives in gold, so their F1 moves a lot with a review or two.",
        "- Students are trained with balanced class weights, so their probabilities lean high "
        "for rare aspects: ECE and Brier describe that, the thresholds correct for it.",
        "- Intervals are 95% percentile bootstraps over reviews (2,000 resamples); paired "
        "intervals resample both models on the same reviews.",
    ]
    return "\n".join(lines) + "\n"
