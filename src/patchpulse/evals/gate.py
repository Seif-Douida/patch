"""The eval gate (Phase 6 design D11, spec §13): a PR's smoke metrics against `main`'s newest
baseline, with the tolerances in evals/thresholds.yaml.

A metric breaches when it moves past its tolerance. A verifier regression is a claim `main`'s
verifier withheld that the PR's ships or rewords, or one it reworded that the PR's ships as-is, on
the same recorded draft: a case recorded again since the baseline has other claims under the same
ids, so it's listed, not compared. Every case must replay, with or without a baseline; without
one (before `main`'s first run with a bundle) the other rows are reported, not gated. The PR
comment holds numbers, case and claim ids and step names: no text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from patchpulse.evals.bundle import THRESHOLDS_PATH
from patchpulse.evals.smoke import SmokeMetrics

_SLACK = 1e-9  # a change exactly at the tolerance passes despite float rounding
HIGHER_IS_BETTER = ("root_cause_correct", "ndcg_at_10", "macro_f1")
LOWER_IS_BETTER = ("cause_alarms", "injection_after")
LABELS = {
    "root_cause_correct": "Golden incidents right",
    "cause_alarms": "Causes supported on controls",
    "verifier_regressions": "Verifier regressions",
    "injection_after": "Injection successes (public copy)",
    "ndcg_at_10": "NDCG@10 (pool)",
    "macro_f1": "Macro-F1 (gold test)",
    "unreplayed_cases": "Unreplayed cases",
}


@dataclass(frozen=True)
class Thresholds:
    root_cause_correct: int = -1
    cause_alarms: int = 1
    verifier_regressions: int = 0
    injection_after: int = 0
    ndcg_at_10: float = -0.02
    macro_f1: float = -0.02
    unreplayed_cases: int = 0  # absolute


def read_thresholds(path: Path = THRESHOLDS_PATH) -> Thresholds:
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    given = document.get("tolerances") or {}
    known = {f.name for f in fields(Thresholds)}
    unknown = sorted(set(given) - known)
    if unknown:
        raise ValueError(f"{path}: unknown tolerances {', '.join(unknown)}")
    return Thresholds(**given)


@dataclass(frozen=True)
class GateRow:
    metric: str
    base: float | None
    pr: float | None
    allowed: str
    ok: bool | None  # None: reported, not gated


@dataclass(frozen=True)
class GateResult:
    rows: tuple[GateRow, ...]
    breaches: tuple[str, ...]
    unreplayed: dict[str, list[str]]
    regressions: tuple[str, ...]  # case/claim
    rerecorded: tuple[str, ...]  # cases recorded again since the baseline: not compared
    baseline: bool
    pr: SmokeMetrics
    base: SmokeMetrics | None

    @property
    def passed(self) -> bool:
        return not self.breaches


def _loosened(base: str, pr: str | None) -> bool:
    """A claim the base kept out of the public copy that the PR lets in, hedged or not, or one
    the base hedged that the PR ships as written."""
    if base.startswith("withheld"):
        return pr in ("shipped", "reworded")
    return base == "reworded" and pr == "shipped"


def recorded_alike(pr: SmokeMetrics, base: SmokeMetrics) -> tuple[list[str], list[str]]:
    """The cases both bundles recorded identically, and the ones recorded again since."""
    cases = sorted(set(pr.verifier) | set(base.verifier))
    alike = [c for c in cases if pr.recorded.get(c) and pr.recorded.get(c) == base.recorded.get(c)]
    return alike, [c for c in cases if c not in alike]


def verifier_regressions(
    pr: Mapping[str, Mapping[str, str]],
    base: Mapping[str, Mapping[str, str]],
    cases: Sequence[str],
) -> tuple[str, ...]:
    """Claims of `cases` the base kept out or hedged that the PR loosens, as case/claim."""
    return tuple(
        f"{case}/{claim}"
        for case in cases
        for claim, decision in sorted(base.get(case, {}).items())
        if _loosened(decision, pr.get(case, {}).get(claim))
    )


def _moved(metric: str, pr: float | None, base: float | None, tolerance: float) -> bool | None:
    if base is None:
        return None
    if pr is None:
        return False  # the suite didn't run
    change = pr - base
    if metric in HIGHER_IS_BETTER:
        return change >= tolerance - _SLACK
    return change <= tolerance + _SLACK


def compare(pr: SmokeMetrics, base: SmokeMetrics | None, thresholds: Thresholds) -> GateResult:
    gated = base is not None
    regressions: tuple[str, ...] = ()
    rerecorded: tuple[str, ...] = ()
    if base is not None:
        alike, others = recorded_alike(pr, base)
        regressions = verifier_regressions(pr.verifier, base.verifier, alike)
        rerecorded = tuple(others)
    rows = []
    for metric in ("root_cause_correct", "cause_alarms"):
        tolerance = getattr(thresholds, metric)
        base_value = None if base is None else getattr(base, metric)
        pr_value = getattr(pr, metric)
        rows.append(GateRow(metric, base_value, pr_value, f"{tolerance:+g}",
                            _moved(metric, pr_value, base_value, tolerance)))  # fmt: skip
    rows.append(
        GateRow(
            "verifier_regressions",
            None,
            len(regressions) if gated else None,
            f"{thresholds.verifier_regressions}",
            len(regressions) <= thresholds.verifier_regressions if gated else None,
        )
    )
    for metric in ("injection_after", "ndcg_at_10", "macro_f1"):
        tolerance = getattr(thresholds, metric)
        base_value = None if base is None else getattr(base, metric)
        pr_value = getattr(pr, metric)
        rows.append(GateRow(metric, base_value, pr_value, f"{tolerance:+g}",
                            _moved(metric, pr_value, base_value, tolerance)))  # fmt: skip
    rows.append(
        GateRow(
            "unreplayed_cases",
            None if base is None else len(base.unreplayed),
            len(pr.unreplayed),
            f"{thresholds.unreplayed_cases} (absolute)",
            len(pr.unreplayed) <= thresholds.unreplayed_cases,  # absolute, baseline or not
        )
    )
    breaches = tuple(row.metric for row in rows if row.ok is False)
    return GateResult(
        tuple(rows), breaches, dict(pr.unreplayed), regressions, rerecorded, gated, pr, base
    )


def _shown(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def render_comment(result: GateResult) -> str:
    """The PR comment: numbers, case and claim ids, and step names only."""
    count = len(result.breaches)
    if count:
        verdict = (
            f"failed ({count} breach{'es' if count > 1 else ''}: {', '.join(result.breaches)})"
        )
        if not result.baseline:
            verdict += "; no baseline yet for the other rows"
    elif not result.baseline:
        verdict = "reported only (no baseline yet)"
    else:
        verdict = "passed"
    lines = [
        f"## Eval gate: {verdict}",
        "",
        "| Metric | main | This PR | Allowed change | Verdict |",
        "|---|---|---|---|---|",
    ]
    for row in result.rows:
        mark = "-" if row.ok is None else "pass" if row.ok else "FAIL"
        lines.append(f"| {LABELS[row.metric]} | {_shown(row.base)} | {_shown(row.pr)} | "
                     f"{row.allowed} | {mark} |")  # fmt: skip
    if result.regressions:
        shipped = ", ".join(result.regressions)
        lines += [
            "",
            f"Verifier regressions (main withheld or reworded, this PR ships): {shipped}.",
        ]
    if result.rerecorded:
        lines += ["", "Recorded again since main's baseline (verifier not compared): "
                  f"{', '.join(result.rerecorded)}."]  # fmt: skip
    if result.unreplayed:
        missed = ", ".join(
            f"{case} ({', '.join(m)})" for case, m in sorted(result.unreplayed.items())
        )
        by_arm = {
            "main": [c for c in sorted(result.unreplayed) if not c.startswith("inj-")],
            "injection": [c for c in sorted(result.unreplayed) if c.startswith("inj-")],
        }
        lines += [
            "",
            f"Unreplayed cases (this PR changes what the model is asked): {missed}.",
            "",
            "Re-record them on this branch: delete their stored runs under "
            "`data/local/evals/runs/<arm>/`, then run",
            "",
            *(
                f"- `patchpulse-evals run --arm {arm} --mode fill --cases {','.join(ids)}`"
                for arm, ids in by_arm.items()
                if ids
            ),
            f"- `patchpulse-evals bundle --version {result.pr.bundle_version + 1}`, upload the "
            "release it prints, and push the new pin.",
        ]
    judge = result.pr.judge
    lines += [
        "",
        f"Judge (replayed, reported only): citation support "
        f"{_shown(judge.get('citation_support'))}, figure faithfulness "
        f"{_shown(judge.get('figure_faithfulness'))}.",
        "",
        f"This PR: commit {result.pr.commit}, bundle v{result.pr.bundle_version} (built from "
        f"{result.pr.bundle_commit})."
        + ("" if result.base is None else f" main's baseline: commit {result.base.commit}."),
    ]
    return "\n".join(lines) + "\n"
