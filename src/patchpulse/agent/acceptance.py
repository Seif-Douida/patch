"""The acceptance run (Phase 5 plan Task 21, design D15; Seif: "decided by experiment").

Ten patch bundles from the night's published bundle, by a fixed rule: the 5 largest and then the
5 smallest |effect| among the served estimates of the positive share of all reviews, each at its
longest horizon with an effect (ties by bundle id). Each gets a templated question naming its
first patch, asked on its game as a visitor would, every run traced; nothing claims a true cause
(Claude never invents incident facts). Then the first pick's nightly report with the router's
chain emptied: the basic-report path.

The coverage table counts each explanation's statuses and the rules that set them, the verifier
rules that fired, V4's rewordings and blocks by cause and the basic report, and lists the paths no
run took. A path that never appears gets bundles chosen from the data to hit it (`extra`), and the
run repeats.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

from patchpulse.agent.bundle import Bundle, GameData
from patchpulse.agent.gate import BLOCKERS
from patchpulse.agent.graph import AgentDeps, Refusal, ReportRequest, run_report
from patchpulse.agent.report import Report, public_copy
from patchpulse.agent.router import ModelRouter
from patchpulse.agent.tools import ModelCall, ToolCall
from patchpulse.agent.tracing import Observer, Tracer
from patchpulse.agent.verifier import HYPOTHESIS_TITLES, NO_ESTIMATE_CITED, CausalCheck
from patchpulse.causal.its import POSITIVE_SHARE
from patchpulse.causal.store import StoredEstimate, serve

log = logging.getLogger("patchpulse.agent")

PICK_EACH = 5
QUESTION = 'What changed after "{title}" ({day})?'
BASIC = "basic"
STATUSES = {"supported": "supported", "rejected": "ruled out", "inconclusive": "inconclusive"}
RULES = ("V1", "V2", "V3", "V4", "V5", "V6", "V7")
RULE_COUNTS = {
    "V1": "claims withheld: evidence the analysis never produced",
    "V2": "claims withheld: a number that didn't match the data",
    "V3": "claims withheld: a quote that didn't match its review",
    "V4": "causal claims reworded or withheld (below)",
    "V5": "claims withheld: a cause implied without an estimate",
    "V6": "model summaries replaced by the template",
    "V7": "claims withheld: a review's own words repeated",
}
CAUSES = (*BLOCKERS, NO_ESTIMATE_CITED)


@dataclass(frozen=True)
class Pick:
    """A patch bundle to ask about: its served estimate (None without one) and first patch."""

    appid: int
    game: str
    bundle_id: str
    horizon: int | None
    effect: float | None
    title: str
    day: date

    @property
    def question(self) -> str:
        return QUESTION.format(title=self.title, day=f"{self.day.day} {self.day:%b %Y}")


@dataclass(frozen=True)
class AcceptanceRun:
    label: str  # "1".."N", or BASIC
    pick: Pick
    request: ReportRequest
    result: Report | Refusal | None  # None when the run raised
    error: str | None
    causal_checks: tuple[CausalCheck, ...]
    trace_id: str | None


def _served(game: GameData, bundle_id: str, method: str) -> StoredEstimate | None:
    """The served positive-share estimate of all reviews at the longest horizon with an effect."""
    rows = [
        e
        for e in game.estimates
        if (e.bundle_id, e.outcome, e.sample) == (bundle_id, POSITIVE_SHARE, "all")
        and e.method in {method, "its"}
    ]
    estimated = [r for r in serve(rows, method, fallback=True) if r.effect is not None]
    return max(estimated, key=lambda r: r.horizon, default=None)


def _bundles(bundle: Bundle) -> dict[str, Pick]:
    found: dict[str, Pick] = {}
    for game in bundle.games:
        for patch in sorted(game.patches, key=lambda p: (p.published_at, p.patch_id)):
            if patch.bundle_id is None or patch.bundle_id in found:
                continue
            row = _served(game, patch.bundle_id, bundle.causal.default_method)
            found[patch.bundle_id] = Pick(
                appid=game.appid,
                game=game.name,
                bundle_id=patch.bundle_id,
                horizon=None if row is None else row.horizon,
                effect=None if row is None else row.effect,
                title=patch.title,
                day=patch.day,
            )
    return found


def pick_bundles(bundle: Bundle, *, each: int = PICK_EACH, extra: Sequence[str] = ()) -> list[Pick]:
    """The `each` largest |effect|, then the `each` smallest, then `extra` bundle ids in order
    (skipping any already picked; they needn't have an estimate)."""
    known = _bundles(bundle)
    estimated = [p for p in known.values() if p.effect is not None]
    largest = sorted(estimated, key=lambda p: (-abs(p.effect or 0.0), p.bundle_id))[:each]
    rest = [p for p in estimated if p not in largest]
    picks = largest + sorted(rest, key=lambda p: (abs(p.effect or 0.0), p.bundle_id))[:each]
    for bundle_id in extra:
        if bundle_id not in known:
            raise ValueError(f"bundle {bundle_id} isn't a patch bundle in data version "
                             f"{bundle.data_version}")  # fmt: skip
        if all(p.bundle_id != bundle_id for p in picks):
            picks.append(known[bundle_id])
    return picks


class _Recorder(Observer):
    """Passes every event on to the run's own observer and keeps what the coverage needs."""

    def __init__(self, inner: Observer) -> None:
        self.inner = inner
        self.causal_checks: tuple[CausalCheck, ...] = ()

    @property
    def trace_id(self) -> str | None:
        return self.inner.trace_id

    def step_started(self, name: str) -> None:
        self.inner.step_started(name)

    def step_ended(self, name: str, update: Mapping[str, Any]) -> None:
        if name == "verify" and "verified" in update:
            self.causal_checks = update["verified"].causal_checks
        self.inner.step_ended(name, update)

    def tool_started(self, tool: str, args: Mapping[str, Any]) -> None:
        self.inner.tool_started(tool, args)

    def tool_ended(self, call: ToolCall, retrieved: Mapping[int, str]) -> None:
        self.inner.tool_ended(call, retrieved)

    def model_started(self, step: str) -> None:
        self.inner.model_started(step)

    def model_ended(self, step: str, call: ModelCall | None) -> None:
        self.inner.model_ended(step, call)

    def finish(self, result: Report | Refusal | None, error: BaseException | None = None) -> None:
        self.inner.finish(result, error)


class _Recording:
    """A tracer around the run's own; the caller closes that one."""

    def __init__(self, inner: Tracer) -> None:
        self.inner = inner
        self.last: _Recorder | None = None

    def start(self, request: ReportRequest, *, data_version: int) -> Observer:
        self.last = _Recorder(self.inner.start(request, data_version=data_version))
        return self.last

    def close(self) -> None:
        return None


def _run(label: str, pick: Pick, request: ReportRequest, deps: AgentDeps) -> AcceptanceRun:
    recording = _Recording(deps.tracer)
    result: Report | Refusal | None = None
    error: str | None = None
    try:
        result = run_report(request, replace(deps, tracer=recording))
    except Exception as failure:  # one run's failure is a row of the table, not the end
        error = type(failure).__name__
        log.warning("acceptance run %s (%s) failed: %s", label, pick.bundle_id, error)
    recorder = recording.last
    checks = () if recorder is None else recorder.causal_checks
    trace_id = None if recorder is None else recorder.trace_id
    return AcceptanceRun(label, pick, request, result, error, checks, trace_id)


def run_acceptance(deps: AgentDeps, picks: Sequence[Pick]) -> list[AcceptanceRun]:
    """Each pick's question on its game, then the first estimated pick's nightly report with the
    router's chain emptied."""
    runs = [
        _run(str(n), pick, ReportRequest.for_question(pick.question, appid=pick.appid), deps)
        for n, pick in enumerate(picks, start=1)
    ]
    first = next((p for p in picks if p.horizon is not None), None)
    if first is not None and first.horizon is not None:
        router = deps.router
        empty = ModelRouter([], {}, router.cache, clock=router.clock)
        request = ReportRequest.for_bundle(first.bundle_id, first.horizon)
        runs.append(_run(BASIC, first, request, replace(deps, router=empty)))
    return runs


# The coverage table.


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def _row(*cells: object) -> str:
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def _counted(counts: Mapping[str, int]) -> str:
    return ", ".join(r if n == 1 else f"{r} x{n}" for r, n in sorted(counts.items()))


def _run_row(run: AcceptanceRun) -> str:
    pick, result = run.pick, run.result
    effect = "none" if pick.effect is None else f"{pick.effect:+.1f}"
    asked = (
        pick.question
        if run.label != BASIC
        else f"the nightly report, {pick.horizon} days, the chain emptied"
    )
    trace = run.trace_id or "not traced"
    if isinstance(result, Report):
        withheld = Counter(w.rule for w in result.withheld_claims)
        held = f"{sum(withheld.values())} ({_counted(withheld)})" if withheld else "0"
        report = result.report_id + (" (basic)" if result.basic else "")
        return _row(run.label, pick.game, pick.bundle_id, effect, asked, report,
                    result.confidence, len(result.claims), held, trace)  # fmt: skip
    outcome = f"refused: {result.reason}" if isinstance(result, Refusal) else f"error: {run.error}"
    return _row(run.label, pick.game, pick.bundle_id, effect, asked, outcome, "", "", "", trace)


def coverage_text(data_version: int, runs: Sequence[AcceptanceRun]) -> str:
    asked = [r for r in runs if r.label != BASIC]
    forced = next((r for r in runs if r.label == BASIC), None)
    reports = [r.result for r in asked if isinstance(r.result, Report)]
    checks = [c for r in asked for c in r.causal_checks]
    lines = [
        f"# Acceptance run: data version {data_version}",
        "",
        f"{len(asked)} questions asked on the published bundle, each traced; "
        f"{len(reports)} answered with a report.",
        "",
        "## Runs",
        "",
        "| Run | Game | Bundle | Effect (points) | Question | Report | Confidence | Claims | "
        "Withheld | Trace |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *(_run_row(r) for r in runs),
    ]
    never: list[str] = []

    lines += [
        "",
        "## Explanations",
        "",
        "| Explanation | Supported | Ruled out | Inconclusive | Rules |",
        "|---|---|---|---|---|",
    ]
    for code, title in HYPOTHESIS_TITLES.items():
        found = [h for report in reports for h in report.hypotheses if h.code == code]
        statuses: Counter[str] = Counter(h.status for h in found)
        rules = Counter(h.rule for h in found)
        ranked = sorted(rules.items(), key=lambda kv: (-kv[1], kv[0]))
        lines.append(_row(f"{code} {title}", *(statuses[s] for s in STATUSES),
                          ", ".join(f"{r} {n}" for r, n in ranked)))  # fmt: skip
        never += [f"{code} {label}" for s, label in STATUSES.items() if not statuses[s]]

    withheld = Counter(w.rule for report in reports for w in report.withheld_claims)
    fired = {rule: withheld[rule] for rule in RULES}
    fired["V4"] = sum(1 for c in checks if c.outcome != "allowed")
    fired["V6"] = sum(1 for report in reports if report.rule_pass_rates.get("V6") == 0.0)
    lines += ["", "## The verifier", "", "| Rule | Fired | What it counts |", "|---|---|---|"]
    lines += [_row(rule, fired[rule], RULE_COUNTS[rule]) for rule in RULES]
    never += [rule for rule in RULES if not fired[rule]]

    allowed = sum(1 for c in checks if c.outcome == "allowed")
    lines += ["", "## V4: causal claims", "", f"Allowed by the gate: {allowed}.", ""]
    lines += ["| Blocked by | Reworded | Withheld |", "|---|---|---|"]
    for cause in CAUSES:
        outcomes = Counter(c.outcome for c in checks if c.blocked_by == cause)
        lines.append(_row(cause, outcomes["reworded"], outcomes["withheld"]))
        if not outcomes["reworded"] and not outcomes["withheld"]:
            never.append(f"V4 {cause}")
    if not allowed:
        never.append("V4 allowed")

    lines += ["", "## The basic report", ""]
    if forced is None:
        lines.append("Not run: no picked bundle has an estimate.")
        never.append("basic report")
    elif isinstance(forced.result, Report) and forced.result.basic:
        lines.append(f"Forced with the router's chain emptied: a basic report "
                     f"({forced.result.report_id}).")  # fmt: skip
    else:
        lines.append("Forced with the router's chain emptied: no basic report (see its row).")
        never.append("basic report")
    lines.append(f"Basic reports among the questions: {sum(r.basic for r in reports)}.")

    lines += ["", "## Never seen", ""]
    lines += [f"- {path}" for path in never] or ["Every path appeared."]
    return "\n".join(lines) + "\n"


def write_acceptance(out_dir: Path, data_version: int, runs: Sequence[AcceptanceRun]) -> str:
    """The coverage table (coverage.md) and each report's public copy (reports/<id>.json, no
    review text) under `out_dir`; returns the table."""
    text = coverage_text(data_version, runs)
    copies = out_dir / "reports"
    copies.mkdir(parents=True, exist_ok=True)
    (out_dir / "coverage.md").write_text(text, encoding="utf-8", newline="\n")
    for run in runs:
        if isinstance(run.result, Report):
            copy = json.dumps(public_copy(run.result), indent=2, ensure_ascii=False, default=str)
            (copies / f"{run.result.report_id}.json").write_text(
                copy + "\n", encoding="utf-8", newline="\n"
            )
    return text
