"""The agent's workflow (design D4, spec §10.1): a bounded LangGraph state graph.

    scope -> core_evidence -> rules -> gather -> draft -> verify -> summarize -> finalize

- scope: a nightly request names its bundle and horizon (no model call); a visitor's question is
  placed by one model call, then checked: a tracked game, a window inside its data, a known patch.
  What can't be placed is a Refusal in plain words, at most the scope call spent.
- core_evidence and rules: the same tool calls for every report, then H1-H5 from fixed rules.
- gather: one model call plans up to MAX_MODEL_TOOL_CALLS review searches and patch-notes reads.
- draft: the model's findings and claims; verify: the verifier's rules; summarize: a second model
  call from the verified claims only, checked by V6 and V7, the template when it fails.
- finalize: the report with its confidence and the rules' pass rates.

Every model call goes through the router. With every model down before the draft, the report is
basic: fixed claims from the core evidence and the template summary. A run uses one bundle (the
deps hold it), so every call and V2's re-execution see the same data version. The deps' tracer
hears of each step, tool call and model call (agent/tracing.py: counts and ids, never text).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from patchpulse.agent.bundle import Bundle
from patchpulse.agent.evidence import (
    BASELINE_DAYS,
    HORIZONS,
    Scope,
    bundle_scope,
    core_evidence,
    days_in,
)
from patchpulse.agent.prompts import PROMPT_VERSION, SCHEMAS, load_prompt
from patchpulse.agent.report import (
    DateWindow,
    Draft,
    Finding,
    Hypothesis,
    ModelUse,
    Report,
    confidence_for,
    renumbered,
)
from patchpulse.agent.router import AllModelsDownError, ModelRouter
from patchpulse.agent.rules import CoreEvidence, HypothesisResult, decide
from patchpulse.agent.search import ReviewSearch
from patchpulse.agent.tools import ModelCall, ToolCall, Tools, Trace
from patchpulse.agent.tracing import NoTracer, Observer, Tracer
from patchpulse.agent.untrusted import wrap_patch_notes, wrap_question, wrap_reviews
from patchpulse.agent.verifier import (
    HYPOTHESIS_TITLES,
    Verified,
    basic_claims,
    safe_findings,
    template_summary,
    verify_claims,
    verify_summary,
)
from patchpulse.ingest.http import SteamHttp

log = logging.getLogger("patchpulse.agent")

MAX_MODEL_TOOL_CALLS = 8  # spec §10.1
MIN_BASELINE_DAYS = 7
MAX_QUESTION_CHARS = 500
MAX_WINDOW_DAYS = 92
MAX_REFUSAL_CHARS = 300
UNAVAILABLE = "The analyst is unavailable right now; please try again in a few minutes."


@dataclass(frozen=True)
class ReportRequest:
    """A nightly report (a bundle and a horizon) or a visitor's question, with the game the
    visitor picked when there is one."""

    bundle_id: str | None = None
    horizon: int | None = None
    question: str | None = None
    appid: int | None = None

    @classmethod
    def for_bundle(cls, bundle_id: str, horizon: int) -> ReportRequest:
        if horizon not in HORIZONS:
            raise ValueError(f"a report's horizon is 7 or 14 days, not {horizon}")
        return cls(bundle_id=bundle_id, horizon=horizon)

    @classmethod
    def for_question(cls, question: str, *, appid: int | None = None) -> ReportRequest:
        return cls(question=question, appid=appid)


@dataclass(frozen=True)
class Refusal:
    reason: str


@dataclass(frozen=True)
class AgentDeps:
    bundle: Bundle
    router: ModelRouter
    search: ReviewSearch | None = None
    steam: SteamHttp | None = None
    tracer: Tracer = field(default_factory=NoTracer)


class _State(TypedDict, total=False):
    request: ReportRequest
    trace: Trace
    observer: Observer
    tools: Tools
    model_ids: dict[str, str]
    scope: Scope
    refusal: Refusal
    evidence: CoreEvidence
    hypotheses: list[HypothesisResult]
    basic: bool
    draft: Draft
    verified: Verified
    findings: list[Finding]
    summary: str
    summary_passed: bool | None
    report: Report
    started: float  # time.monotonic() when the run began


def _hash(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, default=str).encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def question_precheck(bundle: Bundle, question: str, appid: int | None) -> Refusal | None:
    """The refusals a question gets without a model call: empty, too long, or a picked game that
    isn't tracked (the API answers these without counting them against Ask's limits)."""
    text = question.strip()
    if not text:
        return Refusal("Ask a question about the reviews of one of the tracked games.")
    if len(text) > MAX_QUESTION_CHARS:
        return Refusal(f"Questions are limited to {MAX_QUESTION_CHARS} characters.")
    if appid is not None and not any(g.appid == appid and g.days for g in bundle.games):
        return Refusal(f"Game {appid} isn't one of the games PatchPulse tracks.")
    return None


class _Nodes:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps
        self.games = {g.appid: g for g in deps.bundle.games}

    # Helpers.

    def _ask(self, state: _State, step: str, user: str) -> dict[str, Any]:
        observer = state["observer"]
        observer.model_started(step)
        try:
            routed = self.deps.router.generate(step, load_prompt(step).text, user, SCHEMAS[step])
        except AllModelsDownError:
            observer.model_ended(step, None)
            raise
        call = ModelCall(
            step, routed.model_id, routed.input_tokens, routed.output_tokens, routed.cached
        )
        state["trace"].model_calls.append(call)
        observer.model_ended(step, call)
        state["model_ids"][step] = routed.model_id
        return routed.data

    def _scope_user(self, question: str, appid: int | None) -> str:
        games = [g for g in self.deps.bundle.games if appid is None or g.appid == appid]
        lines = [f"Today: {self.deps.bundle.built_on.isoformat()}", "", "Tracked games:"]
        for g in games:
            if g.days:
                lines.append(
                    f"- {g.name}: appid {g.appid}, data from {g.days[0].date} to {g.days[-1].date}"
                )
        lines += ["", "Patches:"]
        for g in games:
            for p in g.patches:
                if p.bundle_id is not None:
                    lines.append(f'- {p.patch_id}: {g.name}, {p.day}, "{p.title}"')
        lines += ["", "Question:", wrap_question(question)]
        return "\n".join(lines)

    def _question_scope(self, state: _State, question: str) -> Scope | Refusal:
        text = question.strip()
        picked = state["request"].appid
        early = question_precheck(self.deps.bundle, text, picked)
        if early is not None:
            return early
        try:
            data = self._ask(state, "scope", self._scope_user(text, picked))
        except AllModelsDownError:
            return Refusal(UNAVAILABLE)
        if data.get("refusal"):
            return Refusal(" ".join(str(data["refusal"]).split())[:MAX_REFUSAL_CHARS])
        if picked is not None and data.get("appid") != picked:
            return Refusal(
                f"The question seems to be about another game than {self.games[picked].name}, "
                "the one picked."
            )
        game = self.games.get(data.get("appid", -1))
        if game is None or not game.days:
            return Refusal(
                f"Game {data.get('appid', '?')} isn't one of the games PatchPulse tracks."
                if "appid" in data
                else "The question doesn't name one of the games PatchPulse tracks."
            )
        try:
            window = (date.fromisoformat(data["start"]), date.fromisoformat(data["end"]))
        except (KeyError, ValueError):
            return Refusal("The analyst couldn't work out the dates the question is about.")
        first, last = game.days[0].date, game.days[-1].date
        if window[0] > window[1]:
            return Refusal("The window asked about ends before it starts.")
        if window[0] < first or window[1] > last:
            return Refusal(
                f"The dates asked about are outside the data for {game.name} ({first} to {last})."
            )
        if days_in(window) > MAX_WINDOW_DAYS:
            return Refusal(f"Ask about at most {MAX_WINDOW_DAYS} days at a time.")
        if (window[0] - first).days < MIN_BASELINE_DAYS:
            return Refusal(
                f"{game.name}'s data starts too close to these dates to compare them with the "
                "weeks before."
            )
        patch = None
        if data.get("patch_id"):
            patch = next((p for p in game.patches if p.patch_id == data["patch_id"]), None)
            if patch is None:
                return Refusal(f"Patch {data['patch_id']} isn't one of {game.name}'s patches.")
        start = window[0]
        baseline_start = max(first, start - timedelta(days=BASELINE_DAYS))
        horizon = None
        if patch is not None:
            # The longest horizon the window covers that the bundle has estimates at: a bundle
            # with another patch within 14 days has no 14-day rows, but still its 7-day ones.
            estimated = {e.horizon for e in game.estimates if e.bundle_id == patch.bundle_id}
            horizon = next(
                (h for h in (14, 7) if days_in(window) >= h and h in estimated),
                14 if days_in(window) >= 14 else 7,
            )
        return Scope(
            appid=game.appid,
            window=window,
            baseline=(baseline_start, start - timedelta(days=1)),
            start=start,
            bundle_id=None if patch is None else patch.bundle_id,
            horizon=horizon,
            patch=patch,
        )

    def _context(self, state: _State) -> list[str]:
        scope = state["scope"]
        game = self.games[scope.appid]
        lines = [
            f"Game: {game.name} (appid {game.appid})",
            f"Window: {scope.window[0]} to {scope.window[1]}; baseline: {scope.baseline[0]} to "
            f"{scope.baseline[1]}",
        ]
        if scope.patch is not None:
            lines.append(
                f'Patch: {scope.patch.patch_id} on {scope.patch.day} ("{scope.patch.title}"); '
                f"horizon {scope.horizon} days"
            )
        else:
            lines.append("Patch: none")
        question = state["request"].question
        if question:
            lines += ["Question:", wrap_question(question)]
        lines += ["", "Explanations, as the rules set them:"]
        for h in state["hypotheses"]:
            lines.append(
                f"- {h.code} ({HYPOTHESIS_TITLES[h.code].lower()}): {h.status}, by {h.rule}"
            )
        return lines

    def _render(self, call: ToolCall, retrieved: Mapping[int, str]) -> str:
        head = f"[{call.call_id}] {call.tool} {_json(call.args)}"
        result = call.result
        if call.tool == "search_reviews" and "error" not in result:
            hits = [(h["review_id"], retrieved[h["review_id"]]) for h in result["hits"]]
            return f"{head} found {len(hits)} reviews:\n{wrap_reviews(hits)}"
        if call.tool == "get_patch_notes" and result.get("found"):
            return f"{head}:\n{wrap_patch_notes(call.args['patch_id'], result['notes'])}"
        return f"{head} -> {_json(result)}"

    # The nodes.

    def scope(self, state: _State) -> dict[str, Any]:
        request = state["request"]
        if request.question is None:
            bundle_id, horizon = str(request.bundle_id), int(request.horizon or HORIZONS[0])
            return {"scope": bundle_scope(self.deps.bundle, bundle_id, horizon)}
        placed = self._question_scope(state, request.question)
        if isinstance(placed, Refusal):
            return {"refusal": placed}
        return {"scope": placed}

    def core_evidence(self, state: _State) -> dict[str, Any]:
        return {"evidence": core_evidence(state["tools"], state["scope"])}

    def rules(self, state: _State) -> dict[str, Any]:
        return {"hypotheses": decide(state["evidence"], self.deps.bundle.causal)}

    def gather(self, state: _State) -> dict[str, Any]:
        tools, scope = state["tools"], state["scope"]
        lines = [*self._context(state), "", "Core tool results:"]
        lines += [self._render(c, tools.trace.retrieved) for c in tools.trace.calls]
        try:
            plan = self._ask(state, "gather", "\n".join(lines))
        except AllModelsDownError:
            return {"basic": True}
        for call in plan["calls"][:MAX_MODEL_TOOL_CALLS]:
            if call["tool"] == "get_patch_notes":
                patch_id = call.get("patch_id") or (scope.patch.patch_id if scope.patch else "")
                tools.get_patch_notes(patch_id)
            else:
                period = scope.baseline if call.get("period") == "baseline" else scope.window
                tools.search_reviews(
                    scope.appid,
                    call.get("query", ""),
                    period,
                    voted_up=call.get("voted_up"),
                    aspect=call.get("aspect"),
                    min_playtime=call.get("min_playtime"),
                )
        return {"basic": False}

    def draft(self, state: _State) -> dict[str, Any]:
        trace = state["trace"]
        lines = [*self._context(state), "", "Tool results:"]
        lines += [self._render(c, trace.retrieved) for c in trace.calls]
        try:
            answer = self._ask(state, "draft", "\n".join(lines))
            return {"draft": renumbered(Draft.model_validate(answer))}
        except (AllModelsDownError, ValidationError) as error:
            log.warning("draft: no usable draft (%s); writing a basic report", type(error).__name__)
            return {"basic": True}

    def verify(self, state: _State) -> dict[str, Any]:
        trace, tools, config = state["trace"], state["tools"], self.deps.bundle.causal
        if state.get("basic"):
            claims = basic_claims(state["evidence"])
            verified = verify_claims(claims, trace, tools, config)
            return {"verified": verified, "findings": _basic_findings(state["evidence"], verified)}
        draft = state["draft"]
        verified = verify_claims(draft.claims, trace, tools, config)
        findings = safe_findings(draft.findings, verified.claims, trace.retrieved)
        return {"verified": verified, "findings": findings}

    def summarize(self, state: _State) -> dict[str, Any]:
        verified, hypotheses = state["verified"], state["hypotheses"]
        template = template_summary(verified.claims, hypotheses)
        if state.get("basic"):
            return {"summary": template, "summary_passed": None}
        lines = ["Verified claims:"]
        for c in verified.claims:
            value = "" if c.value is None else f" [value {c.value} {c.unit or ''}]".rstrip()
            kind = c.claim_type + (", reworded by the verifier" if c.reworded else "")
            lines.append(f"- {c.claim_id} ({kind}): {c.text}{value}")
        lines += ["", "Explanations:"]
        lines += [
            f"- {h.code} ({HYPOTHESIS_TITLES[h.code].lower()}): {h.status}" for h in hypotheses
        ]
        try:
            data = self._ask(state, "summary", "\n".join(lines))
        except AllModelsDownError:
            return {"summary": template, "summary_passed": None}
        checked = verify_summary(str(data["summary"]), verified.claims, state["trace"].retrieved)
        if checked is None:
            return {"summary": template, "summary_passed": False}
        return {"summary": checked, "summary_passed": True}

    def finalize(self, state: _State) -> dict[str, Any]:
        request, scope, verified = state["request"], state["scope"], state["verified"]
        rates = dict(verified.pass_rates)
        if state.get("summary_passed") is not None:
            rates["V6"] = 1.0 if state["summary_passed"] else 0.0
        if request.question is None:
            report_id = f"{request.bundle_id}-{request.horizon}d"
        else:
            normalised = " ".join(request.question.casefold().split())
            report_id = "q-" + _hash(request.appid, normalised)[:12]
        report = Report(
            report_id=report_id,
            appid=scope.appid,
            scope="bundle" if request.question is None else "question",
            bundle_id=scope.bundle_id,
            horizon=scope.horizon,
            question=request.question,
            window=DateWindow(start=scope.window[0], end=scope.window[1]),
            data_version=self.deps.bundle.data_version,
            model_ids=dict(state["model_ids"]),
            prompt_version=PROMPT_VERSION,
            summary=state["summary"],
            findings=state["findings"],
            hypotheses=state["hypotheses"],
            claims=verified.claims,
            withheld_claims=verified.withheld,
            confidence="low",
            basic=bool(state.get("basic")),
            rule_pass_rates=dict(sorted(rates.items())),
            usage=[
                ModelUse(
                    step=c.step,
                    model_id=c.model_id,
                    input_tokens=c.input_tokens,
                    output_tokens=c.output_tokens,
                    cached=c.cached,
                )
                for c in state["trace"].model_calls
            ],
            seconds=round(time.monotonic() - state["started"], 3),
        )
        report = report.model_copy(update={"confidence": confidence_for(report)})
        log.info(
            "report %s: %d claims verified, %d withheld, basic=%s, models %s",
            report.report_id,
            len(report.claims),
            len(report.withheld_claims),
            report.basic,
            ",".join(sorted(set(report.model_ids.values()))) or "none",
        )
        return {"report": report}


def _basic_findings(evidence: CoreEvidence, verified: Verified) -> list[Finding]:
    """The basic report's claims filed under the explanation each bears on."""
    ids = evidence.call_ids
    by_call: dict[str | None, Hypothesis] = {
        ids.get("estimate"): "H1",
        ids.get("aspects"): "H3",
        ids.get("decomposition"): "H4",
    }
    grouped: dict[Hypothesis, list[str]] = {}
    for claim in verified.claims:
        if claim.evidence is None:
            continue
        code: Hypothesis = by_call.get(claim.evidence.call_id) or (
            "H2" if "new_players" in claim.evidence.json_path else "H5"
        )
        grouped.setdefault(code, []).append(claim.claim_id)
    return [
        Finding(title=HYPOTHESIS_TITLES[code], hypothesis=code, claim_ids=grouped[code])
        for code in sorted(grouped)
    ]


class _Step(Protocol):
    """A node, as LangGraph types it: called with the state, by name."""

    def __call__(self, state: _State) -> dict[str, Any]: ...


def build_graph(deps: AgentDeps) -> Any:
    nodes = _Nodes(deps)
    graph = StateGraph(_State)
    steps: dict[str, _Step] = {
        "scope": nodes.scope,
        "core_evidence": nodes.core_evidence,
        "rules": nodes.rules,
        "gather": nodes.gather,
        "draft": nodes.draft,
        "verify": nodes.verify,
        "summarize": nodes.summarize,
        "finalize": nodes.finalize,
    }
    for name, step in steps.items():
        graph.add_node(name, _traced(name, step))
    graph.add_edge(START, "scope")
    graph.add_conditional_edges(
        "scope", lambda s: END if "refusal" in s else "core_evidence", ["core_evidence", END]
    )
    graph.add_edge("core_evidence", "rules")
    graph.add_edge("rules", "gather")
    graph.add_conditional_edges(
        "gather", lambda s: "verify" if s.get("basic") else "draft", ["draft", "verify"]
    )
    graph.add_edge("draft", "verify")
    graph.add_edge("verify", "summarize")
    graph.add_edge("summarize", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def _traced(name: str, step: _Step) -> _Step:
    def run(state: _State) -> dict[str, Any]:
        observer = state["observer"]
        observer.step_started(name)
        update = step(state)
        observer.step_ended(name, update)
        return update

    return run


def run_report(request: ReportRequest, deps: AgentDeps) -> Report | Refusal:
    """One report, on the deps' bundle for every call."""
    run = _hash(
        request.bundle_id,
        request.horizon,
        request.question,
        request.appid,
        deps.bundle.data_version,
    )
    trace = Trace(run=run[:6])
    observer = deps.tracer.start(request, data_version=deps.bundle.data_version)
    tools = Tools(deps.bundle, trace, search=deps.search, steam=deps.steam, observer=observer)
    state: _State = {
        "request": request,
        "trace": trace,
        "observer": observer,
        "tools": tools,
        "model_ids": {},
        "started": time.monotonic(),
    }
    try:
        final = build_graph(deps).invoke(state)
    except BaseException as error:
        observer.finish(None, error)
        raise
    result: Report | Refusal = final["refusal"] if "refusal" in final else final["report"]
    observer.finish(result)
    return result
