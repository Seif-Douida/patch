"""The agent's traces in Langfuse (Phase 5 plan Task 18, design D13; spec §4: Langfuse Hobby).

One trace per report: a span per step, an observation per tool call (its name, its arguments and
the result's size; for search_reviews the review ids, scores and lengths), a generation per model
call (the model, its tokens and latency, whether the answer was a replay) and the verifier's
per-rule pass rates as scores. Review text, prompts, model answers, claims and the visitor's
question never leave the run: every payload below is built from fields picked one by one.

Nightly (and backfilled) reports are always traced. Ask's runs are traced at `ask_sample`, and not
at all once the month's estimated units pass MONTH_UNITS_CAP (of the Hobby plan's 50k: traces,
observations and scores, as Langfuse counts them). Every traced run adds its units to the month's
count in the limits store, which the nightly job and the API share (an estimate: two processes
adding at the same moment can lose one run's units). A trace that fails never fails the report:
the run carries on untraced and a warning is logged.
"""

from __future__ import annotations

import logging
import secrets
import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import SecretStr

if TYPE_CHECKING:
    from langfuse import Langfuse

    from patchpulse.agent.graph import Refusal, ReportRequest
    from patchpulse.agent.report import Report
    from patchpulse.agent.tools import ModelCall, ToolCall

log = logging.getLogger("patchpulse.agent.tracing")

DEFAULT_HOST = "https://cloud.langfuse.com"  # the EU region, where the account is
ASK_SAMPLE = 0.25
MONTH_UNITS_CAP = 45_000
_STEP_TYPES = {"verify": "evaluator"}  # the rest are chains
_TOOL_TYPES = {"search_reviews": "retriever"}  # the rest are tools
_DRAWS = secrets.SystemRandom()


def _utc_now() -> datetime:
    return datetime.now(UTC)


class CounterStore(Protocol):
    """Where the month's units are counted: the API's limits store (api/limits.py)."""

    def get(self, key: str) -> dict[str, Any] | None: ...

    def put(self, key: str, payload: dict[str, Any]) -> None: ...


class MonthUnits:
    """The month's estimated Langfuse units (UTC calendar months), one point in the store."""

    def __init__(self, store: CounterStore, *, clock: Callable[[], datetime] = _utc_now) -> None:
        self.store = store
        self.clock = clock
        self._lock = threading.Lock()

    def _month(self) -> str:
        return self.clock().astimezone(UTC).strftime("%Y-%m")

    def key(self) -> str:
        return f"langfuse-units|{self._month()}"

    def used(self) -> int:
        point = self.store.get(self.key()) or {}
        return int(point.get("units", 0))

    def add(self, units: int) -> None:
        with self._lock:
            month = self._month()
            self.store.put(self.key(), {"month": month, "units": self.used() + units})


# What each event may send: counts, ids, model names, statuses. Nothing else.


def _window(window: tuple[Any, Any]) -> list[str]:
    return [day.isoformat() for day in window]


def step_output(name: str, update: Mapping[str, Any]) -> dict[str, Any]:
    if "refusal" in update:
        return {"refused": True}
    if update.get("basic"):
        return {"basic": True}
    if name == "scope":
        scope = update["scope"]
        return {
            "appid": scope.appid,
            "window": _window(scope.window),
            "baseline": _window(scope.baseline),
            "patch_id": None if scope.patch is None else scope.patch.patch_id,
            "bundle_id": scope.bundle_id,
            "horizon": scope.horizon,
        }
    if name == "rules":
        return {"hypotheses": {h.code: f"{h.status} ({h.rule})" for h in update["hypotheses"]}}
    if name == "draft":
        draft = update["draft"]
        return {"findings": len(draft.findings), "claims": len(draft.claims)}
    if name == "verify":
        verified = update["verified"]
        withheld: dict[str, int] = {}
        for w in verified.withheld:
            withheld[w.rule] = withheld.get(w.rule, 0) + 1
        return {
            "verified": len(verified.claims),
            "withheld": dict(sorted(withheld.items())),
            "findings": len(update["findings"]),
            "pass_rates": dict(sorted(verified.pass_rates.items())),
        }
    if name == "summarize":
        return {"summary_passed": update["summary_passed"]}
    if name == "finalize":
        report = update["report"]
        return {"confidence": report.confidence, "basic": report.basic}
    return {}


def tool_output(call: ToolCall, retrieved: Mapping[int, str]) -> dict[str, Any]:
    result = call.result
    if "error" in result:
        return {"error": str(result["error"])}
    if call.tool == "search_reviews":
        hits = [
            {
                "review_id": h["review_id"],
                "score": h["score"],
                "chars": len(retrieved.get(h["review_id"], "")),
            }
            for h in result["hits"]
        ]
        return {"hits": hits}
    if call.tool == "get_patch_notes":
        return {"found": bool(result.get("found")), "chars": len(str(result.get("notes") or ""))}
    return {"keys": sorted(result), "chars": len(repr(result))}


def run_input(request: ReportRequest) -> dict[str, Any]:
    if request.question is None:
        return {"bundle_id": request.bundle_id, "horizon": request.horizon}
    return {"appid": request.appid, "question_chars": len(request.question)}


def report_output(report: Report) -> dict[str, Any]:
    return {
        "report_id": report.report_id,
        "appid": report.appid,
        "window": [report.window.start.isoformat(), report.window.end.isoformat()],
        "bundle_id": report.bundle_id,
        "horizon": report.horizon,
        "confidence": report.confidence,
        "basic": report.basic,
        "claims": len(report.claims),
        "withheld": len(report.withheld_claims),
        "model_ids": dict(report.model_ids),
    }


# The events a run reports.


class Observer:
    """What one run reports its steps, tool calls and model calls to; this one drops them (an
    untraced run). Events come from one thread, in order: a step's tool and model calls come
    between its start and its end."""

    @property
    def trace_id(self) -> str | None:
        """The trace's id in Langfuse; None for an untraced run."""
        return None

    def step_started(self, name: str) -> None:
        return None

    def step_ended(self, name: str, update: Mapping[str, Any]) -> None:
        return None

    def tool_started(self, tool: str, args: Mapping[str, Any]) -> None:
        return None

    def tool_ended(self, call: ToolCall, retrieved: Mapping[int, str]) -> None:
        return None

    def model_started(self, step: str) -> None:
        return None

    def model_ended(self, step: str, call: ModelCall | None) -> None:
        """`call` is None when no model answered."""
        return None

    def finish(self, result: Report | Refusal | None, error: BaseException | None = None) -> None:
        return None


class Tracer(Protocol):
    def start(self, request: ReportRequest, *, data_version: int) -> Observer: ...

    def close(self) -> None: ...


class NoTracer:
    def start(self, request: ReportRequest, *, data_version: int) -> Observer:
        return Observer()

    def close(self) -> None:
        return None


class LangfuseObserver(Observer):
    """A traced run. Each event runs through `_safely`: the first failure is logged and the rest
    of the run goes untraced."""

    def __init__(self, root: Any, name: str, units: MonthUnits) -> None:
        self.root = root
        self.name = name
        self.units = units
        self.broken = False
        self.created = 1  # the root observation
        self._open: list[Any] = []  # the observations started and not ended, innermost last

    @property
    def trace_id(self) -> str | None:
        found = getattr(self.root, "trace_id", None)
        return found if isinstance(found, str) else None

    def _safely(self, event: Callable[[], None]) -> None:
        if self.broken:
            return
        try:
            event()
        except Exception as error:
            self.broken = True
            log.warning(
                "trace of %s: Langfuse failed (%s); the run goes on untraced",
                self.name,
                type(error).__name__,
            )

    def _start(self, name: str, as_type: str, **fields: Any) -> None:
        parent = self._open[-1] if self._open else self.root
        self._open.append(parent.start_observation(name=name, as_type=as_type, **fields))
        self.created += 1

    def _end(self, **fields: Any) -> None:
        observation = self._open.pop()
        if fields:
            observation.update(**fields)
        observation.end()

    def step_started(self, name: str) -> None:
        self._safely(lambda: self._start(name, _STEP_TYPES.get(name, "chain")))

    def step_ended(self, name: str, update: Mapping[str, Any]) -> None:
        self._safely(lambda: self._end(output=step_output(name, update)))

    def tool_started(self, tool: str, args: Mapping[str, Any]) -> None:
        self._safely(lambda: self._start(tool, _TOOL_TYPES.get(tool, "tool"), input=dict(args)))

    def tool_ended(self, call: ToolCall, retrieved: Mapping[int, str]) -> None:
        self._safely(lambda: self._end(output=tool_output(call, retrieved)))

    def model_started(self, step: str) -> None:
        self._safely(lambda: self._start(step, "generation"))

    def model_ended(self, step: str, call: ModelCall | None) -> None:
        if call is None:
            self._safely(lambda: self._end(level="ERROR", status_message="no model answered"))
            return
        self._safely(
            lambda: self._end(
                model=call.model_id,
                usage_details={"input": call.input_tokens, "output": call.output_tokens},
                metadata={"cached": call.cached},
            )
        )

    def finish(self, result: Report | Refusal | None, error: BaseException | None = None) -> None:
        self._safely(lambda: self._finish(result, error))

    def _finish(self, result: Report | Refusal | None, error: BaseException | None) -> None:
        from patchpulse.agent.report import Report

        while self._open:  # a step that raised leaves its observations open
            self._end(level="ERROR", status_message="the run stopped here")
        scores: dict[str, float] = {}
        if error is not None:
            self.root.update(level="ERROR", status_message=type(error).__name__)
        elif isinstance(result, Report):
            self.root.update(output=report_output(result))
            scores = dict(result.rule_pass_rates)
        else:
            self.root.update(output={"refused": True})
        for rule, rate in scores.items():
            self.root.score_trace(name=f"{rule} pass rate", value=rate)
        self.root.end()
        units = 1 + self.created + len(scores)  # the trace, its observations, its scores
        try:
            self.units.add(units)
        except Exception as failure:
            log.warning("trace of %s: the month's units weren't counted (%s)", self.name,
                        type(failure).__name__)  # fmt: skip


class LangfuseTracer:
    """Traces through `client` (a configured Langfuse client); `rng` draws Ask's sample."""

    def __init__(
        self,
        client: Langfuse,
        *,
        units: MonthUnits,
        ask_sample: float = ASK_SAMPLE,
        cap: int = MONTH_UNITS_CAP,
        rng: Callable[[], float] = _DRAWS.random,
    ) -> None:
        self.client = client
        self.units = units
        self.ask_sample = ask_sample
        self.cap = cap
        self.rng = rng

    def _ask_traced(self) -> bool:
        if self.rng() >= self.ask_sample:
            return False
        try:
            return self.units.used() < self.cap
        except Exception as error:
            log.warning("Ask's trace skipped: the month's units can't be read (%s)",
                        type(error).__name__)  # fmt: skip
            return False

    def start(self, request: ReportRequest, *, data_version: int) -> Observer:
        from langfuse import propagate_attributes

        from patchpulse.agent.prompts import PROMPT_VERSION

        ask = request.question is not None
        if ask and not self._ask_traced():
            return Observer()
        name = "ask" if ask else f"report {request.bundle_id}-{request.horizon}d"
        kind = "ask" if ask else "report"
        try:
            with propagate_attributes(
                trace_name=kind,
                tags=[kind],
                metadata={"data_version": str(data_version), "prompt_version": PROMPT_VERSION},
            ):
                root = self.client.start_observation(
                    name=name, as_type="agent", input=run_input(request)
                )
        except Exception as error:
            log.warning("trace of %s: Langfuse failed (%s); the run goes on untraced", name,
                        type(error).__name__)  # fmt: skip
            return Observer()
        return LangfuseObserver(root, name, self.units)

    def close(self) -> None:
        """Send what's left (the nightly job and the backfill call it at the end)."""
        try:
            self.client.shutdown()
        except Exception as error:
            log.warning("Langfuse shutdown failed (%s)", type(error).__name__)


def make_tracer(
    public_key: str | None,
    secret_key: SecretStr | None,
    host: str | None,
    *,
    store: CounterStore,
    ask_sample: float = ASK_SAMPLE,
) -> Tracer:
    """A Langfuse tracer when both keys are set, else none."""
    if not public_key or secret_key is None:
        return NoTracer()
    from langfuse import Langfuse  # the SDK starts its export threads: only when tracing

    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key.get_secret_value(),
        base_url=host or DEFAULT_HOST,
    )
    return LangfuseTracer(client, units=MonthUnits(store), ask_sample=ask_sample)
