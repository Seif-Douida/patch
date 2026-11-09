"""Stored cases (Phase 6 design D2): one run of one case in one arm, as JSON under data/local.

A stored case keeps what the scoring, the judge and the verifier check read: the report (or the
refusal), the scoped window, the model's draft and raw summary, and the whole trace, the full text
of every review retrieved included. That text is why the files live under data/local, which git
ignores, and why a folder anywhere else is refused.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from patchpulse.agent.graph import Refusal, RunRecord
from patchpulse.agent.report import Draft, Report
from patchpulse.agent.tools import ModelCall, ToolCall, Trace

EVALS_DATA = Path("data") / "local" / "evals"
CACHE_FILE = "answers.sqlite"
RUNS_DIR = "runs"


class _Frozen(BaseModel):
    # A run that no longer fits (a later schema change) must fail without echoing its text:
    # the smoke's errors reach CI's public log. This hides the nested models' inputs too.
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class StoredCall(_Frozen):
    call_id: str
    tool: str
    args: dict[str, Any]
    result: dict[str, Any]


class StoredModelCall(_Frozen):
    step: str
    model_id: str
    input_tokens: int
    output_tokens: int
    cached: bool


class StoredTrace(_Frozen):
    run: str
    calls: list[StoredCall]
    model_calls: list[StoredModelCall]
    retrieved: dict[int, str]  # review id -> the full text search returned


class StoredCase(_Frozen):
    case: str
    arm: str
    data_version: int
    misses: list[str]  # what a replay couldn't find: "model:<step>" or "tool:<name>"
    refusal: str | None
    report: Report | None
    window: tuple[date, date] | None
    draft: Draft | None
    raw_summary: str | None
    trace: StoredTrace
    basic: bool
    indexed_reviews: int | None = None  # reviews in the scoped window the index held

    @classmethod
    def from_record(
        cls,
        record: RunRecord,
        *,
        case: str,
        arm: str,
        data_version: int,
        misses: list[str],
        indexed_reviews: int | None,
    ) -> StoredCase:
        result = record.result
        trace = record.trace
        return cls(
            case=case,
            arm=arm,
            data_version=data_version,
            misses=list(misses),
            refusal=result.reason if isinstance(result, Refusal) else None,
            report=result if isinstance(result, Report) else None,
            window=record.window,
            draft=record.draft,
            raw_summary=record.raw_summary,
            trace=StoredTrace(
                run=trace.run,
                calls=[StoredCall(**vars(c)) for c in trace.calls],
                model_calls=[StoredModelCall(**vars(c)) for c in trace.model_calls],
                retrieved=dict(trace.retrieved),
            ),
            basic=record.basic,
            indexed_reviews=indexed_reviews,
        )

    def record(self) -> RunRecord:
        """The run as the scoring reads it."""
        result: Report | Refusal = (
            self.report if self.report is not None else Refusal(self.refusal or "")
        )
        trace = Trace(
            run=self.trace.run,
            calls=[ToolCall(c.call_id, c.tool, c.args, c.result) for c in self.trace.calls],
            model_calls=[
                ModelCall(c.step, c.model_id, c.input_tokens, c.output_tokens, c.cached)
                for c in self.trace.model_calls
            ],
            retrieved=dict(self.trace.retrieved),
        )
        return RunRecord(
            result=result,
            window=self.window,
            hypotheses=list(self.report.hypotheses) if self.report is not None else [],
            draft=self.draft,
            raw_summary=self.raw_summary,
            trace=trace,
            basic=self.basic,
        )


def require_local(path: Path) -> None:
    """Stored runs, the eval cache and the data hold review text: only under data/local."""
    parts = path.resolve().parts
    if not any(parts[i : i + 2] == ("data", "local") for i in range(len(parts) - 1)):
        raise ValueError(f"eval runs hold review text: keep them under data/local, not {path}")


def runs_dir(arm: str, root: Path = EVALS_DATA) -> Path:
    return root / RUNS_DIR / arm


def save_case(stored: StoredCase, path: Path) -> None:
    require_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = json.dumps(stored.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    path.write_text(document + "\n", encoding="utf-8", newline="\n")


def load_case(path: Path) -> StoredCase:
    return StoredCase.model_validate_json(path.read_text(encoding="utf-8"))


def load_arm(arm: str, *, root: Path = EVALS_DATA) -> dict[str, StoredCase]:
    """Every stored case of an arm, by case id."""
    folder = runs_dir(arm, root)
    return {path.stem: load_case(path) for path in sorted(folder.glob("*.json"))}
