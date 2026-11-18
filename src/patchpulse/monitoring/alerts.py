"""Every alert's rule and the evaluator that raises and resolves its issue (Phase 7 design D7,
ADR-013).

A rule lives in `config/alerts.yaml`. Each check produces an `Outcome` per rule: breached, back
within its threshold, or not measured. A breach opens the rule's issue or comments on it; a
recovery closes it; an unmeasured rule leaves it alone. In the acceptance drill (design D11) titles
carry `[drill] ` and nothing is ever resolved, so a drill can't close a real issue.

Bodies are built by the checks from numbers, ids and run ids. As a guard the evaluator refuses a
long line or a run of plain words, which is what text echoed from a review would look like.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, TypeAdapter

ALERTS_FILE = Path("config") / "alerts.yaml"
DRILL_PREFIX = "[drill] "
MAX_LINE = 200
# Seven words in a row with no digit among them reads like prose, not a measurement.
_PROSE = re.compile(r"(?:\b[^\W\d_]{2,}\b[ ,;'-]+){6}\b[^\W\d_]{2,}\b")

Label = Literal["alert", "drift"]
Where = Literal["nightly", "weekly", "index", "reports", "uptime", "cost_guard"]


class AlertRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    title: str
    label: Label
    where: Where
    threshold: float | None
    meaning: str


@dataclass(frozen=True)
class Outcome:
    rule_id: str
    breached: bool | None  # None: not measured this time; the issue stays as it is
    body: str


class AlertPublisher(Protocol):
    def alert(self, *, title: str, body: str, label: str = ...) -> None: ...

    def resolve(self, *, title: str, comment: str, label: str = ...) -> None: ...


def load_rules(path: Path = ALERTS_FILE) -> dict[str, AlertRule]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    rules = TypeAdapter(list[AlertRule]).validate_python(raw)
    ids = [rule.id for rule in rules]
    titles = [rule.title for rule in rules]
    for kind, values in (("id", ids), ("title", titles)):
        repeated = sorted({v for v in values if values.count(v) > 1})
        if repeated:
            raise ValueError(f"{path}: duplicate alert {kind}: {', '.join(repeated)}")
    return {rule.id: rule for rule in rules}


def _check_body(body: str) -> None:
    for line in body.splitlines():
        if len(line) > MAX_LINE or _PROSE.search(line):
            raise ValueError(
                "an alert body must hold numbers, ids and run ids, not text: "
                f"a line of {len(line)} characters was refused"
            )


def apply_outcomes(
    publisher: AlertPublisher,
    rules: Mapping[str, AlertRule],
    outcomes: Sequence[Outcome],
    *,
    drill: bool = False,
) -> list[str]:
    """Raise each breach and resolve each recovery; returns the titles raised. Every body is
    checked before anything is sent."""
    for outcome in outcomes:
        _check_body(outcome.body)
    raised = []
    for outcome in outcomes:
        rule = rules[outcome.rule_id]
        title = DRILL_PREFIX + rule.title if drill else rule.title
        if outcome.breached:
            publisher.alert(title=title, body=outcome.body, label=rule.label)
            raised.append(title)
        elif outcome.breached is False and not drill:
            publisher.resolve(title=title, comment=outcome.body, label=rule.label)
    return raised


PLAN_FILE = "issues.tsv"


class IssuePlan:
    """An `AlertPublisher` for the workflows that check from GitHub (uptime.yml, cost-guard.yml):
    it records each action, and `.github/scripts/apply-issue-plan.sh` applies them with `gh` and
    the workflow's GITHUB_TOKEN. One line per action: action, label, title, body file."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []

    def alert(self, *, title: str, body: str, label: str = "alert") -> None:
        self.rows.append(("alert", label, title, body))

    def resolve(self, *, title: str, comment: str, label: str = "alert") -> None:
        self.rows.append(("resolve", label, title, comment))

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        lines = []
        for number, (action, label, title, body) in enumerate(self.rows, start=1):
            name = f"issue-{number}.md"
            (directory / name).write_text(body.rstrip("\n") + "\n", encoding="utf-8", newline="\n")
            lines.append("\t".join((action, label, title, name)))
        plan = directory / PLAN_FILE
        plan.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8", newline="\n")
        return plan
