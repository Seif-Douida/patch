"""The promotion rule (Phase 7 design D5): when a retrained S0 replaces the champion.

The challenger and the champion are compared on the same reviews:

- **recent_macro_f1:** macro-F1 on the recent holdout (the last 8 weeks' spot-check holdout,
  labeled by the teacher) at least `margin` above the champion's, with the paired bootstrap's 95%
  lower bound above 0;
- **recent_by_language:** no language with at least `language_min_reviews` of those reviews more
  than `language_drop` lower;
- **gold_dev_macro_f1:** out-of-fold gold dev macro-F1 no more than `dev_drop` lower;
- **reviews_per_second:** at least `min_reviews_per_second` on one thread.

Macro-F1 on teacher labels is over the aspects the teacher labeled at least once (Task 6's
`assessable_macro_f1`), as the weekly agreement is. Gold test never decides: it is reported only.
Numbers only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

from patchpulse.models.metrics import IndexMetric, assessable_macro_f1, paired_bootstrap_ci

RULE_FILE = Path("config") / "retrain.yaml"
_EPSILON = 1e-9  # a value exactly at a limit passes, whatever the float rounding


class PromotionRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    margin: float
    language_drop: float
    language_min_reviews: PositiveInt
    dev_drop: float
    min_reviews_per_second: PositiveFloat
    bootstrap_seed: int


def load_rule(path: Path = RULE_FILE) -> PromotionRule:
    return PromotionRule.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class PairedSet:
    """Reviews both models labeled: (reviews, aspects) boolean matrices, aligned row by row."""

    y_true: NDArray[np.bool_]
    champion: NDArray[np.bool_]
    challenger: NDArray[np.bool_]
    languages: Sequence[str]


@dataclass(frozen=True)
class Check:
    name: str
    champion: float | None
    challenger: float | None
    limit: float | None  # what the challenger's value must reach
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class Decision:
    promote: bool
    checks: list[Check]


def _f1(y_true: NDArray[np.bool_], predicted: NDArray[np.bool_]) -> float:
    return assessable_macro_f1(y_true, predicted)


def _on_rows(y_true: NDArray[np.bool_], predicted: NDArray[np.bool_]) -> IndexMetric:
    return lambda ix: _f1(y_true[ix], predicted[ix])


def _recent_check(recent: PairedSet, rule: PromotionRule) -> Check:
    n = len(recent.y_true)
    champion, challenger = (
        _f1(recent.y_true, recent.champion),
        _f1(recent.y_true, recent.challenger),
    )
    if n == 0 or np.isnan(champion) or np.isnan(challenger):
        return Check("recent_macro_f1", None, None, None, False, "no recent holdout labels")

    lo, hi = paired_bootstrap_ci(
        _on_rows(recent.y_true, recent.challenger),
        _on_rows(recent.y_true, recent.champion),
        n,
        seed=rule.bootstrap_seed,
    )
    gain = challenger - champion
    passed = gain >= rule.margin - _EPSILON and lo > 0
    detail = f"gain {gain:+.3f}, 95% interval {lo:+.3f} to {hi:+.3f}, n {n}"
    return Check("recent_macro_f1", champion, challenger, champion + rule.margin, passed, detail)


def _language_check(recent: PairedSet, rule: PromotionRule) -> Check:
    languages = np.asarray(recent.languages)
    worst: tuple[float, str, float, float, int] | None = None
    for language in sorted(set(recent.languages)):
        rows = languages == language
        n = int(rows.sum())
        if n < rule.language_min_reviews:
            continue
        champion = _f1(recent.y_true[rows], recent.champion[rows])
        challenger = _f1(recent.y_true[rows], recent.challenger[rows])
        if np.isnan(champion) or np.isnan(challenger):
            continue
        if worst is None or challenger - champion < worst[0]:
            worst = (challenger - champion, language, champion, challenger, n)
    if worst is None:
        detail = f"no language with {rule.language_min_reviews}+ recent reviews"
        return Check("recent_by_language", None, None, None, True, detail)
    change, language, champion, challenger, n = worst
    limit = champion - rule.language_drop
    passed = challenger >= limit - _EPSILON
    detail = f"weakest: {language} ({change:+.3f}, n {n})"
    return Check("recent_by_language", champion, challenger, limit, passed, detail)


def decide_promotion(
    recent: PairedSet,
    *,
    dev_champion: float,
    dev_challenger: float,
    reviews_per_second: float,
    rule: PromotionRule,
) -> Decision:
    dev_limit = dev_champion - rule.dev_drop
    checks = [
        _recent_check(recent, rule),
        _language_check(recent, rule),
        Check(
            "gold_dev_macro_f1",
            dev_champion,
            dev_challenger,
            dev_limit,
            dev_challenger >= dev_limit - _EPSILON,
            "out of fold",
        ),
        Check(
            "reviews_per_second",
            None,
            reviews_per_second,
            rule.min_reviews_per_second,
            reviews_per_second >= rule.min_reviews_per_second,
            "one thread",
        ),
    ]
    return Decision(promote=all(check.passed for check in checks), checks=checks)


def _cell(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render_table(
    decision: Decision,
    *,
    champion_version: str,
    challenger_version: str,
    test_champion: float,
    test_challenger: float,
    counts: Mapping[str, int],
) -> str:
    """The comparison as Markdown, for the MLflow run, the PR body and the drift issue."""
    verdict = "promote" if decision.promote else "keep the champion"
    lines = [
        f"Challenger {challenger_version} against champion {champion_version}: {verdict}.",
        "",
        "| Check | Champion | Challenger | Limit | Passed | Detail |",
        "|---|---|---|---|---|---|",
    ]
    for check in decision.checks:
        digits = 0 if check.name == "reviews_per_second" else 3
        lines.append(
            f"| {check.name} | {_cell(check.champion, digits)} | {_cell(check.challenger, digits)} "
            f"| {_cell(check.limit, digits)} | {'yes' if check.passed else 'no'} | {check.detail} |"
        )
    lines.append(
        f"| gold test macro-F1 (reported only) | {test_champion:.3f} | {test_challenger:.3f} "
        "| - | - | never decides |"
    )
    lines += ["", "Reviews: " + ", ".join(f"{name} {n}" for name, n in counts.items()) + "."]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class RetrainResult:
    """What pp-retrain decided, as retrain.yml reads it back from the MLflow run (Task 12)."""

    version: str  # the challenger's model.json version
    decision: Decision
    mlflow_run_id: str
    registered_version: int | None  # in the registry, with the `challenger` alias
    model_json_sha256: str
    table: str


def result_json(result: RetrainResult) -> str:
    return json.dumps(asdict(result), indent=2, sort_keys=True) + "\n"


def parse_result(text: str) -> RetrainResult:
    raw = json.loads(text)
    decision = raw["decision"]
    return RetrainResult(
        version=raw["version"],
        decision=Decision(
            promote=bool(decision["promote"]),
            checks=[Check(**check) for check in decision["checks"]],
        ),
        mlflow_run_id=raw["mlflow_run_id"],
        registered_version=raw["registered_version"],
        model_json_sha256=raw["model_json_sha256"],
        table=raw["table"],
    )
