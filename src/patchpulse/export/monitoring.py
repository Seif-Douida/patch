"""`monitoring.json`, the Monitoring page's data (Phase 7 design D10): numbers only.

Built each night by the export stage from `ops.pipeline_run`, `ops.data_quality_result` and
`ops.eval_metric`, with the rules of `config/alerts.yaml` and the champion's pin:

* `pipeline`: the last 30 finished nightly runs, oldest first: `run_id`, `started_at`, `status`,
  `duration_s` and `stages_s` (seconds per stage). Nothing else of a run's record leaves: its
  error and its stages' details can quote what failed.
* `freshness`: `last_success` (when the last successful night finished) and `age_hours`.
* `data_quality`: per check and severity, how many results the last 30 runs recorded. Never
  the detail text.
* `drift`: per drift rule, its statistic, its threshold (null until calibrated) and its weekly
  `points` (`window_end`, `value`, `drill`: true for the acceptance drill's).
* `agreement`: the weekly teacher-student macro-F1 with its interval and `n`, the champion's
  reference value and the alert's threshold.
* `online`: per night, the reports judged, Gemini errors (basic reports), the verifier's pass rate,
  withheld claims per rule, reworded claims, and the judge's mean scores, marked
  `uncalibrated: true` (Phase 6: the judge isn't calibrated against people yet).
* `retrain`: the last request (`date`, `reason`) and the last decision (`decision`, the recent
  holdout's macro-F1 gain, `n`).
* `rules` and `champion` (registry version, family, model version).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Connection, bindparam, text

from patchpulse.monitoring.alerts import AlertRule
from patchpulse.monitoring.drift import DRIFT_RULES
from patchpulse.monitoring.store import MetricRow, read_metrics

if TYPE_CHECKING:
    from patchpulse.models.champion import ChampionPin

MONITORING_SCHEMA = 1
MONITORING_FILE = "monitoring.json"
RUNS = 30
ONLINE_NIGHTS = 60
_PLACES = 4

_RUNS_SQL = """
SELECT TOP (:runs) run_id, started_at, finished_at, status, stages
FROM ops.pipeline_run
WHERE kind = 'nightly' AND status <> 'running'
ORDER BY run_id DESC
"""
_LAST_SUCCESS_SQL = """
SELECT MAX(finished_at) FROM ops.pipeline_run WHERE kind = 'nightly' AND status = 'succeeded'
"""
_QUALITY_SQL = """
SELECT check_name, severity, COUNT(*) AS n
FROM ops.data_quality_result
WHERE run_id IN :runs
GROUP BY check_name, severity
"""


@dataclass(frozen=True)
class RunRow:
    run_id: int
    started_at: datetime | None
    finished_at: datetime | None
    status: str
    durations: dict[str, float]


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), _PLACES)


def _iso(moment: datetime | None) -> str | None:
    if moment is None:
        return None
    aware = moment if moment.tzinfo else moment.replace(tzinfo=UTC)
    return aware.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _day(day: date | None) -> str | None:
    return None if day is None else day.isoformat()


def durations(stages: str | None) -> dict[str, float]:
    """Only the seconds per stage of a run's record: its other fields can quote an error."""
    try:
        record = json.loads(stages or "{}")
    except ValueError:
        return {}
    found = record.get("durations_s") if isinstance(record, dict) else None
    if not isinstance(found, dict):
        return {}
    return {
        str(name): float(seconds)
        for name, seconds in found.items()
        if isinstance(seconds, int | float) and not isinstance(seconds, bool)
    }


def _pipeline(runs: Sequence[RunRow]) -> list[dict[str, Any]]:
    shaped = []
    for run in sorted(runs, key=lambda r: r.run_id):
        seconds = None
        if run.started_at is not None and run.finished_at is not None:
            seconds = (run.finished_at - run.started_at).total_seconds()
        shaped.append(
            {
                "run_id": run.run_id,
                "started_at": _iso(run.started_at),
                "status": run.status,
                "duration_s": _round(seconds),
                "stages_s": run.durations,
            }
        )
    return shaped


def _drift(rows: Sequence[MetricRow], rules: Mapping[str, AlertRule]) -> list[dict[str, Any]]:
    sections = []
    for rule_id, metric in DRIFT_RULES.items():
        points = sorted(
            (r for r in rows if r.metric == metric and r.scope in ("week", "drill")),
            key=lambda r: (r.window_end or date.min, r.scope == "drill"),
        )
        sections.append(
            {
                "rule": rule_id,
                "metric": metric,
                "threshold": _round(rules[rule_id].threshold) if rule_id in rules else None,
                "points": [
                    {
                        "window_end": _day(r.window_end),
                        "value": _round(r.value),
                        "drill": r.scope == "drill",
                    }
                    for r in points
                ],
            }
        )
    return sections


def _agreement(
    rows: Sequence[MetricRow], rules: Mapping[str, AlertRule], reference: float | None
) -> dict[str, Any]:
    rule = rules.get("agreement_dropped")
    weeks = [
        {
            "window_end": _day(r.window_end),
            "value": _round(r.value),
            "lo": _round(r.lo),
            "hi": _round(r.hi),
            "n": r.n,
        }
        for r in rows
        if r.metric == "macro_f1" and r.scope == "week"
    ]
    return {
        "reference": _round(reference),
        "threshold": None if rule is None else _round(rule.threshold),
        "weeks": weeks,
    }


def _count(value: float | None) -> int | None:
    return None if value is None else round(value)


def _online(rows: Sequence[MetricRow]) -> dict[str, Any]:
    nights: dict[date, list[MetricRow]] = defaultdict(list)
    for row in rows:
        if row.window_end is not None:
            nights[row.window_end].append(row)
    shaped = []
    for night in sorted(nights)[-ONLINE_NIGHTS:]:
        own = {r.metric: r for r in nights[night] if r.scope == "night"}
        withheld: dict[str, float] = defaultdict(float)
        reworded = 0.0
        judged: dict[str, list[float]] = defaultdict(list)
        for row in nights[night]:
            if not row.scope.startswith("report:") or row.value is None:
                continue
            if row.metric.startswith("withheld."):
                withheld[row.metric.removeprefix("withheld.")] += row.value
            elif row.metric == "reworded":
                reworded += row.value
            elif row.metric.startswith("judge."):
                judged[row.metric.removeprefix("judge.")].append(row.value)

        def value(metric: str, nights_own: Mapping[str, MetricRow] = own) -> float | None:
            found = nights_own.get(metric)
            return None if found is None else found.value

        shaped.append(
            {
                "date": night.isoformat(),
                "reports": _count(value("reports")),
                "gemini_errors": _count(value("gemini_errors")),
                "pass_rate": _round(value("pre_verifier_pass_rate")),
                "withheld": {rule: round(n) for rule, n in sorted(withheld.items())},
                "reworded": round(reworded),
                "judge": {
                    "uncalibrated": True,
                    "scores": {
                        key: _round(sum(scores) / len(scores))
                        for key, scores in sorted(judged.items())
                    },
                },
            }
        )
    return {"nights": shaped}


def _retrain(rows: Sequence[MetricRow]) -> dict[str, Any]:
    requests = [r for r in rows if r.metric.startswith("requested.")]
    decisions = [r for r in rows if r.metric.startswith("decision.")]
    last_request = requests[-1] if requests else None
    last_decision = decisions[-1] if decisions else None
    return {
        "last_request": None
        if last_request is None
        else {
            "date": _day(last_request.window_end),
            "reason": last_request.metric.removeprefix("requested."),
        },
        "last_decision": None
        if last_decision is None
        else {
            "date": _day(last_decision.window_end),
            "decision": last_decision.metric.removeprefix("decision."),
            "recent_gain": _round(last_decision.value),
            "n": last_decision.n,
        },
    }


def shape_monitoring(
    *,
    runs: Sequence[RunRow],
    last_success: datetime | None,
    quality: Sequence[tuple[str, str, int]],
    metrics: Mapping[str, Sequence[MetricRow]],
    rules: Mapping[str, AlertRule],
    champion: ChampionPin | None,
    reference_agreement: float | None,
    generated_at: datetime,
) -> bytes:
    age = None
    if last_success is not None:
        finished = last_success if last_success.tzinfo else last_success.replace(tzinfo=UTC)
        age = round((generated_at - finished).total_seconds() / 3600, 1)
    document = {
        "schema_version": MONITORING_SCHEMA,
        "generated_at": _iso(generated_at),
        "pipeline": _pipeline(runs),
        "freshness": {"last_success": _iso(last_success), "age_hours": age},
        "data_quality": [
            {"check": check, "severity": severity, "count": int(n)}
            for check, severity, n in sorted(quality)
        ],
        "drift": _drift(metrics.get("drift", []), rules),
        "agreement": _agreement(metrics.get("agreement", []), rules, reference_agreement),
        "online": _online(metrics.get("online", [])),
        "retrain": _retrain(metrics.get("retrain", [])),
        "rules": [
            {
                "id": rule.id,
                "title": rule.title,
                "label": rule.label,
                "threshold": rule.threshold,
                "meaning": rule.meaning,
            }
            for rule in rules.values()
        ],
        "champion": None
        if champion is None
        else {
            "version": champion.version,
            "family": champion.family,
            "model_version": champion.model_version,
        },
    }
    dumped = json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return (dumped + "\n").encode("utf-8")


def build_monitoring(
    connection: Connection,
    *,
    rules: Mapping[str, AlertRule],
    champion: ChampionPin | None,
    generated_at: datetime,
    reference_agreement: float | None = None,
) -> bytes:
    runs = [
        RunRow(int(r.run_id), r.started_at, r.finished_at, str(r.status), durations(r.stages))
        for r in connection.execute(text(_RUNS_SQL), {"runs": RUNS})
    ]
    last_success: datetime | None = connection.execute(text(_LAST_SUCCESS_SQL)).scalar_one()
    quality: list[tuple[str, str, int]] = []
    if runs:
        statement = text(_QUALITY_SQL).bindparams(bindparam("runs", expanding=True))
        quality = [
            (str(r.check_name), str(r.severity), int(r.n))
            for r in connection.execute(statement, {"runs": [run.run_id for run in runs]})
        ]
    metrics = {
        family: read_metrics(connection, family=family)
        for family in ("drift", "agreement", "online", "retrain")
    }
    return shape_monitoring(
        runs=runs,
        last_success=last_success,
        quality=quality,
        metrics=metrics,
        rules=rules,
        champion=champion,
        reference_agreement=reference_agreement,
        generated_at=generated_at,
    )


def live_monitoring(connection: Connection, *, generated_at: datetime) -> bytes:
    """The nightly's file: the rules of config/alerts.yaml, the champion's pin and the reference's
    agreement (models/, baked into the jobs image), each left out if missing."""
    from patchpulse.models.champion import PIN_FILE, read_pin
    from patchpulse.monitoring.alerts import load_rules
    from patchpulse.monitoring.reference import REFERENCE_FILE, read_reference

    champion = read_pin() if PIN_FILE.exists() else None
    reference = read_reference().teacher_agreement_macro_f1 if REFERENCE_FILE.exists() else None
    return build_monitoring(
        connection,
        rules=load_rules(),
        champion=champion,
        generated_at=generated_at,
        reference_agreement=reference,
    )
