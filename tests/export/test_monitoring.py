"""Phase 7 Task 15: monitoring.json, the Monitoring page's data (design D10), numbers only."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import Engine, insert

from patchpulse.db import tables
from patchpulse.export.monitoring import MONITORING_SCHEMA, build_monitoring
from patchpulse.ingest.load import record_dq
from patchpulse.models.champion import ChampionPin
from patchpulse.monitoring.alerts import load_rules
from patchpulse.monitoring.store import MetricRow, write_metrics

pytestmark = pytest.mark.integration

RULES = load_rules()
CHAMPION = ChampionPin(
    version=3,
    family="s0",
    model_version="20261012T223218Z",
    model_json_sha256="ab" * 32,
    taxonomy_version="v1",
)
GENERATED = datetime(2026, 12, 1, 3, 40, tzinfo=UTC)
SENTINEL = "SENTINEL-REVIEW the patch broke my saves and the support ignored me"
SECTIONS = {
    "schema_version",
    "generated_at",
    "pipeline",
    "freshness",
    "data_quality",
    "drift",
    "agreement",
    "online",
    "retrain",
    "rules",
    "champion",
}


def build(engine: Engine, **kwargs: Any) -> dict[str, Any]:
    with engine.connect() as connection:
        raw = build_monitoring(
            connection, rules=RULES, champion=CHAMPION, generated_at=GENERATED, **kwargs
        )
    loaded: dict[str, Any] = json.loads(raw)
    return loaded


def add_run(engine: Engine, run_id: int, *, status: str = "succeeded", **fields: Any) -> None:
    started = datetime(2026, 11, 1, 3, 0) + timedelta(days=run_id)
    row = {
        "run_id": run_id,
        "kind": "nightly",
        "started_at": started,
        "finished_at": started + timedelta(minutes=31),
        "status": status,
        "stages": json.dumps({"durations_s": {"reviews": 120.5, "score": 61.0}}),
        **fields,
    }
    with engine.begin() as connection:  # run ids as the test needs them: an identity column
        connection.exec_driver_sql("SET IDENTITY_INSERT ops.pipeline_run ON")
        connection.execute(insert(tables.pipeline_run), [row])
        connection.exec_driver_sql("SET IDENTITY_INSERT ops.pipeline_run OFF")


def test_the_file_has_every_section_when_empty(migrated_engine: Engine) -> None:
    document = build(migrated_engine)

    assert set(document) == SECTIONS
    assert document["schema_version"] == MONITORING_SCHEMA
    assert document["pipeline"] == []
    assert document["freshness"] == {"last_success": None, "age_hours": None}
    assert document["data_quality"] == []
    assert [s["rule"] for s in document["drift"]] == [
        "drift_language",
        "drift_length",
        "drift_embedding",
        "drift_aspects",
        "drift_confidence",
    ]
    assert all(s["points"] == [] for s in document["drift"])
    assert document["agreement"]["weeks"] == []
    assert document["agreement"]["reference"] is None
    assert document["online"] == {"nights": []}
    assert document["retrain"] == {"last_request": None, "last_decision": None}
    assert document["champion"] == {
        "version": 3,
        "family": "s0",
        "model_version": "20261012T223218Z",
    }
    assert {r["id"] for r in document["rules"]} == set(RULES)


def test_the_pipeline_keeps_the_last_thirty_finished_nights(migrated_engine: Engine) -> None:
    for run_id in range(1, 36):
        add_run(migrated_engine, run_id, status="failed" if run_id == 34 else "succeeded")
    add_run(migrated_engine, 36, status="running", finished_at=None)
    add_run(migrated_engine, 37, kind="migrate")

    document = build(migrated_engine)

    runs = document["pipeline"]
    assert [r["run_id"] for r in runs] == list(range(6, 36))  # oldest first
    assert runs[-1] == {
        "run_id": 35,
        "started_at": "2026-12-06T03:00:00Z",
        "status": "succeeded",
        "duration_s": 1860.0,
        "stages_s": {"reviews": 120.5, "score": 61.0},
    }
    assert document["freshness"]["last_success"] == "2026-12-06T03:31:00Z"


def test_drill_points_are_marked(migrated_engine: Engine) -> None:
    week = (date(2026, 11, 23), date(2026, 11, 29))
    with migrated_engine.begin() as connection:
        write_metrics(
            connection,
            [
                MetricRow("drift", "language_psi", "week", *week, 0.04, None, None, 900),
                MetricRow("drift", "language_psi", "drill", *week, 0.61, None, None, 900),
            ],
            run_id=7,
        )

    (language,) = [s for s in build(migrated_engine)["drift"] if s["rule"] == "drift_language"]

    assert language["metric"] == "language_psi"
    assert language["points"] == [
        {"window_end": "2026-11-29", "value": 0.04, "drill": False},
        {"window_end": "2026-11-29", "value": 0.61, "drill": True},
    ]


def test_judge_scores_are_marked_uncalibrated(migrated_engine: Engine) -> None:
    night = date(2026, 11, 30)

    def row(metric: str, scope: str, value: float, n: int | None = None) -> MetricRow:
        return MetricRow("online", metric, scope, night, night, value, None, None, n)

    with migrated_engine.begin() as connection:
        write_metrics(
            connection,
            [
                row("reports", "night", 2.0),
                row("gemini_errors", "night", 1.0),
                row("pre_verifier_pass_rate", "night", 0.8, 10),
                row("withheld.V3", "report:5001", 1.0),
                row("withheld.V3", "report:5002", 2.0),
                row("reworded", "report:5001", 1.0),
                row("judge.faithfulness", "report:5001", 0.9, 5),
                row("judge.faithfulness", "report:5002", 0.7, 5),
            ],
            run_id=8,
        )

    (entry,) = build(migrated_engine)["online"]["nights"]

    assert entry == {
        "date": "2026-11-30",
        "reports": 2,
        "gemini_errors": 1,
        "pass_rate": 0.8,
        "withheld": {"V3": 3},
        "reworded": 1,
        "judge": {"uncalibrated": True, "scores": {"faithfulness": 0.8}},
    }


def test_agreement_and_retrain_come_from_their_rows(migrated_engine: Engine) -> None:
    with migrated_engine.begin() as connection:
        write_metrics(
            connection,
            [
                MetricRow(
                    "agreement",
                    "macro_f1",
                    "week",
                    date(2026, 11, 23),
                    date(2026, 11, 29),
                    0.71,
                    0.66,
                    0.75,
                    700,
                ),
                MetricRow(
                    "agreement",
                    "f1.content",
                    "week",
                    date(2026, 11, 23),
                    date(2026, 11, 29),
                    0.8,
                    None,
                    None,
                    90,
                ),
                MetricRow(
                    "retrain",
                    "requested.drift",
                    "night",
                    date(2026, 11, 30),
                    date(2026, 11, 30),
                    1.0,
                    None,
                    None,
                    None,
                ),
                MetricRow(
                    "retrain",
                    "decision.keep",
                    "run",
                    date(2026, 11, 30),
                    date(2026, 11, 30),
                    0.004,
                    None,
                    None,
                    812,
                ),
            ],
            run_id=9,
        )

    document = build(migrated_engine, reference_agreement=0.74)

    assert document["agreement"] == {
        "reference": 0.74,
        "threshold": RULES["agreement_dropped"].threshold,
        "weeks": [{"window_end": "2026-11-29", "value": 0.71, "lo": 0.66, "hi": 0.75, "n": 700}],
    }
    assert document["retrain"] == {
        "last_request": {"date": "2026-11-30", "reason": "drift"},
        "last_decision": {"date": "2026-11-30", "decision": "keep", "recent_gain": 0.004, "n": 812},
    }


def test_monitoring_file_holds_no_text(migrated_engine: Engine) -> None:
    add_run(
        migrated_engine,
        1,
        status="failed",
        error=f"reports: {SENTINEL}",
        stages=json.dumps({"durations_s": {"reports": 3.0}, "reports": {"error": SENTINEL}}),
    )
    with migrated_engine.begin() as connection:
        record_dq(connection, run_id=1, check="reports_failed", appid=None, severity="warn",
                  detail=SENTINEL)  # fmt: skip

    with migrated_engine.connect() as connection:
        raw = build_monitoring(connection, rules=RULES, champion=CHAMPION, generated_at=GENERATED)

    assert b"SENTINEL" not in raw
    document = json.loads(raw)
    assert document["data_quality"] == [{"check": "reports_failed", "severity": "warn", "count": 1}]
    assert document["pipeline"][0]["stages_s"] == {"reports": 3.0}
