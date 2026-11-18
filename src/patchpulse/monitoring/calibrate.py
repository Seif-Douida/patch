"""Drift thresholds from the project's own history (Phase 7 design D2, plan Task 4).

Every past week of the local snapshot (Monday to Sunday, the tracked games' reviews by creation
date) is scored by the champion and measured against the reference, as the nightly will measure a
new week. Each statistic's threshold is its 99th percentile over those weeks: an alert then means
"more unusual than 99 weeks in 100 of this project's history". Weeks with fewer than 200 reviews
are skipped and counted. The thresholds go into `config/alerts.yaml`, the percentiles into a report.
Numbers only.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from patchpulse.models.artifact import Scorer
from patchpulse.models.snapshot import Snapshot
from patchpulse.monitoring.alerts import ALERTS_FILE
from patchpulse.monitoring.drift import (
    DRIFT_RULES,
    MIN_WEEK_REVIEWS,
    VectorSource,
    WeekSample,
    drift_metrics,
)
from patchpulse.monitoring.reference import Reference

CALIBRATION_PERCENTILE = 99
REPORT_PERCENTILES = (50, 90, 95, 99)

Window = tuple[date, date]


@dataclass(frozen=True)
class Calibration:
    thresholds: dict[str, float]  # drift rule id -> threshold
    weeks: int  # weeks measured
    skipped: int  # weeks under MIN_WEEK_REVIEWS
    values: dict[str, list[float]] = field(default_factory=dict)  # statistic -> its weekly values


def _first_monday(day: date) -> date:
    return day + timedelta(days=(7 - day.weekday()) % 7)


def weekly_samples(
    snapshot: Snapshot, scorer: Scorer, *, start: date, end: date
) -> Iterator[tuple[Window, WeekSample]]:
    """Every full Monday-to-Sunday week from `start` to `end`, small ones included."""
    reviews = snapshot.reviews
    created = pd.to_datetime(reviews["date"]).dt.date
    monday = _first_monday(start)
    while monday + timedelta(days=6) <= end:
        sunday = monday + timedelta(days=6)
        week = reviews[(created >= monday) & (created <= sunday)]
        texts = [str(t) for t in week["text"]]
        probs = scorer.predict_proba(texts) if texts else np.zeros((0, 10))
        yield (
            (monday, sunday),
            WeekSample(
                review_ids=[int(i) for i in week["review_id"]],
                languages=[str(lang) for lang in week["language"]],
                lengths=[len(t) for t in texts],
                labels=scorer.labels(probs) if texts else np.zeros((0, 10), dtype=bool),
                probs=probs,
            ),
        )
        monday += timedelta(days=7)


def percentile_thresholds(
    values: Mapping[str, list[float]], percentile: float = CALIBRATION_PERCENTILE
) -> dict[str, float]:
    """Drift rule id -> the percentile of its statistic's weekly values (rules with none left
    out)."""
    return {
        rule_id: float(np.percentile(values[metric], percentile))
        for rule_id, metric in DRIFT_RULES.items()
        if values.get(metric)
    }


def calibrate(
    reference: Reference,
    samples: Iterable[tuple[Window, WeekSample]],
    vectors: VectorSource,
    *,
    percentile: float = CALIBRATION_PERCENTILE,
) -> Calibration:
    values: dict[str, list[float]] = {metric: [] for metric in DRIFT_RULES.values()}
    weeks = skipped = 0
    for window, sample in samples:
        if len(sample.review_ids) < MIN_WEEK_REVIEWS:
            skipped += 1
            continue
        weeks += 1
        for row in drift_metrics(reference, sample, vectors, window=window):
            if row.metric in values and row.value is not None and not math.isnan(row.value):
                values[row.metric].append(row.value)
    return Calibration(percentile_thresholds(values, percentile), weeks, skipped, values)


def write_thresholds(thresholds: Mapping[str, float], path: Path = ALERTS_FILE) -> None:
    """Rewrite the `threshold:` line of each named drift rule; every other byte stays."""
    unknown = set(thresholds) - set(DRIFT_RULES)
    if unknown:
        raise ValueError(f"not drift rules: {sorted(unknown)}")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    current: str | None = None
    written = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("- id:"):
            current = stripped.removeprefix("- id:").strip()
        elif stripped.startswith("threshold:") and current is not None and current in thresholds:
            indent = line[: len(line) - len(line.lstrip())]
            lines[i] = f"{indent}threshold: {thresholds[current]:.4g}\n"
            written.add(current)
    missing = set(thresholds) - written
    if missing:
        raise ValueError(f"{path}: no threshold line for {sorted(missing)}")
    path.write_text("".join(lines), encoding="utf-8", newline="\n")


def render_calibration(calibration: Calibration) -> str:
    lines = [
        "# Drift thresholds, calibrated on past weeks",
        "",
        f"{calibration.weeks} weeks measured, {calibration.skipped} skipped (fewer than "
        f"{MIN_WEEK_REVIEWS} reviews). Each threshold is its statistic's "
        f"{CALIBRATION_PERCENTILE}th percentile over the weeks measured (Phase 7 design D2).",
        "",
        "| Rule | Statistic | p50 | p90 | p95 | p99 | Threshold | Weeks above it "
        "| Weeks measured |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rule_id, metric in DRIFT_RULES.items():
        weekly = calibration.values.get(metric, [])
        threshold = calibration.thresholds.get(rule_id)
        if not weekly:
            lines.append(f"| {rule_id} | {metric} | - | - | - | - | none | - | 0 |")
            continue
        cuts = [f"{np.percentile(weekly, p):.3f}" for p in REPORT_PERCENTILES]
        above = "-" if threshold is None else str(sum(v > threshold for v in weekly))
        shown = "none" if threshold is None else f"{threshold:.4g}"
        lines.append(
            f"| {rule_id} | {metric} | {' | '.join(cuts)} | {shown} | {above} | {len(weekly)} |"
        )
    lines += [
        "",
        "The embedding statistic needs each week's e5 vectors from the review index, which holds "
        "each game's 40,000 most recent reviews and its patch and launch windows (ADR-022): it is "
        "measured on the weeks the index covers, the other statistics on every week.",
    ]
    return "\n".join(lines) + "\n"
