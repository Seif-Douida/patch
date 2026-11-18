"""Phase 7 Task 4: drift thresholds from the project's own history (design D2)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.monitoring.alerts import ALERTS_FILE, load_rules
from patchpulse.monitoring.calibrate import (
    Calibration,
    calibrate,
    percentile_thresholds,
    render_calibration,
    weekly_samples,
    write_thresholds,
)
from patchpulse.monitoring.drift import DRIFT_RULES, MIN_WEEK_REVIEWS
from patchpulse.monitoring.reference import Reference

SENTINEL = "SENTINEL-REVIEW-the menus lag on every console since launch"


class FakeScorer:
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return np.full((len(texts), len(ASPECTS)), 0.3)

    def labels(self, probs: np.ndarray) -> np.ndarray:
        return probs > 0.5


def snapshot_reviews() -> pd.DataFrame:
    days = pd.date_range("2026-01-05", "2026-01-25", freq="D")  # Monday 5 to Sunday 25 Jan
    rows = []
    for i, day in enumerate(days):
        count = MIN_WEEK_REVIEWS if day < pd.Timestamp("2026-01-19") else 3
        for k in range(count // 7 + 1):
            rows.append(
                {
                    "review_id": i * 1000 + k,
                    "date": day.date(),
                    "language": "english",
                    "text": SENTINEL,
                }
            )
    return pd.DataFrame(rows)


class Snap:
    def __init__(self, reviews: pd.DataFrame) -> None:
        self.reviews = reviews


def test_weeks_run_monday_to_sunday_and_skip_small_ones() -> None:
    weeks = list(
        weekly_samples(
            Snap(snapshot_reviews()),  # type: ignore[arg-type]
            FakeScorer(),  # type: ignore[arg-type]
            start=date(2026, 1, 3),
            end=date(2026, 1, 26),
        )
    )

    assert [w for w, _ in weeks] == [
        (date(2026, 1, 5), date(2026, 1, 11)),
        (date(2026, 1, 12), date(2026, 1, 18)),
        (date(2026, 1, 19), date(2026, 1, 25)),
    ]
    sizes = [len(sample.review_ids) for _, sample in weeks]
    assert sizes[0] >= MIN_WEEK_REVIEWS
    assert sizes[2] < MIN_WEEK_REVIEWS
    assert weeks[0][1].lengths[0] == len(SENTINEL)


def test_the_threshold_is_the_99th_percentile_of_past_weeks() -> None:
    values = {metric: [float(v) for v in range(1, 101)] for metric in DRIFT_RULES.values()}
    thresholds = percentile_thresholds(values, 99)

    assert set(thresholds) == set(DRIFT_RULES)
    assert thresholds["drift_language"] == pytest.approx(99.01)


class NoVectors:
    def vectors(self, ids: Sequence[int]) -> np.ndarray:
        return np.zeros((0, 4), dtype=np.float32)


def test_calibrate_skips_small_weeks_and_counts_them() -> None:
    reference = Reference(
        model_version="m",
        languages_top=["english"],
        language_shares={"english": 1.0},
        length_shares={"bin4": 1.0},
        aspect_shares={c: 0.0 for c in ASPECTS},
        uncertain_share=0.0,
        thresholds={c: 0.5 for c in ASPECTS},
        teacher_agreement_macro_f1=0.7,
        sample_ids=[],
        n=1,
    )
    samples = weekly_samples(
        Snap(snapshot_reviews()),  # type: ignore[arg-type]
        FakeScorer(),  # type: ignore[arg-type]
        start=date(2026, 1, 5),
        end=date(2026, 1, 25),
    )
    result = calibrate(reference, samples, NoVectors())

    assert (result.weeks, result.skipped) == (2, 1)
    assert len(result.values["language_psi"]) == 2
    assert result.values["embedding_auc"] == []  # no vectors: no AUC, no threshold
    assert "drift_embedding" not in result.thresholds


def test_write_thresholds_changes_only_the_drift_lines(tmp_path: Path) -> None:
    path = tmp_path / "alerts.yaml"
    original = ALERTS_FILE.read_text(encoding="utf-8")
    path.write_text(original, encoding="utf-8", newline="\n")

    write_thresholds({"drift_language": 0.2345, "drift_embedding": 0.61}, path)

    rules = load_rules(path)
    assert rules["drift_language"].threshold == pytest.approx(0.2345)
    assert rules["drift_embedding"].threshold == pytest.approx(0.61)
    assert rules["drift_length"].threshold == load_rules(ALERTS_FILE)["drift_length"].threshold
    changed = [
        (a, b)
        for a, b in zip(
            original.splitlines(), path.read_text(encoding="utf-8").splitlines(), strict=True
        )
        if a != b
    ]
    assert len(changed) == 2
    assert all(
        a.strip().startswith("threshold:") and b.strip().startswith("threshold:")
        for a, b in changed
    )


def test_the_calibration_report_holds_no_text() -> None:
    calibration = Calibration(
        thresholds={"drift_language": 0.31},
        weeks=90,
        skipped=4,
        values={"language_psi": [0.01 * i for i in range(90)]},
    )
    report = render_calibration(calibration)

    assert "0.31" in report
    assert "90 weeks" in report
    assert SENTINEL not in report


def test_the_report_counts_the_weeks_each_statistic_was_measured_on() -> None:
    # 18 Nov: the review index holds each game's recent reviews and its patch windows (ADR-022),
    # so the embedding statistic is measured only on the weeks it covers; the others on all.
    calibration = Calibration(
        thresholds={"drift_language": 0.31, "drift_embedding": 0.6},
        weeks=90,
        skipped=4,
        values={"language_psi": [0.01] * 90, "embedding_auc": [0.55] * 12},
    )

    rows = {
        line.split(" | ")[0].strip("| "): line
        for line in render_calibration(calibration).splitlines()
        if line.startswith("| drift_")
    }

    assert rows["drift_language"].endswith("| 90 |")
    assert rows["drift_embedding"].endswith("| 12 |")
    assert rows["drift_length"].endswith("| 0 |")
