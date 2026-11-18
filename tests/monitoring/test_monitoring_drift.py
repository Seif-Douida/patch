"""Phase 7 Task 3: the weekly drift statistics (design D2)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

import numpy as np
import pytest

from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.monitoring.alerts import load_rules
from patchpulse.monitoring.drift import (
    DRIFT_RULES,
    WeekSample,
    aspect_shares,
    domain_auc,
    drift_metrics,
    drift_outcomes,
    language_shares,
    length_shares,
    psi,
    uncertain_share,
)
from patchpulse.monitoring.reference import Reference
from patchpulse.monitoring.store import MetricRow

WINDOW = (date(2026, 11, 16), date(2026, 11, 22))


def test_psi_is_zero_for_equal_mixes_and_grows_with_the_shift() -> None:
    base = {"english": 0.6, "french": 0.2, "other": 0.2}
    small = {"english": 0.5, "french": 0.3, "other": 0.2}
    large = {"english": 0.3, "french": 0.5, "other": 0.2}

    assert psi(base, base) == pytest.approx(0.0)
    assert 0 < psi(base, small) < psi(base, large)


def test_psi_takes_the_union_of_keys() -> None:
    assert psi({"a": 1.0}, {"b": 1.0}) > 1.0


def test_unseen_languages_pool_as_other() -> None:
    shares = language_shares(["english", "english", "klingon", "french"], ["english", "french"])
    assert shares == {"english": 0.5, "french": 0.25, "other": 0.25}


def test_length_bins_cover_every_length() -> None:
    shares = length_shares([1, 50_000, 300])
    assert sum(shares.values()) == pytest.approx(1.0)
    assert sum(1 for v in shares.values() if v > 0) == 3


def test_domain_auc_is_about_half_for_one_distribution_and_high_for_two() -> None:
    rng = np.random.default_rng(1)
    same_a = rng.normal(size=(400, 16)).astype(np.float32)
    same_b = rng.normal(size=(400, 16)).astype(np.float32)
    shifted = rng.normal(loc=1.0, size=(400, 16)).astype(np.float32)

    assert 0.4 <= domain_auc(same_a, same_b) <= 0.6
    assert domain_auc(same_a, shifted) > 0.9
    assert math.isnan(domain_auc(same_a[:49], shifted))


def test_aspect_and_uncertain_shares() -> None:
    labels = np.zeros((4, len(ASPECTS)), dtype=bool)
    labels[:2, 0] = True
    probs = np.full((4, len(ASPECTS)), 0.9)
    probs[0, :] = 0.52
    thresholds = np.full(len(ASPECTS), 0.5)

    assert aspect_shares(labels)[ASPECTS[0]] == pytest.approx(0.5)
    assert uncertain_share(probs, thresholds) == pytest.approx(0.25)


class FakeVectors:
    def __init__(self, table: dict[int, np.ndarray]) -> None:
        self.table = table

    def vectors(self, ids: Sequence[int]) -> np.ndarray:
        return np.array([self.table[i] for i in ids if i in self.table], dtype=np.float32)


def reference(sample_ids: list[int], aspect_share: float = 0.1) -> Reference:
    return Reference(
        model_version="m1",
        languages_top=["english", "french"],
        language_shares={"english": 0.7, "french": 0.2, "other": 0.1},
        length_shares=length_shares([100] * 5 + [1000] * 5),
        aspect_shares={code: aspect_share for code in ASPECTS},
        uncertain_share=0.05,
        thresholds={code: 0.5 for code in ASPECTS},
        teacher_agreement_macro_f1=0.8,
        sample_ids=sample_ids,
        n=1000,
    )


def week(n: int, *, language: str = "english", ids_from: int = 10_000) -> WeekSample:
    labels = np.zeros((n, len(ASPECTS)), dtype=bool)
    labels[: n // 2, 3] = True
    return WeekSample(
        review_ids=list(range(ids_from, ids_from + n)),
        languages=[language] * n,
        lengths=[200] * n,
        labels=labels,
        probs=np.where(labels, 0.9, 0.1),
    )


def test_drift_metrics_name_every_statistic_and_the_largest_aspect_move() -> None:
    rng = np.random.default_rng(2)
    table = {i: rng.normal(size=8) for i in range(100)} | {
        i: rng.normal(loc=2.0, size=8) for i in range(10_000, 10_100)
    }
    rows = drift_metrics(reference(list(range(100))), week(100), FakeVectors(table), window=WINDOW)
    by_name = {row.metric: row for row in rows}

    assert {
        "language_psi",
        "length_psi",
        "embedding_auc",
        "aspect_max_change",
        "uncertain_change",
    } <= set(by_name)
    assert {f"aspect_share.{code}" for code in ASPECTS} <= set(by_name)
    assert all(row.family == "drift" and row.scope == "week" for row in rows)
    assert all((row.window_start, row.window_end) == WINDOW for row in rows)
    content = by_name[f"aspect_share.{ASPECTS[3]}"]
    assert content.value == pytest.approx(0.5)
    assert content.lo is not None
    assert content.hi is not None
    assert by_name["aspect_max_change"].value == pytest.approx(40.0)  # points
    assert (by_name["embedding_auc"].value or 0.0) > 0.9
    assert by_name["language_psi"].n == 100


def test_drift_outcomes_need_a_threshold_and_a_value() -> None:
    rules = load_rules()
    rules = rules | {k: rules[k].model_copy(update={"threshold": None}) for k in DRIFT_RULES}
    rules = rules | {
        "drift_language": rules["drift_language"].model_copy(update={"threshold": 0.2}),
        "drift_length": rules["drift_length"].model_copy(update={"threshold": 0.2}),
    }
    rows = [
        MetricRow("drift", "language_psi", "week", *WINDOW, 0.5, None, None, 100),
        MetricRow("drift", "length_psi", "week", *WINDOW, 0.1, None, None, 100),
        MetricRow("drift", "embedding_auc", "week", *WINDOW, 0.9, None, None, 100),
    ]
    outcomes = {o.rule_id: o for o in drift_outcomes(rows, rules)}

    assert outcomes["drift_language"].breached is True
    assert outcomes["drift_length"].breached is False
    assert outcomes["drift_embedding"].breached is None  # no threshold yet
    assert outcomes["drift_aspects"].breached is None  # no value
    assert "0.5" in outcomes["drift_language"].body
    assert set(outcomes) == set(DRIFT_RULES)
