"""Phase 7 Task 3: the drift reference, the champion's training data as numbers and ids."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.monitoring.reference import build_reference, read_reference, write_reference

SENTINEL = "SENTINEL-REVIEW-text that must never reach the reference file"


def inputs(n: int = 3000) -> dict[str, object]:
    rng = np.random.default_rng(3)
    labels = rng.random((n, len(ASPECTS))) < 0.2
    return {
        "model_version": "20261012T223218Z",
        "languages": ["english"] * (n - 400) + ["french"] * 300 + ["koreana"] * 100,
        "lengths": [int(x) for x in rng.integers(5, 5000, size=n)],
        "labels": labels,
        "probs": np.where(labels, 0.8, 0.2),
        "thresholds": np.full(len(ASPECTS), 0.5),
        "ids": list(range(1, n + 1)),
        "holdout_agreement": 0.71,
    }


def test_the_reference_round_trips_and_holds_ids_and_numbers_only(tmp_path: Path) -> None:
    reference = build_reference(**inputs())  # type: ignore[arg-type]
    path = tmp_path / "reference.json"
    write_reference(reference, path)

    assert read_reference(path) == reference
    assert reference.n == 3000
    assert len(reference.sample_ids) == 1000
    assert set(reference.sample_ids) <= set(range(1, 3001))
    assert reference.languages_top[:2] == ["english", "french"]
    assert reference.teacher_agreement_macro_f1 == 0.71
    raw = path.read_text(encoding="utf-8")
    assert SENTINEL not in raw
    assert raw.endswith("\n")
    assert "\r" not in raw
    strings = [v for v in _leaves(json.loads(raw)) if isinstance(v, str)]
    assert all(len(s) <= 40 for s in strings)  # names and versions, never text


def test_the_reference_is_fixed_by_its_seed() -> None:
    assert build_reference(**inputs()) == build_reference(**inputs())  # type: ignore[arg-type]


def _leaves(value: object) -> list[object]:
    if isinstance(value, dict):
        return [*value.keys(), *(x for v in value.values() for x in _leaves(v))]
    if isinstance(value, list):
        return [x for v in value for x in _leaves(v)]
    return [value]


def test_the_committed_reference_is_the_pinned_champions() -> None:
    # Final review I3 (18 Nov): a pin changed without `patchpulse-ml reference build` turns the
    # weekly drift check off ("the reference doesn't match"); CI catches it on the pull request.
    from patchpulse.models.champion import read_pin

    assert read_reference().model_version == read_pin().model_version
