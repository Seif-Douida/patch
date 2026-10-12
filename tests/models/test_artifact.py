"""The model artifact contract (plan Task 11): `model.json` plus checksummed files, loaded only if
the taxonomy, the cleaning and every checksum match."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from toy_reviews import toy_reviews

from patchpulse.labeling.taxonomy import ASPECTS, TAXONOMY_VERSION
from patchpulse.models.artifact import MODEL_FILE, ArtifactError, load_scorer, read_artifact
from patchpulse.models.s0 import S0_FILE, S0Params, save_s0, train_s0
from patchpulse.models.text import CLEANING_VERSION


@pytest.fixture
def saved(tmp_path: Path) -> Path:
    texts, labels = toy_reviews(200, seed=4)
    model = train_s0(texts, labels, params=S0Params(min_df=1))
    save_s0(model, tmp_path, thresholds=np.linspace(0.3, 0.7, 10), version="20261012T200000Z")
    return tmp_path


def test_artifact_round_trip_gives_identical_probabilities(tmp_path: Path) -> None:
    texts, labels = toy_reviews(200, seed=4)
    unseen, _ = toy_reviews(30, seed=5)
    model = train_s0(texts, labels, params=S0Params(min_df=1))
    cuts = np.linspace(0.3, 0.7, 10)

    written = save_s0(model, tmp_path, thresholds=cuts, version="20261012T200000Z")
    scorer = load_scorer(tmp_path)
    probs = scorer.predict_proba(unseen)

    assert np.array_equal(probs, model.predict_proba(unseen))
    assert np.array_equal(scorer.labels(probs), probs >= cuts)
    assert read_artifact(tmp_path) == written
    assert written.family == "s0"
    assert written.taxonomy_version == TAXONOMY_VERSION
    assert written.cleaning_version == CLEANING_VERSION
    assert written.aspects == ASPECTS
    assert set(written.files) == {S0_FILE}


def edit_record(change: Callable[[dict[str, Any]], None]) -> Callable[[Path], None]:
    def apply(model_dir: Path) -> None:
        path = model_dir / MODEL_FILE
        record = json.loads(path.read_text(encoding="utf-8"))
        change(record)
        path.write_text(json.dumps(record), encoding="utf-8")

    return apply


def corrupt(model_dir: Path) -> None:
    with (model_dir / S0_FILE).open("ab") as file:
        file.write(b"\0")


def remove(model_dir: Path) -> None:
    (model_dir / S0_FILE).unlink()


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (edit_record(lambda r: r.update(taxonomy_version="v0")), "taxonomy"),
        (edit_record(lambda r: r.update(cleaning_version="c0")), "cleaning"),
        (edit_record(lambda r: r.update(aspects=list(reversed(ASPECTS)))), "aspects"),
        (edit_record(lambda r: r["thresholds"].pop("off_topic")), "thresholds"),
        (edit_record(lambda r: r.update(family="s9")), "family"),
        (edit_record(lambda r: r["files"].update({"../elsewhere.joblib": "0" * 64})), "file name"),
        (corrupt, "checksum"),
        (remove, "missing"),
    ],
)
def test_artifact_with_unknown_taxonomy_or_bad_checksum_is_refused(
    saved: Path, damage: Callable[[Path], None], message: str
) -> None:
    damage(saved)

    with pytest.raises(ArtifactError, match=message):
        load_scorer(saved)


def test_a_directory_without_a_record_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match=MODEL_FILE):
        load_scorer(tmp_path)


def test_scoring_needs_only_the_scoring_group() -> None:
    """The jobs image has NumPy, scikit-learn and joblib, not pandas, pyarrow or MLflow."""
    blocked = "import sys; sys.modules.update(dict.fromkeys(['pandas', 'pyarrow', 'mlflow']))"
    modules = "patchpulse.models.artifact, patchpulse.models.s0, patchpulse.models.thresholds"
    result = subprocess.run(
        [sys.executable, "-c", f"{blocked}; import {modules}"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
