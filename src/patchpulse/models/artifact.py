"""The model artifact every student is saved as, and the one way to load it (plan Task 11).

A model directory holds `model.json` and the files it lists:

    {"family": "s0", "version": "20261012T200000Z", "taxonomy_version": "v1",
     "cleaning_version": "c1", "aspects": [...], "thresholds": {"performance": 0.31, ...},
     "files": {"s0.joblib": "<sha256>"}}

`load_scorer` refuses a taxonomy or cleaning version this code doesn't speak, an aspect order that
isn't the taxonomy's, and any listed file that is missing or whose SHA-256 doesn't match, before a
single byte of the model is deserialized. A joblib file can run code when loaded, so only our own
artifacts are loaded: built by `patchpulse-ml train`, logged to our MLflow, baked into our image.

NumPy and the standard library only, so the jobs image can load models (the `scoring` group).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray

from patchpulse.labeling.taxonomy import ASPECTS, TAXONOMY_VERSION
from patchpulse.models.text import CLEANING_VERSION

MODEL_FILE = "model.json"
Family = Literal["s0", "s1a", "s1b"]
FAMILIES: tuple[str, ...] = get_args(Family)


class ArtifactError(RuntimeError):
    """A model directory that can't be trusted or used by this code."""


@dataclass(frozen=True)
class ModelArtifact:
    family: Family
    version: str
    taxonomy_version: str
    cleaning_version: str
    aspects: tuple[str, ...]
    thresholds: tuple[float, ...]  # one cut per aspect, in `aspects` order
    files: dict[str, str]  # file name in the model directory -> SHA-256

    def cuts(self) -> NDArray[np.float64]:
        return np.array(self.thresholds, dtype=np.float64)


class Scorer(Protocol):
    artifact: ModelArtifact

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Raw review texts in, (reviews, aspects) probabilities out; cleaning is the scorer's."""
        ...

    def labels(self, probs: NDArray[np.float64]) -> NDArray[np.bool_]: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_name(name: str) -> bool:
    return name not in ("", ".", "..", MODEL_FILE) and Path(name).name == name and "\\" not in name


def write_artifact(
    model_dir: Path,
    *,
    family: Family,
    version: str,
    thresholds: ArrayLike,
    files: Sequence[str],
) -> ModelArtifact:
    """Record the model files already written to `model_dir`, with their checksums."""
    cuts = np.asarray(thresholds, dtype=np.float64)
    if cuts.shape != (len(ASPECTS),) or not np.isfinite(cuts).all():
        raise ValueError(f"need one finite threshold for each of the {len(ASPECTS)} aspects")
    if not all(_plain_name(name) for name in files):
        raise ValueError(f"model files must be plain names in the model directory: {files}")
    artifact = ModelArtifact(
        family=family,
        version=version,
        taxonomy_version=TAXONOMY_VERSION,
        cleaning_version=CLEANING_VERSION,
        aspects=ASPECTS,
        thresholds=tuple(float(cut) for cut in cuts),
        files={name: sha256_file(model_dir / name) for name in files},
    )
    record = {
        "family": artifact.family,
        "version": artifact.version,
        "taxonomy_version": artifact.taxonomy_version,
        "cleaning_version": artifact.cleaning_version,
        "aspects": list(artifact.aspects),
        "thresholds": dict(zip(artifact.aspects, artifact.thresholds, strict=True)),
        "files": artifact.files,
    }
    (model_dir / MODEL_FILE).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return artifact


def _parse(record: dict[str, Any]) -> ModelArtifact:
    if record.get("family") not in FAMILIES:
        raise ArtifactError(f"unknown model family {record.get('family')!r}")
    if record.get("taxonomy_version") != TAXONOMY_VERSION:
        raise ArtifactError(
            f"taxonomy {record.get('taxonomy_version')!r} isn't this code's {TAXONOMY_VERSION!r}"
        )
    if record.get("cleaning_version") != CLEANING_VERSION:
        raise ArtifactError(
            f"cleaning {record.get('cleaning_version')!r} isn't this code's {CLEANING_VERSION!r}"
        )
    if tuple(record.get("aspects", ())) != ASPECTS:
        raise ArtifactError("the model's aspects aren't the taxonomy's, in its order")
    thresholds = record.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(ASPECTS):
        raise ArtifactError("the model needs thresholds for exactly the taxonomy's aspects")
    files = record.get("files")
    if not isinstance(files, dict) or not files:
        raise ArtifactError("the model lists no files")
    bad = [name for name in files if not isinstance(name, str) or not _plain_name(name)]
    if bad:
        raise ArtifactError(f"model file name outside the model directory: {bad}")
    return ModelArtifact(
        family=record["family"],
        version=str(record.get("version", "")),
        taxonomy_version=TAXONOMY_VERSION,
        cleaning_version=CLEANING_VERSION,
        aspects=ASPECTS,
        thresholds=tuple(float(thresholds[code]) for code in ASPECTS),
        files={str(name): str(digest) for name, digest in files.items()},
    )


def read_artifact(model_dir: Path) -> ModelArtifact:
    """`model.json`, checked against this code and against every file's checksum."""
    path = model_dir / MODEL_FILE
    if not path.is_file():
        raise ArtifactError(f"no {MODEL_FILE} in {model_dir}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"{path} isn't valid JSON: {error}") from error
    if not isinstance(record, dict):
        raise ArtifactError(f"{path} isn't a model record")
    artifact = _parse(record)
    for name, expected in artifact.files.items():
        file = model_dir / name
        if not file.is_file():
            raise ArtifactError(f"model file {name} is missing from {model_dir}")
        if sha256_file(file) != expected:
            raise ArtifactError(f"model file {name} doesn't match its checksum")
    return artifact


def load_scorer(model_dir: Path) -> Scorer:
    artifact = read_artifact(model_dir)
    if artifact.family == "s0":
        from patchpulse.models.s0 import S0Scorer  # here, not at the top: s0 imports this module

        return S0Scorer.load(model_dir, artifact)
    if artifact.family == "s1a":
        from patchpulse.models.s1a import S1aScorer  # here, not at the top: s1a imports this module

        return S1aScorer.load(model_dir, artifact)
    if artifact.family == "s1b":
        from patchpulse.models.s1b import S1bScorer  # here, not at the top: s1b imports this module

        return S1bScorer.load(model_dir, artifact)
    raise ArtifactError(f"no scorer for model family {artifact.family!r} yet")
