"""The drift reference (Phase 7 design D2): the champion's training data, as numbers and ids.

`models/reference.json` sits next to `models/champion.yaml` and changes with it: a promotion PR
replaces both. It holds the language mix, the length bins, the champion's predicted aspect shares
and its share of uncertain probabilities on its training set, its thresholds, its agreement with the
teacher on the silver holdout, and 1,000 review ids whose e5 vectors stand for the training data in
the embedding test. No text.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from patchpulse.labeling.taxonomy import ASPECTS

REFERENCE_FILE = Path("models") / "reference.json"
REFERENCE_SAMPLE = 1000
REFERENCE_SEED = 20261117  # drift.DRIFT_SEED; drift imports this module, so it's repeated here
LANGUAGE_TOP = 8  # drift.LANGUAGE_TOP, for the same reason


class Reference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_version: str  # the champion's model.json version
    languages_top: list[str]
    language_shares: dict[str, float]
    length_shares: dict[str, float]
    aspect_shares: dict[str, float]
    uncertain_share: float
    thresholds: dict[str, float]
    teacher_agreement_macro_f1: float
    sample_ids: list[int]
    n: int


def build_reference(
    *,
    model_version: str,
    languages: Sequence[str],
    lengths: Sequence[int],
    labels: NDArray[np.bool_],
    probs: NDArray[np.float64],
    thresholds: NDArray[np.float64],
    ids: Sequence[int],
    holdout_agreement: float,
    seed: int = REFERENCE_SEED,
) -> Reference:
    from patchpulse.monitoring import drift  # here: drift imports this module

    top = [lang for lang, _ in Counter(languages).most_common(LANGUAGE_TOP)]
    rng = np.random.default_rng(seed)
    pool = np.asarray(sorted(int(i) for i in ids))
    sample = (
        pool if len(pool) <= REFERENCE_SAMPLE else rng.choice(pool, REFERENCE_SAMPLE, replace=False)
    )
    return Reference(
        model_version=model_version,
        languages_top=top,
        language_shares=drift.language_shares(languages, top),
        length_shares=drift.length_shares(lengths),
        aspect_shares=drift.aspect_shares(labels),
        uncertain_share=drift.uncertain_share(probs, np.asarray(thresholds, dtype=float)),
        thresholds={code: float(t) for code, t in zip(ASPECTS, thresholds, strict=True)},
        teacher_agreement_macro_f1=float(holdout_agreement),
        sample_ids=sorted(int(i) for i in sample),
        n=len(ids),
    )


def read_reference(path: Path = REFERENCE_FILE) -> Reference:
    return Reference.model_validate_json(path.read_text(encoding="utf-8"))


def write_reference(reference: Reference, path: Path = REFERENCE_FILE) -> None:
    text = json.dumps(reference.model_dump(), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
