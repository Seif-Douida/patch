"""The classification suite (Phase 6 design D11): the champion aspect student on gold test, in CI.

Gold test is Phase 2's 450 test reviews with Claude's audited labels. Their raw text goes through
the scorer's `predict_proba` and `labels`, as the nightly scoring stage and Phase 2's report do,
and macro-F1 is computed overall and per language the same way, so the real champion scores
Phase 2's 0.614. The text travels only inside the encrypted eval bundle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from patchpulse.labeling.gold import format_aspects, parse_aspects, read_labels
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.artifact import load_scorer
from patchpulse.models.metrics import macro_f1
from patchpulse.models.snapshot import Snapshot

GOLD_COLUMNS = ("review_id", "text", "language", "aspects")
LANGUAGES = ("english", "french")


class AspectScorer(Protocol):
    """What the suite needs of a student: `models.artifact.Scorer` without its record."""

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]: ...

    def labels(self, probs: NDArray[np.float64]) -> NDArray[np.bool_]: ...


def export_gold_test(sample_csv: Path, labels_csv: Path, snapshot: Snapshot) -> pd.DataFrame:
    """Gold test's reviews by id: raw text, language and labels (`a|b` or `none`)."""
    sample = pd.read_csv(sample_csv)
    test = sample[sample["split"] == "test"]
    languages = dict(zip(test["review_id"].astype(int), test["language"].astype(str), strict=True))
    ids = sorted(languages)
    labels = read_labels(labels_csv)
    texts = snapshot.reviews.set_index("review_id")["text"]
    missing = [rid for rid in ids if rid not in labels or rid not in texts.index]
    if missing:
        raise ValueError(
            f"{len(missing)} gold test reviews have no label or text, e.g. {missing[:3]}"
        )
    return pd.DataFrame(
        {
            "review_id": ids,
            "text": [str(texts[rid]) for rid in ids],
            "language": [languages[rid] for rid in ids],
            "aspects": [format_aspects(labels[rid]) for rid in ids],
        },
        columns=list(GOLD_COLUMNS),
    )


@dataclass(frozen=True)
class ClassificationScores:
    macro_f1: float
    english: float | None
    french: float | None
    n: int


def score_classification(scorer: AspectScorer, gold: pd.DataFrame) -> ClassificationScores:
    truth = np.array(
        [[code in parse_aspects(cell) for code in ASPECTS] for cell in gold["aspects"]], dtype=bool
    )
    predicted = np.asarray(scorer.labels(scorer.predict_proba([str(t) for t in gold["text"]])))
    languages = np.asarray(gold["language"].astype(str))
    by_language = {
        language: macro_f1(truth[rows], predicted[rows]) if (rows := languages == language).any()
        else None
        for language in LANGUAGES
    }  # fmt: skip
    return ClassificationScores(
        macro_f1=macro_f1(truth, predicted),
        english=by_language["english"],
        french=by_language["french"],
        n=len(gold),
    )


def run_classification(model_dir: Path, gold: pd.DataFrame) -> ClassificationScores:
    """The student in `model_dir` on gold test."""
    return score_classification(load_scorer(model_dir), gold)
