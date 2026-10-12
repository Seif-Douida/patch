"""S0, the CPU baseline student (plan Task 11, design D5): TF-IDF and one-vs-rest logistic
regression, trained on invented reviews here."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from toy_reviews import toy_reviews

from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.artifact import load_scorer
from patchpulse.models.s0 import S0Params, save_s0, train_s0

TOY = S0Params(min_df=1)


def test_s0_learns_a_toy_multilabel_set() -> None:
    texts, labels = toy_reviews(400, seed=1)
    unseen, truth = toy_reviews(100, seed=2)

    model = train_s0(texts, labels, params=TOY)
    probs = model.predict_proba(unseen)

    assert probs.shape == (100, len(ASPECTS))
    assert ((probs >= 0.5) == truth).mean() > 0.97


def test_labels_must_cover_the_taxonomy() -> None:
    texts, labels = toy_reviews(40, seed=1)

    with pytest.raises(ValueError, match="10 aspects"):
        train_s0(texts, labels[:, :9], params=TOY)


def test_scorer_accepts_odd_reviews(tmp_path: Path) -> None:
    texts, labels = toy_reviews(200, seed=3)
    save_s0(train_s0(texts, labels, params=TOY), tmp_path, thresholds=np.full(10, 0.5), version="t")
    scorer = load_scorer(tmp_path)
    odd = [
        "",
        "   ",
        "\U0001f525\U0001f525\U0001f525\U0001f44d",  # emoji only, written as escapes
        "[h1][b][/b][/h1][list][*][/list][url=https://example.com][/url]",
        "非常好玩的游戏，服务器经常掉线",  # noqa: RUF001 (a Chinese comma, on purpose)
        "Очень тормозит, fps drops " * 400,  # ~10,400 characters, cut to 2,000
        "x" * 8_000,
    ]

    probs = scorer.predict_proba(odd)

    assert probs.shape == (len(odd), len(ASPECTS))
    assert np.isfinite(probs).all()
    assert ((probs >= 0) & (probs <= 1)).all()
    assert scorer.labels(probs).dtype == bool
