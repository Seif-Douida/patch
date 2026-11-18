"""Phase 7 Task 10: the promotion rule (design D5), fixed in config/retrain.yaml before any run."""

from __future__ import annotations

import inspect

import numpy as np
from numpy.typing import NDArray

from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.promotion import (
    Decision,
    PairedSet,
    PromotionRule,
    decide_promotion,
    load_rule,
    render_table,
)

RULE = load_rule()


def truth(n: int, seed: int) -> NDArray[np.bool_]:
    rng = np.random.default_rng(seed)
    y = rng.random((n, len(ASPECTS))) < 0.3
    y[np.arange(n), rng.integers(0, len(ASPECTS), n)] = True
    return y


def noisy(y: NDArray[np.bool_], rate: float, seed: int) -> NDArray[np.bool_]:
    return y ^ (np.random.default_rng(seed).random(y.shape) < rate)


def block(n: int, language: str, *, champion: float, challenger: float, seed: int) -> PairedSet:
    y = truth(n, seed)
    return PairedSet(
        y_true=y,
        champion=noisy(y, champion, seed + 1),
        challenger=noisy(y, challenger, seed + 2),
        languages=[language] * n,
    )


def joined(*sets: PairedSet) -> PairedSet:
    return PairedSet(
        y_true=np.vstack([s.y_true for s in sets]),
        champion=np.vstack([s.champion for s in sets]),
        challenger=np.vstack([s.challenger for s in sets]),
        languages=[lang for s in sets for lang in s.languages],
    )


CLEAR = block(500, "english", champion=0.25, challenger=0.05, seed=1)


def decide(
    recent: PairedSet = CLEAR,
    *,
    dev: tuple[float, float] = (0.60, 0.61),
    speed: float = 1500.0,
    rule: PromotionRule = RULE,
) -> Decision:
    return decide_promotion(
        recent, dev_champion=dev[0], dev_challenger=dev[1], reviews_per_second=speed, rule=rule
    )


def failed(decision: Decision) -> list[str]:
    return [check.name for check in decision.checks if not check.passed]


def test_the_rule_file_holds_the_designs_values() -> None:
    assert (
        PromotionRule(
            margin=0.01,
            language_drop=0.02,
            language_min_reviews=100,
            dev_drop=0.02,
            min_reviews_per_second=70,
            bootstrap_seed=20261117,
        )
        == RULE
    )


def test_a_clear_gain_on_every_check_promotes() -> None:
    decision = decide()

    assert decision.promote
    assert [check.name for check in decision.checks] == [
        "recent_macro_f1",
        "recent_by_language",
        "gold_dev_macro_f1",
        "reviews_per_second",
    ]
    assert failed(decision) == []


def test_a_gain_under_the_margin_does_not_promote() -> None:
    # A real gain (its interval well above 0) that the margin still asks more of.
    decision = decide(rule=RULE.model_copy(update={"margin": 0.9}))

    recent = decision.checks[0]
    assert recent.challenger is not None
    assert recent.champion is not None
    assert recent.challenger > recent.champion
    assert not decision.promote
    assert failed(decision) == ["recent_macro_f1"]


def test_a_gain_whose_interval_touches_zero_does_not_promote() -> None:
    # 30 reviews, the challenger right on one review the champion gets all wrong: a gain far over
    # the margin, but a third of the resamples don't draw that review, so the lower bound is 0.
    y = truth(30, seed=5)
    champion = y.copy()
    champion[0] = ~y[0]
    decision = decide(PairedSet(y_true=y, champion=champion, challenger=y, languages=["en"] * 30))

    recent = decision.checks[0]
    assert recent.challenger is not None
    assert recent.champion is not None
    assert recent.challenger - recent.champion >= RULE.margin
    assert not decision.promote
    assert failed(decision) == ["recent_macro_f1"]


def test_a_language_drop_blocks_promotion() -> None:
    english = block(400, "english", champion=0.25, challenger=0.03, seed=11)
    french = block(150, "french", champion=0.03, challenger=0.12, seed=12)

    decision = decide(joined(english, french))

    assert not decision.promote
    assert failed(decision) == ["recent_by_language"]
    assert "french" in decision.checks[1].detail


def test_a_language_with_under_100_reviews_is_ignored() -> None:
    english = block(400, "english", champion=0.25, challenger=0.03, seed=11)
    german = block(99, "german", champion=0.03, challenger=0.20, seed=13)

    decision = decide(joined(english, german))

    assert decision.promote
    assert "german" not in decision.checks[1].detail


def test_a_dev_drop_blocks_promotion() -> None:
    assert failed(decide(dev=(0.65, 0.62))) == ["gold_dev_macro_f1"]
    assert decide(dev=(0.65, 0.63)).promote  # exactly 0.02 lower is allowed


def test_a_slow_challenger_does_not_promote() -> None:
    assert failed(decide(speed=69.0)) == ["reviews_per_second"]


def test_no_recent_labels_never_promote() -> None:
    empty = np.zeros((0, len(ASPECTS)), dtype=bool)

    decision = decide(PairedSet(y_true=empty, champion=empty, challenger=empty, languages=[]))

    assert not decision.promote
    assert failed(decision) == ["recent_macro_f1"]


def test_gold_test_is_reported_and_never_decides() -> None:
    # The decision takes no gold-test input at all; the table shows it, marked as reported only.
    assert not any("test" in name for name in inspect.signature(decide_promotion).parameters)
    decision = decide()

    table = render_table(
        decision,
        champion_version="20261012T200000Z",
        challenger_version="20261201T030000Z",
        test_champion=0.80,
        test_challenger=0.60,
        counts={"train": 27000, "recent": 500},
    )

    assert decision.promote  # a worse gold-test score with every check passing still promotes
    assert "| gold test macro-F1 (reported only) | 0.800 | 0.600 |" in table
    assert "promote" in table
    assert "27000" in table
