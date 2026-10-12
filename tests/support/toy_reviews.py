"""Invented reviews whose aspects follow from their words, for the students' tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from patchpulse.labeling.taxonomy import ASPECTS

WORDS: dict[str, tuple[str, ...]] = {
    "performance": ("fps drops", "stutters", "laggy frames"),
    "stability_bugs": ("crashes", "buggy mess", "glitches everywhere"),
    "gameplay_balance": ("overpowered weapons", "nerfed", "unbalanced builds"),
    "content": ("new maps", "more missions", "endgame grind"),
    "monetization": ("microtransactions", "battle pass", "pay to win"),
    "online_servers": ("server down", "matchmaking queue", "disconnects online"),
    "story_world": ("great story", "lore characters", "beautiful world"),
    "ux_controls": ("clunky controls", "menu ui", "keybinds"),
    "platform_policy": ("refund steam", "region locked", "drm launcher"),
    "off_topic": ("my cat", "hello mom", "pizza recipe"),
}
FILLER = ("honestly", "i think", "after hours", "the game", "overall", "today")


def toy_reviews(n: int, *, seed: int) -> tuple[list[str], NDArray[np.bool_]]:
    """`n` reviews, each about one or two aspects (one in ten about none)."""
    rng = np.random.default_rng(seed)
    texts: list[str] = []
    labels = np.zeros((n, len(ASPECTS)), dtype=bool)
    for row in range(n):
        if row % 10 == 9:
            texts.append(" ".join(rng.choice(FILLER, size=3)))
            continue
        chosen = rng.choice(len(ASPECTS), size=1 + row % 2, replace=False)
        parts = [str(rng.choice(FILLER))]
        for index in chosen:
            labels[row, index] = True
            parts.append(str(rng.choice(WORDS[ASPECTS[index]])))
        texts.append(" ".join(parts))
    return texts, labels
