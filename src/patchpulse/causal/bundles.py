"""Patches close together are estimated as one change, a bundle (design D2).

A patch inside the 7-day window after a bundle's first patch joins it. A bundle gets a 14-day
estimate only when the game's next bundle starts at least 14 days later.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

JOIN_DAYS = 7
LONG_HORIZON = 14


@dataclass(frozen=True)
class Patch:
    patch_id: str
    appid: int
    day: date


@dataclass(frozen=True)
class Bundle:
    appid: int
    start: date
    patch_ids: tuple[str, ...]
    allow_14: bool

    @property
    def bundle_id(self) -> str:
        return self.patch_ids[0]


def bundle_patches(patches: Sequence[Patch]) -> list[Bundle]:
    """Each game's patches grouped into bundles, ordered by game and start."""
    by_game: dict[int, list[Patch]] = defaultdict(list)
    for patch in patches:
        by_game[patch.appid].append(patch)
    bundles: list[Bundle] = []
    for appid in sorted(by_game):
        groups: list[list[Patch]] = []
        for patch in sorted(by_game[appid], key=lambda p: (p.day, p.patch_id)):
            if groups and (patch.day - groups[-1][0].day).days < JOIN_DAYS:
                groups[-1].append(patch)
            else:
                groups.append([patch])
        for index, group in enumerate(groups):
            start = group[0].day
            following = groups[index + 1][0].day if index + 1 < len(groups) else None
            bundles.append(
                Bundle(
                    appid=appid,
                    start=start,
                    patch_ids=tuple(p.patch_id for p in group),
                    allow_14=following is None or (following - start).days >= LONG_HORIZON,
                )
            )
    return bundles


def window_complete(bundle: Bundle, horizon: int, last_complete_day: date) -> bool:
    """Whether the `horizon` days from the bundle's start (its first day included) have ended."""
    return bundle.start + timedelta(days=horizon - 1) <= last_complete_day
