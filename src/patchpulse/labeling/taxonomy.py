"""The aspect taxonomy (spec §7.1): the one list the app, the teacher, the students and dbt share.

A review can mention zero or more aspects. The order is fixed: it is the column order of every
label matrix and probability array, so changing it is a new taxonomy version.
"""

from __future__ import annotations

TAXONOMY_VERSION = "v1"

ASPECTS: tuple[str, ...] = (
    "performance",
    "stability_bugs",
    "gameplay_balance",
    "content",
    "monetization",
    "online_servers",
    "story_world",
    "ux_controls",
    "platform_policy",
    "off_topic",
)

NONE = "none"  # the label cell of a review with no aspect


def parse_aspects(cell: str) -> frozenset[str]:
    """`performance|content` or `none`; anything else is an error, never a guess."""
    parts = [part.strip() for part in str(cell).split("|") if part.strip()]
    if parts == [NONE]:
        return frozenset()
    if NONE in parts:
        raise ValueError(f"'none' can't be combined with aspect codes: {cell!r}")
    unknown = [part for part in parts if part not in ASPECTS]
    if unknown or not parts:
        raise ValueError(f"unknown aspect code(s) {unknown or [cell]}; known: {', '.join(ASPECTS)}")
    return frozenset(parts)


def format_aspects(aspects: frozenset[str]) -> str:
    return "|".join(code for code in ASPECTS if code in aspects) or NONE
