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
