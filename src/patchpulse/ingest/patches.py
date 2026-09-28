"""Rules-first patch classifier (spec §6.1). Version `rules-v1`.

Rules, first match wins:

0. Only official posts (`steam_community_announcements`) can be patches; GetNewsForApp mixes in
   press articles, which mention "hotfix" and "update" too.
1. Beta-branch builds are not public patches.
2. "Hotfix" means `hotfix`.
3. Major-update signals: "major update", "remaster(ed)", a named update "out now", or a version
   whose minor parts are all zero (7.0, 7.0.0).
4. The `patchnotes` tag, a version number, or the word "patch" means `patch`.
5. Sales, merchandise, awards and sales milestones mean `marketing`; anniversaries, festivals
   and events mean `event`.
6. Anything else is `other` and flagged ambiguous; Phase 2's LLM fallback revisits those.

Only `major_update` and `patch` are treatment candidates for the causal analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from patchpulse.ingest.models import NewsItem

CLASSIFIER_VERSION = "rules-v1"
OFFICIAL_FEED = "steam_community_announcements"


class PatchType(StrEnum):
    MAJOR_UPDATE = "major_update"
    PATCH = "patch"
    HOTFIX = "hotfix"
    EVENT = "event"
    MARKETING = "marketing"
    OTHER = "other"


TREATMENT_TYPES = frozenset({PatchType.MAJOR_UPDATE, PatchType.PATCH})


@dataclass(frozen=True)
class Classification:
    patch_type: PatchType
    rule: str
    ambiguous: bool = False

    @property
    def is_treatment_candidate(self) -> bool:
        return self.patch_type in TREATMENT_TYPES


_VERSION = re.compile(r"\b[vV]?(\d+)\.(\d+)(?:\.(\d+))?(?:[a-z]\d+)?\b")
_BETA = re.compile(r"\bbeta\b", re.IGNORECASE)
_HOTFIX = re.compile(r"\bhot-?fix", re.IGNORECASE)
_MAJOR = re.compile(r"\bmajor update\b|\bremaster", re.IGNORECASE)
_UPDATE_OUT_NOW = re.compile(
    r"\bupdate\b.*\b(out now|is live|now live|available now)\b", re.IGNORECASE
)
_PATCH_WORD = re.compile(r"\bpatch\b", re.IGNORECASE)
_MARKETING = re.compile(
    r"\bsale\b|\bsave \d+%|\d+% off|\bdiscount|\bmerch|\bawards?\b|\bvote\b|copies sold"
    r"|\btrailer\b|\bwishlist|\bpre-?order|\bbundle\b",
    re.IGNORECASE,
)
_EVENT = re.compile(
    r"\banniversary\b|\bfestival\b|\bevent\b|\bcelebrat|\bexpedition\b|\btournament\b",
    re.IGNORECASE,
)


def _is_major_version(title: str) -> bool:
    for match in _VERSION.finditer(title):
        major, minor, patch = match.group(1), match.group(2), match.group(3)
        if int(major) >= 1 and int(minor) == 0 and (patch is None or int(patch) == 0):
            return True
    return False


def classify(item: NewsItem) -> Classification:
    title = item.title
    if item.feedname != OFFICIAL_FEED:
        return Classification(PatchType.OTHER, "not_official")
    if _BETA.search(title):
        return Classification(PatchType.OTHER, "beta_branch")
    if _HOTFIX.search(title):
        return Classification(PatchType.HOTFIX, "hotfix_keyword")
    if _MAJOR.search(title) or _UPDATE_OUT_NOW.search(title) or _is_major_version(title):
        return Classification(PatchType.MAJOR_UPDATE, "major_update_signal")
    if "patchnotes" in item.tags or _VERSION.search(title) or _PATCH_WORD.search(title):
        return Classification(PatchType.PATCH, "patch_signal")
    if _MARKETING.search(title):
        return Classification(PatchType.MARKETING, "marketing_keyword")
    if _EVENT.search(title):
        return Classification(PatchType.EVENT, "event_keyword")
    return Classification(PatchType.OTHER, "no_rule", ambiguous=True)
