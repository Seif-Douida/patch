"""Rules-first patch classifier (spec §6.1). Version `rules-v2` (8 Nov 2026).

A treatment candidate is a public release of a game update, on its release day. Rules, first
match wins; only the title, the tags, the feed and the game are read, never the post's body:

0. Only official posts (`steam_community_announcements`) can be patches; GetNewsForApp mixes in
   press articles, which mention "hotfix" and "update" too.
1. Test builds are not public patches: beta, stress test, playtest, public test, a test server
   or branch, experimental.
2. "Hotfix" means `hotfix`.
3. Announcements are not releases: a post that something is coming, upcoming or incoming, an
   announcement, preview, showcase, maintenance notice, delay or postponement, a dev diary,
   livestream or "releasing", "an update on/regarding/about", plans, a roadmap, or a date it
   will be available, launch or arrive. Such posts made false patches or started real ones days
   early (Wilds' Title Update 3 showcase, six days before it shipped).
4. A studio's own release-title style, for the tracked games whose release posts no general
   rule recognises (STUDIO_RELEASE_TITLES), means `major_update`.
5. Major-update signals: "major update", "remaster(ed)", an update "out now", or a version whose
   minor parts are all zero (7.0, 7.0.0).
6. The `patchnotes` tag, a version number, or the word "patch" means `patch`.
7. Sales, merchandise, awards and sales milestones mean `marketing`; anniversaries, festivals
   and events mean `event`.
8. Anything else is `other` and flagged ambiguous (the spec's LLM fallback isn't built).

A version number counts only when it reads as a build: three or more parts (6.0.3, 01.002.105),
a build suffix (1.5.4f1, 5.00c), or marked as a version (v, ver., version, update or patch just
before it; a colon, bracket or any dash just before it; or opening the title). A bare two-part
number after a word is a name ("Economy 2.0", "Photo Mode Challenge 3.0"), and a count is not a
build ("1.2M copies sold", "1.5x rewards", "1.5 million players").

A general "out now" or "Introducing" rule was tried and dropped: on the comparison games it
marked other games' launches, soundtracks, DLC and merchandise as patches (8 Nov 2026).

Only `major_update` and `patch` are treatment candidates for the causal analysis. Changing a
rule or a studio's style changes the version: the nightly reclassifies every stored post once.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from patchpulse.ingest.models import NewsItem

CLASSIFIER_VERSION = "rules-v2"
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


# The tracked games whose studios title their release posts in a way no general rule can read
# (each checked on the post bodies, 8 Nov 2026).
STUDIO_RELEASE_TITLES: Mapping[int, re.Pattern[str]] = {
    # No Man's Sky: "Introducing...No Man's Sky: RELICS" is "available to download free ... from
    # today"; an "Introducing..." Expedition is a timed event.
    275850: re.compile(r"^introducing\b(?!.*\bexpedition\b)", re.IGNORECASE),
    # Helldivers 2: "Into the Unjust - Out Now!" (2 Sep 2025); a Warbond is a paid pack.
    553850: re.compile(r"^(?!.*\bwarbond\b).*\bout now\b", re.IGNORECASE),
    # Deep Rock Galactic: "Season 06: 'Relics of Hoxxes' is NOW LIVE!".
    548430: re.compile(r"^season \d+\b.*\b(now live|is live|live now)\b", re.IGNORECASE),
}

_VERSION = re.compile(r"\b[vV]?(\d+)\.(\d+)(?:\.(\d+))?(?:[a-z]\d+)?\b")
# A build's suffix is lowercase: a letter and digits (1.5.4f1) or one letter (5.00c), never k, m or
# x, which make counts and multipliers (4.5k, 1.2M, 1.5x); a number before million, thousand or
# billion is a count too. Any dash marks a version: Steam titles often use en and em dashes.
_SUFFIX = r"(?:[a-z]\d+|[a-jln-wyz])"
_END = r"(?![\w%])(?!\s*(?i:million|thousand|billion)\b)"
_BUILD = re.compile(
    r"(?:^|[:(\[]\s*|[-\u2013\u2014]\s*|\b(?i:v|ver\.?|version|update|patch)\s*)"
    rf"(?i:v)?\d+\.\d+(?:\.\d+)*{_SUFFIX}?{_END}"
    rf"|\b\d+\.\d+\.\d+(?:\.\d+)*{_SUFFIX}?{_END}"
    rf"|\b\d+\.\d+{_SUFFIX}{_END}"
)
_TEST_BUILD = re.compile(
    r"\bbeta\b|\bstress test|\bplaytest|\bpublic test|\btest (?:server|branch)|\bexperimental\b",
    re.IGNORECASE,
)
_HOTFIX = re.compile(r"\bhot-?fix", re.IGNORECASE)
_DAYS = (
    "monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may"
    "|june|july|august|september|october|november|december"
)
_ANNOUNCEMENT = re.compile(
    r"\bcoming\b|\bupcoming\b|\bincoming\b|\bannounc|\bpreview\b|\bshowcase\b|\bmaintenance\b"
    r"|\bdelay|\bpostpone|\bdev(?:eloper)? diary|\blivestream|\breleasing\b"
    r"|\ban update (?:on|regarding|about)\b|\bplans\b|\broadmap\b"
    rf"|\b(?:available|launches|arrives)\s+(?:on\s+)?(?:{_DAYS})\b",
    re.IGNORECASE,
)
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
    for build in _BUILD.finditer(title):
        match = _VERSION.search(build.group(0))
        if match is None:
            continue
        major, minor, patch = match.group(1), match.group(2), match.group(3)
        if int(major) >= 1 and int(minor) == 0 and (patch is None or int(patch) == 0):
            return True
    return False


def classify(item: NewsItem) -> Classification:
    title = item.title
    if item.feedname != OFFICIAL_FEED:
        return Classification(PatchType.OTHER, "not_official")
    if _TEST_BUILD.search(title):
        return Classification(PatchType.OTHER, "test_build")
    if _HOTFIX.search(title):
        return Classification(PatchType.HOTFIX, "hotfix_keyword")
    if _ANNOUNCEMENT.search(title):
        return Classification(PatchType.OTHER, "announcement")
    studio = STUDIO_RELEASE_TITLES.get(item.appid)
    if studio is not None and studio.search(title):
        return Classification(PatchType.MAJOR_UPDATE, "studio_release_title")
    if _MAJOR.search(title) or _UPDATE_OUT_NOW.search(title) or _is_major_version(title):
        return Classification(PatchType.MAJOR_UPDATE, "major_update_signal")
    if "patchnotes" in item.tags or _BUILD.search(title) or _PATCH_WORD.search(title):
        return Classification(PatchType.PATCH, "patch_signal")
    if _MARKETING.search(title):
        return Classification(PatchType.MARKETING, "marketing_keyword")
    if _EVENT.search(title):
        return Classification(PatchType.EVENT, "event_keyword")
    return Classification(PatchType.OTHER, "no_rule", ambiguous=True)
