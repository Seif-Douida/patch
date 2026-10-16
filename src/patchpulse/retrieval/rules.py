"""The index rules (design D1), as dbt's `core.review_index` applies them (`dbt_project.yml` vars),
for the loader and the reports. A test keeps the two equal."""

from __future__ import annotations

# Long enough to be useful: 20 words, or 40 characters in the languages that don't put spaces
# between words (a 20-word English review has a median of 107 characters).
MIN_WORDS = 20
CHAR_LANGUAGES = ("schinese", "tchinese", "japanese", "thai")
MIN_CHARS = 40
# Spec §6.3: each game's 40,000 most recent eligible reviews, plus every eligible review within
# 30 days either side of a treatment-candidate patch or of the game's launch.
CAP_PER_GAME = 40_000
PATCH_WINDOW_DAYS = 30
INDEX_RULES_VERSION = "r1"


def eligible(language: str, text: str) -> bool:
    """Long enough to index, as `core.review_index` decides it. Words are counted by Python's
    whitespace split, which differs slightly from SQL's count of spaces (runs of spaces, tabs):
    for estimates on the snapshot only; the index set itself comes from dbt."""
    if language in CHAR_LANGUAGES:
        return len(text.rstrip()) >= MIN_CHARS
    return len(text.split()) >= MIN_WORDS
