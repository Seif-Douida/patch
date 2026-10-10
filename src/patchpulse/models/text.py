"""Review text cleaning, shared by the teacher, training and Azure scoring (plan Task 8).

Standard library only, so the jobs image can score without the `ml` group. Every model sees text
cleaned the same way; `CLEANING_VERSION` changes whenever this does, and it's part of the teacher's
prompt hash and of every trained model's record.

- Steam BBCode tags are removed by name and their inner text kept. Block tags become line breaks;
  inline tags vanish. Brackets that aren't Steam tags ("[x] Good") stay.
- URLs become `<url>`.
- Whitespace is normalised: lines are trimmed, runs of spaces collapse, blank lines go.
- Text longer than `max_chars` is cut at the last space in its second half, or hard-cut.
"""

from __future__ import annotations

import re
import unicodedata

CLEANING_VERSION = "c1"
MAX_CHARS = 2000

_BLOCK_TAGS = "h1|h2|h3|list|olist|\\*|quote|code|table|tr|th|td|hr|previewyoutube"
_INLINE_TAGS = "b|i|u|strike|spoiler|noparse|url|img"
_BLOCK = re.compile(rf"\[/?(?:{_BLOCK_TAGS})(?:=[^\]]*)?\]", re.IGNORECASE)
_INLINE = re.compile(rf"\[/?(?:{_INLINE_TAGS})(?:=[^\]]*)?\]", re.IGNORECASE)
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_SPACES = re.compile(r"[^\S\n]+")


def clean_text(text: str, *, max_chars: int = MAX_CHARS) -> str:
    text = unicodedata.normalize("NFC", text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _BLOCK.sub("\n", text)
    text = _INLINE.sub("", text)
    text = _URL.sub("<url>", text)
    lines = (_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    text = "\n".join(line for line in lines if line)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    boundary = max(cut.rfind(" "), cut.rfind("\n"))
    return cut[:boundary].rstrip() if boundary >= max_chars // 2 else cut
