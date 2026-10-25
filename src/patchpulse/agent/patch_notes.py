"""get_patch_notes (design D3): a patch's own post from Steam's public news API, read when the
model asks, cleaned and cut, and never republished.

GetNewsForApp is asked for the few posts up to the patch's timestamp and the post is matched by
its news item id; a post Steam no longer returns is "not found", a result rather than an error.
The request goes through the shared SteamHttp, so the 2.25-second limiter paces it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from patchpulse.ingest.http import SteamHttp
from patchpulse.ingest.news import NEWS_URL
from patchpulse.models.text import clean_text

PATCH_NOTES_CHARS = 3000
NEWS_COUNT = 5  # posts up to the patch's timestamp; the patch's own is among the latest
_HTML = re.compile(r"<[a-zA-Z/!][^>]*>")  # tags, and clean_text's <url> placeholders
_CLAN_IMAGE = re.compile(r"\{STEAM_CLAN_IMAGE\}\S*")
_UNCUT = 10**9


@dataclass(frozen=True)
class Notes:
    found: bool
    text: str | None
    truncated: bool


def clean_notes(contents: str) -> tuple[str, bool]:
    """The post's text without BBCode, HTML, images or URLs, cut to PATCH_NOTES_CHARS; and
    whether it was cut."""
    text = clean_text(contents, max_chars=_UNCUT)  # BBCode out, URLs as <url>
    text = _CLAN_IMAGE.sub(" ", _HTML.sub(" ", text))
    whole = clean_text(text, max_chars=_UNCUT)
    return clean_text(whole, max_chars=PATCH_NOTES_CHARS), len(whole) > PATCH_NOTES_CHARS


def fetch_patch_notes(http: SteamHttp, *, appid: int, gid: str, published_at: datetime) -> Notes:
    """The post `gid` among the game's latest posts up to `published_at`. Raises SteamError when
    Steam doesn't answer."""
    params: dict[str, str | int] = {
        "appid": appid,
        "count": NEWS_COUNT,
        "maxlength": 0,
        "enddate": int(published_at.timestamp()) + 1,
    }
    payload, _ = http.get_json(NEWS_URL, params)
    items = (payload.get("appnews") or {}).get("newsitems") or []
    match = next((i for i in items if str(i.get("gid")) == gid), None)
    if match is None:
        return Notes(found=False, text=None, truncated=False)
    text, truncated = clean_notes(str(match.get("contents") or ""))
    return Notes(found=True, text=text, truncated=truncated)
