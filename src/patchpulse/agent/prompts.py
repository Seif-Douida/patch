"""The agent's prompts (design D4, D12): one file per model step in `prompts/`, each with the JSON
schema its answer must fit. A prompt's version is a hash of its text, and PROMPT_VERSION a hash of
all four, recorded on every report; the record/replay cache keys on the prompts themselves, so an
edited prompt is a fresh cache. The files hold only invented examples, never review text.
"""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from patchpulse.agent.report import DRAFT_SCHEMA
from patchpulse.agent.search import PLAYTIME_FLOORS
from patchpulse.labeling.taxonomy import ASPECTS

PROMPT_NAMES = ("scope", "gather", "draft", "summary")
VERSION_CHARS = 12

_STRING = {"type": "string"}

SCOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "appid": {"type": "integer"},
        "start": _STRING,
        "end": _STRING,
        "patch_id": _STRING,
        "refusal": _STRING,
    },
}

GATHER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": ["search_reviews", "get_patch_notes"]},
                    "reason": _STRING,
                    "query": _STRING,
                    "period": {"type": "string", "enum": ["window", "baseline"]},
                    "voted_up": {"type": "boolean"},
                    "aspect": {"type": "string", "enum": list(ASPECTS)},
                    "min_playtime": {"type": "integer", "enum": list(PLAYTIME_FLOORS)},
                    "patch_id": _STRING,
                },
                "required": ["tool", "reason"],
            },
        }
    },
    "required": ["calls"],
}

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"summary": _STRING},
    "required": ["summary"],
}

SCHEMAS: dict[str, dict[str, Any]] = {
    "scope": SCOPE_SCHEMA,
    "gather": GATHER_SCHEMA,
    "draft": DRAFT_SCHEMA,
    "summary": SUMMARY_SCHEMA,
}


@dataclass(frozen=True)
class Prompt:
    text: str
    version: str  # a hash of the text


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:VERSION_CHARS]


@functools.cache
def prompt_texts() -> dict[str, str]:
    root = files("patchpulse.agent").joinpath("prompts")
    return {name: root.joinpath(f"{name}.md").read_text("utf-8") for name in PROMPT_NAMES}


def version_of(texts: Mapping[str, str]) -> str:
    """One version for a set of prompts: it changes when any of them does."""
    return _hash("\n\x00\n".join(f"{name}\n{texts[name]}" for name in sorted(texts)))


def load_prompt(name: str) -> Prompt:
    if name not in PROMPT_NAMES:
        raise ValueError(f"unknown prompt {name!r}; the prompts are {', '.join(PROMPT_NAMES)}")
    text = prompt_texts()[name]
    return Prompt(text=text, version=_hash(text))


PROMPT_VERSION = version_of(prompt_texts())
