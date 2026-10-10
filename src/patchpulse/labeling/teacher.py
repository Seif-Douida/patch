"""The teacher: an LLM labels reviews by aspect (plan Task 8, spec §7.3).

- **Prompt:** the system prompt is `prompts/v1.md`, which is public and uses only invented
  examples. `prompt_hash()` identifies it, with the cleaning version, in the label cache.
- **Envelope:** reviews are cleaned (`clean_text`) and sent as a JSON array, at most 20 at a time,
  each under a short alias so the model never echoes long ids. JSON escaping means no review can
  close the array or pass for an instruction outside it.
- **Validation:** an answer must hold exactly the batch's aliases and only known codes. An invalid
  answer gets one repair request quoting the error. If that fails too, each review goes alone, and a
  review that never validates is skipped, never guessed.
- **Resuming:** `label_reviews` skips reviews already cached for this model and prompt, writes after
  every batch, and stops cleanly when the daily quota runs out or Google stays unavailable.

Nothing here logs review text: only ids, counts and validation errors.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from patchpulse.labeling.cache import LabelCache
from patchpulse.labeling.gemini import GeminiError, GeminiResponse, GeminiUnavailableError
from patchpulse.labeling.limits import QuotaExhaustedError
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.snapshot import Snapshot
from patchpulse.models.text import CLEANING_VERSION, clean_text

log = logging.getLogger("patchpulse.labeling.teacher")

PROMPT_VERSION = "v1"
MAX_REVIEWS = 20
MAX_INPUT_TOKENS = 4000  # for the reviews; the system prompt comes on top
_CHARS_PER_TOKEN = 3  # cautious; Google's real count replaces it in the limiter
_OFF_TOPIC = "off_topic"
PROGRESS_EVERY = 1000  # reviews between progress lines
# Precedes the reviews' JSON array, which always ends the user message.
REVIEWS_MARKER = "Reviews (JSON array):\n"

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "alias": {"type": "integer"},
                    "aspects": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(ASPECTS)},
                    },
                },
                "required": ["alias", "aspects"],
            },
        }
    },
    "required": ["labels"],
}


class AnswerError(ValueError):
    """The model's answer isn't valid for its batch; the message says why (no review text)."""


class TextModel(Protocol):
    def generate(
        self,
        system: str,
        user: str,
        *,
        json_schema: Mapping[str, Any] | None,
        expected_output_tokens: int = ...,
        max_output_tokens: int = ...,
    ) -> GeminiResponse: ...


@dataclass(frozen=True)
class ReviewIn:
    review_id: int
    game: str
    voted_up: bool
    text: str


def review_inputs(snapshot: Snapshot, review_ids: Sequence[int]) -> list[ReviewIn]:
    """The teacher's view of snapshot reviews, in the order of `review_ids`."""
    reviews = snapshot.reviews.set_index("review_id")
    names = snapshot.games.set_index("appid")["name"]
    return [
        ReviewIn(
            review_id=int(i),
            game=str(names[reviews.at[i, "appid"]]),
            voted_up=bool(reviews.at[i, "voted_up"]),
            text=str(reviews.at[i, "text"]),
        )
        for i in review_ids
    ]


@dataclass(frozen=True)
class Batch:
    review_ids: tuple[int, ...]  # alias n is review_ids[n - 1]
    payload: str  # the JSON array the model sees
    input_tokens: int  # estimated, for the payload


@dataclass(frozen=True)
class TeacherLabel:
    review_id: int
    aspects: frozenset[str]


@dataclass(frozen=True)
class LabelRun:
    labeled: int
    cached: int
    remaining: int
    stopped_for_quota: bool
    stopped_for_outage: bool = False  # Google kept failing; rerun later


class _Item(BaseModel):
    model_config = ConfigDict(extra="ignore")

    alias: int
    aspects: list[str]


class _Answer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    labels: list[_Item]


@functools.cache
def _prompt_text(version: str) -> str:
    return files("patchpulse.labeling").joinpath(f"prompts/{version}.md").read_text("utf-8")


def load_prompt() -> str:
    return _prompt_text(PROMPT_VERSION)


def prompt_hash() -> str:
    """Changes with the prompt's text or the cleaning, so either starts a fresh label cache."""
    identity = f"{load_prompt()}\n--\ncleaning {CLEANING_VERSION}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _estimate(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN + 1


def _payload(items: Sequence[Mapping[str, Any]]) -> str:
    numbered = [{"alias": alias, **item} for alias, item in enumerate(items, start=1)]
    return json.dumps(numbered, ensure_ascii=False)


def make_batches(
    reviews: Iterable[ReviewIn],
    *,
    max_reviews: int = MAX_REVIEWS,
    max_input_tokens: int = MAX_INPUT_TOKENS,
) -> list[Batch]:
    batches: list[Batch] = []
    ids: list[int] = []
    items: list[dict[str, Any]] = []

    def flush() -> None:
        if items:
            payload = _payload(items)
            batches.append(Batch(tuple(ids), payload, _estimate(payload)))
            ids.clear()
            items.clear()

    for review in reviews:
        item = {
            "game": review.game,
            "thumbs": "up" if review.voted_up else "down",
            "text": clean_text(review.text),
        }
        too_many = len(items) == max_reviews
        if items and (too_many or _estimate(_payload([*items, item])) > max_input_tokens):
            flush()
        ids.append(review.review_id)
        items.append(item)
    flush()
    return batches


def parse_answer(text: str, aliases: Sequence[int]) -> dict[int, frozenset[str]]:
    """Validate an answer against its batch. Code fences and text around the object are ignored."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise AnswerError("the answer isn't valid JSON: it holds no JSON object")
    try:
        data = json.loads(text[start : end + 1])
    except ValueError as error:
        raise AnswerError(f"the answer isn't valid JSON: {error}") from None
    try:
        answer = _Answer.model_validate(data)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}"
            for issue in error.errors(include_input=False)
        )
        raise AnswerError(f"the answer doesn't match the format: {problems}") from None
    given = [item.alias for item in answer.labels]
    repeated = sorted({alias for alias in given if given.count(alias) > 1})
    if repeated:
        raise AnswerError(f"aliases given more than once: {repeated}")
    missing, extra = sorted(set(aliases) - set(given)), sorted(set(given) - set(aliases))
    if missing or extra:
        mismatches = [f"missing aliases {missing}"] if missing else []
        mismatches += [f"unexpected aliases {extra}"] if extra else []
        raise AnswerError("; ".join(mismatches))
    unknown = sorted({code for item in answer.labels for code in item.aspects} - set(ASPECTS))
    if unknown:
        raise AnswerError(f"unknown aspect codes {unknown}; use only {', '.join(ASPECTS)}")
    labels: dict[int, frozenset[str]] = {}
    for item in answer.labels:
        codes = frozenset(item.aspects)
        if _OFF_TOPIC in codes and len(codes) > 1:  # exclusive: the named aspects win
            codes -= {_OFF_TOPIC}
        labels[item.alias] = codes
    return labels


def _user_message(batch: Batch) -> str:
    count = len(batch.review_ids)
    return f"Label these {count} reviews. {REVIEWS_MARKER}{batch.payload}"


def _ask(client: TextModel, batch: Batch) -> dict[int, frozenset[str]]:
    """One request and, if its answer is invalid, one repair request quoting the error."""
    aliases = list(range(1, len(batch.review_ids) + 1))
    system, user = load_prompt(), _user_message(batch)
    expected = 30 + 15 * len(aliases)
    response = client.generate(
        system, user, json_schema=ANSWER_SCHEMA, expected_output_tokens=expected
    )
    try:
        return parse_answer(response.text, aliases)
    except AnswerError as error:
        repair = (
            f"Your previous answer was rejected: {error}\n"
            f"Answer again, with exactly one entry for each alias from 1 to {len(aliases)}.\n\n"
            f"{user}"
        )
        response = client.generate(
            system, repair, json_schema=ANSWER_SCHEMA, expected_output_tokens=expected
        )
        return parse_answer(response.text, aliases)


def _singles(batch: Batch) -> list[Batch]:
    items = json.loads(batch.payload)
    singles = []
    for review_id, item in zip(batch.review_ids, items, strict=True):
        payload = _payload([{key: value for key, value in item.items() if key != "alias"}])
        singles.append(Batch((review_id,), payload, _estimate(payload)))
    return singles


def label_batch(client: TextModel, batch: Batch) -> list[TeacherLabel]:
    """Labels for the batch's reviews; a review that never validates is left out."""
    try:
        labels = _ask(client, batch)
    except (AnswerError, GeminiError) as error:
        if len(batch.review_ids) == 1:
            log.warning("review %d skipped: %s", batch.review_ids[0], error)
            return []
        log.info("a batch of %d failed (%s); labeling each alone", len(batch.review_ids), error)
        return [label for single in _singles(batch) for label in label_batch(client, single)]
    return [
        TeacherLabel(batch.review_ids[alias - 1], aspects)
        for alias, aspects in sorted(labels.items())
    ]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def label_reviews(
    client: TextModel,
    cache: LabelCache,
    reviews: Iterable[ReviewIn],
    *,
    model: str,
    clock: Callable[[], datetime] = _utc_now,
) -> LabelRun:
    digest = prompt_hash()
    reviews = list(reviews)
    missing = set(cache.missing([r.review_id for r in reviews], model=model, prompt_hash=digest))
    todo = [r for r in reviews if r.review_id in missing]
    cached = len(reviews) - len(todo)
    labeled = 0
    next_mark = PROGRESS_EVERY
    batches = make_batches(todo)
    for number, batch in enumerate(batches, start=1):
        try:
            labels = label_batch(client, batch)
        except QuotaExhaustedError as error:
            log.warning("stopped after %d of %d batches: %s", number - 1, len(batches), error)
            return LabelRun(labeled, cached, len(todo) - labeled, stopped_for_quota=True)
        except GeminiUnavailableError as error:
            log.warning(
                "stopped after %d of %d batches: Google unavailable (%s)",
                number - 1,
                len(batches),
                error,
            )
            return LabelRun(
                labeled,
                cached,
                len(todo) - labeled,
                stopped_for_quota=False,
                stopped_for_outage=True,
            )
        cache.put_many(
            {label.review_id: label.aspects for label in labels},
            model=model,
            prompt_hash=digest,
            created_at=clock(),
        )
        labeled += len(labels)
        log.debug(
            "batch %d of %d: %d of %d labeled",
            number,
            len(batches),
            len(labels),
            len(batch.review_ids),
        )
        while labeled >= next_mark:
            log.info(
                "%s of %s labeled so far (%s were cached)",
                f"{next_mark:,}",
                f"{len(todo):,}",
                f"{cached:,}",
            )
            next_mark += PROGRESS_EVERY
    return LabelRun(labeled, cached, len(todo) - labeled, stopped_for_quota=False)
