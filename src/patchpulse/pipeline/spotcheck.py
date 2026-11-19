"""The nightly teacher spot-check (Phase 7 design D3, spec §14 "Online accuracy proxy").

Each night 200 reviews first seen in that run are labeled by the teacher (prompt v1, the silver
set's batching and validation), and the labels stored in `raw.teacher_label` (source `spotcheck`).
The teacher is Gemma 4 31B on the friend's server, the silver set's own teacher (ADR-020, amended
19 Nov: Google's free copy kept answering 503). When the server is down, Google's copy labels the
rest of that night; each label names the model that gave it.

Each week the champion's labels are compared with the teacher's on that week's spot-check reviews
(the online accuracy proxy), against the same comparison on the silver holdout (the reference).
The labels also train the next challenger: a fixed fifth of reviews, by a hash of the id, is held
out for deciding promotions (Task 10).

The text is read from SQL into memory for the request and never stored or logged.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx2
import numpy as np
from numpy.typing import NDArray
from sqlalchemy import Connection, Engine, bindparam, text

from patchpulse.config import Settings
from patchpulse.labeling.limits import QuotaExhaustedError, RequestLimiter
from patchpulse.labeling.llm import TeacherUnavailableError
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.labeling.teacher import (
    Batch,
    ReviewIn,
    TeacherLabel,
    TextModel,
    label_batch,
    make_batches,
    prompt_hash,
)
from patchpulse.labeling.teachers import load_model_config
from patchpulse.models.metrics import assessable_macro_f1, bootstrap_ci, prf
from patchpulse.monitoring.alerts import AlertRule, Outcome
from patchpulse.monitoring.reference import Reference
from patchpulse.monitoring.store import MetricRow, StoredTeacherLabel, write_teacher_labels

log = logging.getLogger("patchpulse.pipeline.spotcheck")

SPOTCHECK_SIZE = 200
SERVER_MODEL = "selfhosted-gemma-4-31b"  # config/models.yaml: the friend's server (gemma-31b)
GOOGLE_MODEL = "gemma-4-31b"  # config/models.yaml: Google's gemma-4-31b-it, the backup
# A server that's off costs 10 s, not the 5 minutes a busy GPU may take (offline labeling's
# timeout); a 20-review batch takes ~12 s (ADR-020).
SERVER_TIMEOUT = httpx2.Timeout(90.0, connect=10.0)
SPOTCHECK_SOURCE = "spotcheck"
HOLDOUT_SHARE = 0.2
SPOTCHECK_ATTEMPTS = 2  # the second only waits out a per-minute limit; errors raise at once
# An outage (an overloaded Gemma's 503, a timeout) pauses and tries the same batch again, up to
# this many tries in all, while the pause ends before the deadline (the first night, 19 Nov: one
# 503 ended the spot-check with 0 of 200 labeled).
SPOTCHECK_OUTAGE_TRIES = 3
SPOTCHECK_OUTAGE_PAUSE_S = 20.0
AGREEMENT_SEED = 20261117

_CANDIDATES = """
SELECT r.recommendation_id
FROM raw.steam_review AS r
WHERE r.first_seen_run_id = :run
  AND NOT EXISTS (
      SELECT 1 FROM raw.steam_review AS newer
      WHERE newer.recommendation_id = r.recommendation_id
        AND newer.timestamp_updated > r.timestamp_updated)
ORDER BY r.recommendation_id
"""
_PICKED = """
SELECT r.recommendation_id, r.timestamp_updated, r.voted_up, r.review_text, g.name
FROM raw.steam_review AS r
JOIN raw.game AS g ON g.appid = r.appid
WHERE r.first_seen_run_id = :run AND r.recommendation_id IN :ids
  AND NOT EXISTS (
      SELECT 1 FROM raw.steam_review AS newer
      WHERE newer.recommendation_id = r.recommendation_id
        AND newer.timestamp_updated > r.timestamp_updated)
"""
_LABELS = ", ".join(f"t.l_{code}" for code in ASPECTS)
_SCORES = ", ".join(f"s.l_{code}" for code in ASPECTS)
_PAIRS = f"""
SELECT {_LABELS}, {_SCORES}
FROM raw.teacher_label AS t
JOIN raw.review_aspect_score AS s
  ON s.recommendation_id = t.recommendation_id
 AND s.timestamp_updated = t.timestamp_updated
 AND s.model_version = :model
WHERE t.source = 'spotcheck' AND t.created_at >= :start AND t.created_at < :stop
ORDER BY t.recommendation_id
"""  # noqa: S608 (column names from the taxonomy, values bound)
_TIMES = "SELECT MIN(created_at), MAX(created_at) FROM raw.teacher_label WHERE source = 'spotcheck'"


@dataclass(frozen=True)
class SpotcheckTeacher:
    client: TextModel
    model_id: str  # what its labels are stored under (`raw.teacher_label.model`)


@dataclass(frozen=True)
class SpotcheckOutcome:
    drawn: int
    labeled: int
    failed_batches: int
    stopped: str | None  # deadline, outage, quota


def is_holdout(recommendation_id: int) -> bool:
    """A fixed fifth of reviews, the same in every run: the recent holdout for promotions."""
    first = hashlib.sha256(str(int(recommendation_id)).encode()).digest()[0]
    return first < int(HOLDOUT_SHARE * 256)


def new_reviews(
    connection: Connection, *, run_id: int, n: int, seed: int
) -> list[tuple[ReviewIn, datetime]]:
    """Up to `n` reviews first seen in this run (their latest version), drawn with `seed`."""
    ids = [int(row[0]) for row in connection.execute(text(_CANDIDATES), {"run": run_id})]
    if len(ids) > n:
        rng = np.random.default_rng(seed)
        ids = sorted(int(i) for i in rng.choice(np.asarray(ids), size=n, replace=False))
    if not ids:
        return []
    statement = text(_PICKED).bindparams(bindparam("ids", expanding=True))
    rows = connection.execute(statement, {"run": run_id, "ids": ids}).all()
    return [
        (
            ReviewIn(review_id=int(r[0]), game=str(r[4]), voted_up=bool(r[2]), text=str(r[3])),
            r[1],
        )
        for r in sorted(rows, key=lambda row: int(row[0]))
    ]


def make_spotcheck_client(
    settings: Settings,
    http: httpx2.Client,
    *,
    state_path: Path,
    clock: Callable[[], datetime],
    sleep: Callable[[float], None] = time.sleep,
) -> TextModel:
    """Google's Gemma 4 31B with its limiter, the backup teacher. The online judge's copy counts
    its own day apart, so each uses the model's whole free quota (Phase 2 split it between two
    runs)."""
    from patchpulse.labeling.gemini import GeminiClient

    if settings.gemini_api_key is None:
        raise ValueError("the spot-check needs PP_GEMINI_API_KEY")
    config = load_model_config().candidates[GOOGLE_MODEL].model_copy(update={"use_fraction": 1.0})
    limiter = RequestLimiter(config, state_path=state_path, clock=clock, sleep=sleep)
    return GeminiClient(
        settings.gemini_api_key.get_secret_value(),
        http,
        limiter,
        config,
        sleep=sleep,
        max_attempts=SPOTCHECK_ATTEMPTS,
        # A timeout or an outage raises at once (final review C1: two tries of a stalled request
        # cost the night four minutes past the stage's budget); `run_spotcheck` decides whether
        # there's time to pause and try again.
        retry_errors=False,
    )


def make_server_client(
    settings: Settings,
    http: httpx2.Client,
    *,
    model: str = SERVER_MODEL,
    sleep: Callable[[float], None] = time.sleep,
) -> TextModel:
    """The friend's server (ADR-020): one attempt and a short connect timeout. Its caller decides
    what an outage means: the spot-check pauses or hands the night to Google, the online judge's
    router moves on and rests it."""
    from patchpulse.labeling.openai_compat import OpenAICompatClient

    if settings.llm_server_url is None or settings.llm_server_key is None:
        raise ValueError("the friend's server needs PP_LLM_SERVER_URL and PP_LLM_SERVER_KEY")
    return OpenAICompatClient(
        settings.llm_server_key.get_secret_value(),
        settings.llm_server_url,
        http,
        load_model_config().candidates[model],
        sleep=sleep,
        max_attempts=1,
        timeout=SERVER_TIMEOUT,
    )


def spotcheck_teachers(
    settings: Settings,
    http: httpx2.Client,
    *,
    state_path: Path,
    clock: Callable[[], datetime],
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[SpotcheckTeacher, SpotcheckTeacher | None]:
    """The friend's server, and Google's copy as its backup; either alone when the other's
    settings are missing."""
    candidates = load_model_config().candidates
    found: list[SpotcheckTeacher] = []
    if settings.llm_server_url is not None and settings.llm_server_key is not None:
        client = make_server_client(settings, http, sleep=sleep)
        found.append(SpotcheckTeacher(client, candidates[SERVER_MODEL].id))
    else:
        log.warning("spot-check: no PP_LLM_SERVER_URL or PP_LLM_SERVER_KEY; Google's copy alone")
    if settings.gemini_api_key is not None:
        google = make_spotcheck_client(
            settings, http, state_path=state_path, clock=clock, sleep=sleep
        )
        found.append(SpotcheckTeacher(google, candidates[GOOGLE_MODEL].id))
    if not found:
        raise ValueError(
            "the spot-check needs PP_LLM_SERVER_URL and PP_LLM_SERVER_KEY, or PP_GEMINI_API_KEY"
        )
    return found[0], found[1] if len(found) > 1 else None


def _label_through_outages(
    client: TextModel,
    batch: Batch,
    *,
    deadline: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> list[TeacherLabel]:
    """The batch's labels. An outage pauses and tries the batch again, `SPOTCHECK_OUTAGE_TRIES`
    tries in all, and only while the pause ends before the deadline; then it's raised."""
    tries = 1
    while True:
        try:
            return label_batch(client, batch, stop=lambda: clock() >= deadline)
        except TeacherUnavailableError as error:
            if tries >= SPOTCHECK_OUTAGE_TRIES or clock() + SPOTCHECK_OUTAGE_PAUSE_S >= deadline:
                raise
            tries += 1
            log.warning(
                "spot-check: the teacher is unavailable (%s); try %d of %d in %.0f s",
                error,
                tries,
                SPOTCHECK_OUTAGE_TRIES,
                SPOTCHECK_OUTAGE_PAUSE_S,
            )
            sleep(SPOTCHECK_OUTAGE_PAUSE_S)


def _stop(error: Exception) -> tuple[str, str]:
    """The outcome's `stopped`, and the log's reason: the teacher's message for an outage (never
    review text; the server's address and key are scrubbed by its client)."""
    if isinstance(error, QuotaExhaustedError):
        return "quota", ""
    return "outage", f": {error}"


def run_spotcheck(
    engine: Engine,
    *,
    run_id: int,
    client: TextModel,
    model_id: str,
    deadline: float,
    clock: Callable[[], float],
    n: int = SPOTCHECK_SIZE,
    sleep: Callable[[float], None] = time.sleep,
    backup: SpotcheckTeacher | None = None,
) -> SpotcheckOutcome:
    """Label tonight's draw with `client`; when it stays unavailable through its tries (or runs
    out of quota), `backup` labels the rest of the night, while there's time."""
    with engine.connect() as connection:
        drawn = new_reviews(connection, run_id=run_id, n=n, seed=run_id)
    versions = {review.review_id: updated for review, updated in drawn}
    digest = prompt_hash()
    labeled = failed = 0
    stopped: str | None = None
    reason = ""
    teacher = SpotcheckTeacher(client, model_id)
    for batch in make_batches([review for review, _ in drawn]):
        if clock() >= deadline:
            stopped = "deadline"
            break
        labels: list[TeacherLabel] | None = None
        while labels is None:
            try:
                labels = _label_through_outages(
                    teacher.client, batch, deadline=deadline, clock=clock, sleep=sleep
                )
            except (QuotaExhaustedError, TeacherUnavailableError) as error:
                if backup is None or clock() >= deadline:
                    stopped, reason = _stop(error)
                    break
                log.warning(
                    "spot-check: %s is unavailable (%s); %s labels the rest of the night",
                    teacher.model_id,
                    error,
                    backup.model_id,
                )
                teacher, backup = backup, None
        if labels is None:
            break
        if len(labels) < len(batch.review_ids):
            failed += 1
        stored = [
            StoredTeacherLabel(
                recommendation_id=label.review_id,
                timestamp_updated=versions[label.review_id],
                model=teacher.model_id,
                prompt_hash=digest,
                source=SPOTCHECK_SOURCE,
                aspects=label.aspects,
            )
            for label in labels
        ]
        with engine.begin() as connection:
            labeled += write_teacher_labels(connection, stored, run_id=run_id)
    log.info(
        "spot-check: %d drawn, %d labeled, %d batches incomplete%s",
        len(drawn),
        labeled,
        failed,
        f", stopped ({stopped}{reason})" if stopped else "",
    )
    return SpotcheckOutcome(
        drawn=len(drawn), labeled=labeled, failed_batches=failed, stopped=stopped
    )


def _matrix(rows: Sequence[Sequence[object]], start: int) -> NDArray[np.bool_]:
    k = len(ASPECTS)
    return np.array(
        [[bool(v) for v in row[start : start + k]] for row in rows], dtype=bool
    ).reshape(len(rows), k)


def weekly_agreement(
    connection: Connection, window: tuple[date, date], *, scored_as: str
) -> list[MetricRow]:
    """The champion's labels against the teacher's on the window's spot-check labels; `scored_as`
    is the name its scores are stored under (`scoring.scores_version`)."""
    start, end = window
    rows = connection.execute(
        text(_PAIRS),
        {
            "model": scored_as,
            "start": datetime.combine(start, datetime.min.time()),
            "stop": datetime.combine(end + timedelta(days=1), datetime.min.time()),
        },
    ).all()
    if not rows:
        return []
    teacher, student = _matrix(rows, 0), _matrix(rows, len(ASPECTS))
    n = len(rows)

    def macro(ix: NDArray[np.intp]) -> float:
        return assessable_macro_f1(teacher[ix], student[ix])

    lo, hi = bootstrap_ci(macro, n, seed=AGREEMENT_SEED)
    scores = prf(teacher, student)
    result = [
        MetricRow(
            "agreement",
            "macro_f1",
            "week",
            start,
            end,
            assessable_macro_f1(teacher, student),
            None if np.isnan(lo) else float(lo),
            None if np.isnan(hi) else float(hi),
            n,
        )
    ]
    result += [
        MetricRow(
            "agreement", f"f1.{code}", "week", start, end, float(f1), None, None, int(support)
        )
        for code, f1, support in zip(ASPECTS, scores.f1, scores.support, strict=True)
    ]
    return result


def label_times(connection: Connection) -> tuple[datetime | None, datetime | None]:
    """The first and the last spot-check label's time (naive UTC)."""
    first, last = connection.execute(text(_TIMES)).one()
    return first, last


def teacher_outcomes(
    rows: Sequence[MetricRow],
    reference: Reference,
    *,
    last_label_at: datetime | None,
    first_label_at: datetime | None,
    now: datetime,
    rules: Mapping[str, AlertRule],
) -> list[Outcome]:
    """`agreement_dropped` (weekly) and `teacher_unavailable` (nightly)."""
    outcomes = []
    drop = rules["agreement_dropped"].threshold
    macro = next((r for r in rows if r.metric == "macro_f1"), None)
    if macro is None or macro.hi is None or drop is None:
        outcomes.append(Outcome("agreement_dropped", None, "agreement: not measured"))
    else:
        floor = reference.teacher_agreement_macro_f1 - drop
        body = (
            f"macro_f1 {macro.value:.3f} (95% {macro.lo:.3f} to {macro.hi:.3f}), reference "
            f"{reference.teacher_agreement_macro_f1:.3f}, floor {floor:.3f}, "
            f"week {macro.window_start} to {macro.window_end}, n {macro.n}"
        )
        outcomes.append(Outcome("agreement_dropped", macro.hi < floor, body))
    outcomes.append(
        unavailable_outcome(first_label_at, last_label_at, now, rules["teacher_unavailable"])
    )
    return outcomes


def unavailable_outcome(
    first_label_at: datetime | None, last_label_at: datetime | None, now: datetime, rule: AlertRule
) -> Outcome:
    """`teacher_unavailable`: no spot-check label for `threshold` days, once there was one."""
    if first_label_at is None or last_label_at is None or rule.threshold is None:
        return Outcome(rule.id, None, "spot-check: no label yet")
    idle = (now - last_label_at).total_seconds() / 86_400
    body = f"last spot-check label {last_label_at:%Y-%m-%d %H:%M} UTC, {idle:.1f} days ago"
    return Outcome(rule.id, idle >= rule.threshold, body)


def run_live_spotcheck(
    engine: Engine,
    settings: Settings,
    *,
    run_id: int,
    clock: Callable[[], float],
    deadline: float,
) -> SpotcheckOutcome:
    """The nightly stage: the friend's server, Google's Gemma 4 31B as the backup with its day
    count in a temporary file (the job's file system is gone by morning)."""
    import tempfile
    from datetime import UTC

    def now() -> datetime:
        return datetime.now(UTC)

    with httpx2.Client() as http, tempfile.TemporaryDirectory() as scratch:
        first, backup = spotcheck_teachers(
            settings, http, state_path=Path(scratch) / "gemini-requests.json", clock=now
        )
        return run_spotcheck(
            engine, run_id=run_id, client=first.client, model_id=first.model_id, backup=backup,
            deadline=deadline, clock=clock,
        )  # fmt: skip
