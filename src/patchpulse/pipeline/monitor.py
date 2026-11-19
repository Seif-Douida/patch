"""The nightly's weekly drift stage (Phase 7 design D1, D2, D11).

It runs inside `pp-nightly`'s wake when the last stored drift week ended 6 or more days ago, on
the 7 full days before today: the latest version of each review created then, with the current
champion's scores only (a promotion's rescoring nights mix models otherwise). The text is never
selected, only its length. The week is measured against `models/reference.json` (Task 3), the
numbers stored, and the drift rules raised or resolved (Task 2).

In the acceptance drill (`PP_DRIFT_DRILL=language_shift`) it runs whatever the timing, resamples
the week so non-English reviews are twice their share, stores the numbers with scope `drill` and
raises `[drill]` issues only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
from sqlalchemy import Connection, Engine, text

from patchpulse.config import Settings
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.monitoring.alerts import AlertPublisher, AlertRule, apply_outcomes
from patchpulse.monitoring.drift import (
    DRIFT_SEED,
    MIN_WEEK_REVIEWS,
    VectorSource,
    WeekSample,
    drift_metrics,
    drift_outcomes,
)
from patchpulse.monitoring.reference import REFERENCE_FILE, read_reference
from patchpulse.monitoring.store import MetricRow, last_window_end, read_metrics, write_metrics
from patchpulse.pipeline.scoring import scores_version
from patchpulse.pipeline.spotcheck import teacher_outcomes, weekly_agreement

log = logging.getLogger("patchpulse.pipeline.monitor")

MONITOR_EVERY_DAYS = 6
WEEK_DAYS = 7
DRILLS = frozenset({"language_shift"})
DRILL_FACTOR = 2.0
ENGLISH = "english"

_PROBS = ", ".join(f"s.p_{code}" for code in ASPECTS)
_LABELS = ", ".join(f"s.l_{code}" for code in ASPECTS)
_WEEK = f"""
WITH latest AS (
    SELECT r.recommendation_id, r.timestamp_updated, r.language,
           LEN(r.review_text) AS n_chars,
           ROW_NUMBER() OVER (
               PARTITION BY r.recommendation_id ORDER BY r.timestamp_updated DESC
           ) AS version_rank
    FROM raw.steam_review AS r
    WHERE r.timestamp_created >= :start AND r.timestamp_created < :stop
)
SELECT l.recommendation_id, l.language, l.n_chars, {_PROBS}, {_LABELS}
FROM latest AS l
JOIN raw.review_aspect_score AS s
  ON s.recommendation_id = l.recommendation_id
 AND s.timestamp_updated = l.timestamp_updated
 AND s.model_version = :model
WHERE l.version_rank = 1
ORDER BY l.recommendation_id
"""  # noqa: S608 (column names from the taxonomy, values bound)


@dataclass(frozen=True)
class MonitorOutcome:
    ran: bool
    skipped: str | None = None
    raised: list[str] = field(default_factory=list)
    new_drift: bool = False


def week_due(last_end: date | None, today: date) -> bool:
    return last_end is None or (today - last_end).days >= MONITOR_EVERY_DAYS


def week_window(today: date) -> tuple[date, date]:
    return today - timedelta(days=WEEK_DAYS), today - timedelta(days=1)


def load_week(connection: Connection, window: tuple[date, date], *, scored_as: str) -> WeekSample:
    """The week's reviews with the scores stored under `scored_as` (`scoring.scores_version`)."""
    start, end = window
    rows = connection.execute(
        text(_WEEK),
        {
            "start": datetime.combine(start, time()),
            "stop": datetime.combine(end + timedelta(days=1), time()),
            "model": scored_as,
        },
    ).all()
    k = len(ASPECTS)
    return WeekSample(
        review_ids=[int(r[0]) for r in rows],
        languages=[str(r[1]) for r in rows],
        lengths=[int(r[2] or 0) for r in rows],
        probs=np.array([[float(v) for v in r[3 : 3 + k]] for r in rows], dtype=np.float64).reshape(
            len(rows), k
        ),
        labels=np.array([[bool(v) for v in r[3 + k :]] for r in rows], dtype=bool).reshape(
            len(rows), k
        ),
    )


def drill_resample(
    week: WeekSample, *, factor: float = DRILL_FACTOR, seed: int = DRIFT_SEED
) -> WeekSample:
    """The same number of reviews, non-English ones drawn (with replacement) to `factor` times
    their share."""
    n = len(week.review_ids)
    other = [i for i, lang in enumerate(week.languages) if lang != ENGLISH]
    english = [i for i, lang in enumerate(week.languages) if lang == ENGLISH]
    if not n or not other or not english:
        return week
    target = min(n, round(len(other) / n * factor * n))
    rng = np.random.default_rng(seed)
    picks = np.concatenate(
        [rng.choice(other, target, replace=True), rng.choice(english, n - target, replace=True)]
    )
    return WeekSample(
        review_ids=[week.review_ids[i] for i in picks],
        languages=[week.languages[i] for i in picks],
        lengths=[week.lengths[i] for i in picks],
        labels=week.labels[picks],
        probs=week.probs[picks],
    )


def _breached(rows: list[MetricRow], rules: Mapping[str, AlertRule]) -> set[str]:
    return {o.rule_id for o in drift_outcomes(rows, rules) if o.breached}


def _previous_week(connection: Connection, before: date) -> list[MetricRow]:
    rows = read_metrics(connection, family="drift", scope="week")
    ends = [r.window_end for r in rows if r.window_end is not None and r.window_end < before]
    if not ends:
        return []
    last = max(ends)
    return [r for r in rows if r.window_end == last]


def drift_pending(connection: Connection, rules: Mapping[str, AlertRule]) -> bool:
    """A drift retrain is owed:

    - the latest stored week breached a drift rule the week before didn't, and no drift retrain
      was asked for since it ended: a request lost to a failed night is asked the next one (final
      review I1);
    - or the latest week still breaches, and the last drift request got no decision from
      pp-retrain (retrain.yml refuses a second run within 12 hours, 19 Nov). Asked once more only:
      a pp-retrain that keeps failing would spend the SQL allowance every night.
    """
    rows = read_metrics(connection, family="drift", scope="week")
    ends = sorted({r.window_end for r in rows if r.window_end is not None})
    if not ends:
        return False
    latest = [r for r in rows if r.window_end == ends[-1]]
    before = [r for r in rows if len(ends) > 1 and r.window_end == ends[-2]]
    breached = _breached(latest, rules)
    if not breached:
        return False
    retrain = read_metrics(connection, family="retrain")  # requests and decisions, oldest first
    if breached - _breached(before, rules):
        asked = [
            r.window_end
            for r in retrain
            if r.scope == "night" and r.metric == "requested.drift" and r.window_end is not None
        ]
        if not any(end >= ends[-1] for end in asked):
            return True
    unanswered = 0  # drift requests since pp-retrain's last decision
    for row in retrain:
        if row.metric.startswith("decision."):
            unanswered = 0
        elif row.metric == "requested.drift":
            unanswered += 1
    return unanswered == 1


def run_monitor(
    engine: Engine,
    *,
    run_id: int,
    today: date,
    family: str,
    model_version: str,
    vectors: VectorSource,
    publisher: AlertPublisher,
    rules: Mapping[str, AlertRule],
    drill: str | None,
    deadline: float,
    clock: Callable[[], float],
    reference_path: Path = REFERENCE_FILE,
) -> MonitorOutcome:
    if drill is not None and drill not in DRILLS:
        raise ValueError(f"unknown drill {drill!r}; known: {sorted(DRILLS)}")
    if clock() >= deadline:
        return MonitorOutcome(ran=False, skipped="no time left")
    reference = read_reference(reference_path)
    if reference.model_version != model_version:
        skipped = (
            f"{reference_path} is for model {reference.model_version}, the champion is "
            f"{model_version}"
        )
        log.warning("drift: skipped, the reference doesn't match: %s", skipped)
        return MonitorOutcome(ran=False, skipped=skipped)
    window = week_window(today)
    # The reference names the champion's version; its scores are stored under family-version.
    scored_as = scores_version(family, model_version)
    with engine.connect() as connection:
        if drill is None and not week_due(last_window_end(connection, family="drift"), today):
            return MonitorOutcome(ran=False, skipped="not due")
        week = load_week(connection, window, scored_as=scored_as)
        previous = [] if drill else _previous_week(connection, window[1])
    if len(week.review_ids) < MIN_WEEK_REVIEWS:
        skipped = f"{len(week.review_ids)} reviews this week, fewer than {MIN_WEEK_REVIEWS}"
        log.info("drift: skipped, %s", skipped)
        return MonitorOutcome(ran=False, skipped=skipped)
    if drill:
        week = drill_resample(week)
    scope = "drill" if drill else "week"
    rows = drift_metrics(reference, week, vectors, window=window, scope=scope)
    with engine.begin() as connection:
        write_metrics(connection, rows, run_id=run_id)
    outcomes = drift_outcomes(rows, rules)
    before = _breached(previous, rules)
    new_drift = any(o.breached and o.rule_id not in before for o in outcomes)
    if not drill:  # the teacher-student agreement (Task 6), on the same week
        with engine.begin() as connection:
            agreement = weekly_agreement(connection, window, scored_as=scored_as)
            write_metrics(connection, agreement, run_id=run_id)
        outcomes += [
            o
            for o in teacher_outcomes(
                agreement,
                reference,
                last_label_at=None,
                first_label_at=None,
                now=datetime.combine(today, time()),
                rules=rules,
            )
            if o.rule_id == "agreement_dropped"
        ]
    raised = apply_outcomes(publisher, rules, outcomes, drill=drill is not None)
    log.info(
        "drift: week %s to %s, %d reviews%s; raised %s",
        window[0],
        window[1],
        len(week.review_ids),
        " (drill)" if drill else "",
        raised or "nothing",
    )
    return MonitorOutcome(ran=True, raised=raised, new_drift=new_drift)


def run_live_monitor(
    engine: Engine,
    settings: Settings,
    *,
    run_id: int,
    today: date,
    publisher: AlertPublisher,
    rules: Mapping[str, AlertRule],
    clock: Callable[[], float],
    deadline: float,
) -> MonitorOutcome:
    """The nightly stage: the champion baked into the image, the index's collection the retrieval
    pin names, and the drill from PP_DRIFT_DRILL."""
    from patchpulse.export.site import read_aspect_model
    from patchpulse.monitoring.vectors import QdrantVectors
    from patchpulse.retrieval.pin import read_pin
    from patchpulse.retrieval.qdrant import make_client

    model = read_aspect_model(settings.model_dir)
    if model is None or settings.qdrant_url is None:
        raise ValueError("the drift check needs the champion (PP_MODEL_DIR) and PP_QDRANT_URL")
    client = make_client(settings.qdrant_url, settings.qdrant_api_key)
    try:
        return run_monitor(
            engine,
            run_id=run_id,
            today=today,
            family=model.family,
            model_version=model.version,
            vectors=QdrantVectors(client, read_pin().collection),
            publisher=publisher,
            rules=rules,
            drill=settings.drift_drill,
            deadline=deadline,
            clock=clock,
        )
    finally:
        client.close()
