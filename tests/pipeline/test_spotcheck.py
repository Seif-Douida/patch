"""Phase 7 Task 6: the nightly teacher spot-check and the weekly agreement (design D3)."""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from monitoring_fixtures import SENTINEL_TEXT, add_rows, review_row, score_row
from sqlalchemy import Engine, insert, inspect, text

from patchpulse.db import tables
from patchpulse.labeling.llm import LLMResponse, TeacherUnavailableError
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.metrics import assessable_macro_f1
from patchpulse.monitoring.alerts import load_rules
from patchpulse.monitoring.reference import Reference
from patchpulse.monitoring.store import MetricRow, read_teacher_labels
from patchpulse.pipeline.spotcheck import (
    SPOTCHECK_OUTAGE_PAUSE_S,
    SPOTCHECK_OUTAGE_TRIES,
    SPOTCHECK_SIZE,
    SpotcheckTeacher,
    is_holdout,
    run_spotcheck,
    teacher_outcomes,
    weekly_agreement,
)

CHAMPION = "20261012T223218Z"
RUN_ID = 42
CREATED = datetime(2026, 11, 22, 10, 0)


class FakeTeacher:
    """Labels every review `content`; can fail from a given call on, or on given calls only."""

    def __init__(
        self, *, fail_from: int | None = None, fail_on: frozenset[int] = frozenset()
    ) -> None:
        self.calls = 0
        self.fail_from = fail_from
        self.fail_on = fail_on
        self.users: list[str] = []

    def generate(
        self, system: str, user: str, *, json_schema: Mapping[str, Any] | None, **_: Any
    ) -> LLMResponse:
        self.calls += 1
        failing = self.fail_from is not None and self.calls >= self.fail_from
        if failing or self.calls in self.fail_on:
            raise TeacherUnavailableError(503, "high demand")
        self.users.append(user)
        items = json.loads(user[user.index("[") :])
        answer = {"labels": [{"alias": item["alias"], "aspects": ["content"]} for item in items]}
        return LLMResponse(json.dumps(answer), 100, 50, "STOP")


def seed(engine: Engine, n: int = 250) -> None:
    with engine.begin() as connection:
        connection.execute(
            insert(tables.game),
            [
                {
                    "appid": 553850,
                    "name": "Helldivers 2",
                    "genres": "[]",
                    "role": "tracked",
                    "backfill_start": date(2024, 2, 8),
                    "updated_run_id": 1,
                }
            ],
        )
    reviews = [review_row(i, CREATED, first_seen_run_id=RUN_ID) for i in range(1, n + 1)]
    reviews.append(review_row(9_999, CREATED, first_seen_run_id=RUN_ID - 1))  # an older review
    add_rows(engine, reviews)


def test_the_holdout_split_is_stable_and_about_a_fifth() -> None:
    ids = range(1, 10_001)
    share = sum(is_holdout(i) for i in ids) / 10_000
    assert 0.18 <= share <= 0.22
    assert [is_holdout(i) for i in range(50)] == [is_holdout(i) for i in range(50)]


@pytest.mark.integration
def test_two_hundred_new_reviews_in_ten_batches(migrated_engine: Engine) -> None:
    seed(migrated_engine)
    teacher = FakeTeacher()

    outcome = run_spotcheck(
        migrated_engine, run_id=RUN_ID, client=teacher, model_id="gemma-4-31b-it",
        deadline=float("inf"), clock=lambda: 0.0,
    )  # fmt: skip

    assert (outcome.drawn, outcome.labeled, outcome.stopped) == (
        SPOTCHECK_SIZE,
        SPOTCHECK_SIZE,
        None,
    )
    assert teacher.calls == 10
    with migrated_engine.connect() as connection:
        labels = read_teacher_labels(connection, source="spotcheck")
    assert len(labels) == SPOTCHECK_SIZE
    assert 9_999 not in {label.recommendation_id for label in labels}
    assert all(label.aspects == frozenset({"content"}) for label in labels)


@pytest.mark.integration
def test_an_outage_keeps_the_batches_already_labeled(
    migrated_engine: Engine, caplog: pytest.LogCaptureFixture
) -> None:
    seed(migrated_engine)
    teacher = FakeTeacher(fail_from=4)
    pauses: list[float] = []

    with caplog.at_level(logging.INFO, logger="patchpulse.pipeline.spotcheck"):
        outcome = run_spotcheck(
            migrated_engine, run_id=RUN_ID, client=teacher, model_id="m",
            deadline=float("inf"), clock=lambda: 0.0, sleep=pauses.append,
        )  # fmt: skip

    assert outcome.labeled == 60
    assert outcome.stopped == "outage"
    assert teacher.calls == 3 + SPOTCHECK_OUTAGE_TRIES  # the fourth batch, tried and given up
    assert pauses == [SPOTCHECK_OUTAGE_PAUSE_S] * (SPOTCHECK_OUTAGE_TRIES - 1)
    # The first night's line said only "stopped (outage)": the reason is in it now.
    assert "stopped (outage: 503: high demand)" in caplog.text
    with migrated_engine.connect() as connection:
        assert len(read_teacher_labels(connection)) == 60


@pytest.mark.integration
def test_a_passing_outage_is_waited_out_and_the_batch_tried_again(migrated_engine: Engine) -> None:
    # The first night (19 Nov): one 503 from an overloaded Gemma ended the spot-check at once,
    # 0 of 200 labeled, and the month's retrain had no recent labels to judge on.
    seed(migrated_engine)
    teacher = FakeTeacher(fail_on=frozenset({4}))
    pauses: list[float] = []

    outcome = run_spotcheck(
        migrated_engine, run_id=RUN_ID, client=teacher, model_id="m",
        deadline=float("inf"), clock=lambda: 0.0, sleep=pauses.append,
    )  # fmt: skip

    assert (outcome.labeled, outcome.stopped) == (SPOTCHECK_SIZE, None)
    assert teacher.calls == 11
    assert pauses == [SPOTCHECK_OUTAGE_PAUSE_S]


@pytest.mark.integration
def test_an_outage_never_pauses_past_the_deadline(migrated_engine: Engine) -> None:
    seed(migrated_engine)
    teacher = FakeTeacher(fail_from=1)
    now = [0.0]

    def pause(seconds: float) -> None:
        now[0] += seconds

    outcome = run_spotcheck(
        migrated_engine, run_id=RUN_ID, client=teacher, model_id="m",
        deadline=1.5 * SPOTCHECK_OUTAGE_PAUSE_S, clock=lambda: now[0], sleep=pause,
    )  # fmt: skip

    assert teacher.calls == 2  # a second pause would end past the deadline
    assert now[0] == SPOTCHECK_OUTAGE_PAUSE_S
    assert (outcome.labeled, outcome.stopped) == (0, "outage")


SERVER_MODEL_ID, GOOGLE_MODEL_ID = "gemma-31b", "gemma-4-31b-it"


@pytest.mark.integration
def test_a_server_outage_hands_the_night_to_the_backup(
    migrated_engine: Engine, caplog: pytest.LogCaptureFixture
) -> None:
    # Seif, 19 Nov: the friend's server first, Google's Gemma when it's down (ADR-020 amended).
    seed(migrated_engine)
    server, google = FakeTeacher(fail_from=1), FakeTeacher()

    with caplog.at_level(logging.INFO, logger="patchpulse.pipeline.spotcheck"):
        outcome = run_spotcheck(
            migrated_engine, run_id=RUN_ID, client=server, model_id=SERVER_MODEL_ID,
            backup=SpotcheckTeacher(google, GOOGLE_MODEL_ID), deadline=float("inf"),
            clock=lambda: 0.0, sleep=lambda _: None,
        )  # fmt: skip

    assert (outcome.labeled, outcome.stopped) == (SPOTCHECK_SIZE, None)
    assert server.calls == SPOTCHECK_OUTAGE_TRIES  # its tries, then never again tonight
    assert google.calls == 10
    assert f"{SERVER_MODEL_ID} is unavailable" in caplog.text
    assert f"{GOOGLE_MODEL_ID} labels the rest of the night" in caplog.text


@pytest.mark.integration
def test_each_label_names_the_teacher_that_gave_it(migrated_engine: Engine) -> None:
    seed(migrated_engine)
    server, google = FakeTeacher(fail_from=4), FakeTeacher()

    run_spotcheck(
        migrated_engine, run_id=RUN_ID, client=server, model_id=SERVER_MODEL_ID,
        backup=SpotcheckTeacher(google, GOOGLE_MODEL_ID), deadline=float("inf"),
        clock=lambda: 0.0, sleep=lambda _: None,
    )  # fmt: skip

    with migrated_engine.connect() as connection:
        labels = read_teacher_labels(connection, source="spotcheck")
    assert Counter(label.model for label in labels) == {SERVER_MODEL_ID: 60, GOOGLE_MODEL_ID: 140}


@pytest.mark.integration
def test_the_backup_failing_too_ends_the_night_as_an_outage(migrated_engine: Engine) -> None:
    seed(migrated_engine)
    server, google = FakeTeacher(fail_from=1), FakeTeacher(fail_from=1)

    outcome = run_spotcheck(
        migrated_engine, run_id=RUN_ID, client=server, model_id=SERVER_MODEL_ID,
        backup=SpotcheckTeacher(google, GOOGLE_MODEL_ID), deadline=float("inf"),
        clock=lambda: 0.0, sleep=lambda _: None,
    )  # fmt: skip

    assert (outcome.labeled, outcome.stopped) == (0, "outage")
    assert (server.calls, google.calls) == (SPOTCHECK_OUTAGE_TRIES, SPOTCHECK_OUTAGE_TRIES)


class StalledTeacher(FakeTeacher):
    """Fails every request after `seconds` (a server that accepts and never answers)."""

    def __init__(self, seconds: float) -> None:
        super().__init__(fail_from=1)
        self.now = 0.0
        self.seconds = seconds

    def generate(self, system: str, user: str, **kwargs: Any) -> LLMResponse:
        self.now += self.seconds
        return super().generate(system, user, **kwargs)


@pytest.mark.integration
def test_the_backup_never_starts_past_the_deadline(migrated_engine: Engine) -> None:
    seed(migrated_engine)
    server, google = StalledTeacher(seconds=60.0), FakeTeacher()

    outcome = run_spotcheck(
        migrated_engine, run_id=RUN_ID, client=server, model_id=SERVER_MODEL_ID,
        backup=SpotcheckTeacher(google, GOOGLE_MODEL_ID), deadline=30.0,
        clock=lambda: server.now, sleep=lambda _: None,
    )  # fmt: skip

    assert (server.calls, google.calls) == (1, 0)
    assert (outcome.labeled, outcome.stopped) == (0, "outage")


@pytest.mark.integration
def test_the_deadline_stops_between_batches(migrated_engine: Engine) -> None:
    seed(migrated_engine)
    ticks = iter(range(100))

    outcome = run_spotcheck(
        migrated_engine, run_id=RUN_ID, client=FakeTeacher(), model_id="m",
        deadline=3.0, clock=lambda: float(next(ticks)),
    )  # fmt: skip

    assert outcome.stopped == "deadline"
    assert 0 < outcome.labeled < SPOTCHECK_SIZE


@pytest.mark.integration
def test_spotcheck_labels_hold_no_text(
    migrated_engine: Engine, caplog: pytest.LogCaptureFixture
) -> None:
    seed(migrated_engine)
    caplog.set_level(logging.DEBUG)

    run_spotcheck(
        migrated_engine, run_id=RUN_ID, client=FakeTeacher(), model_id="m",
        deadline=float("inf"), clock=lambda: 0.0,
    )  # fmt: skip

    columns = [
        c["name"] for c in inspect(migrated_engine).get_columns("teacher_label", schema="raw")
    ]
    with migrated_engine.connect() as connection:
        stored = connection.execute(text("SELECT * FROM raw.teacher_label")).all()
    assert all(SENTINEL_TEXT not in str(value) for row in stored for value in row)
    assert "text" not in " ".join(columns)
    assert SENTINEL_TEXT not in caplog.text


def test_agreement_on_constructed_labels() -> None:
    truth = np.zeros((10, len(ASPECTS)), dtype=bool)
    truth[:5, 0] = True
    truth[5:, 3] = True
    assert assessable_macro_f1(truth, truth) == pytest.approx(1.0)  # aspects never seen don't count
    wrong = truth.copy()
    wrong[:, 0] = False
    assert assessable_macro_f1(truth, wrong) == pytest.approx(0.5)


@pytest.mark.integration
def test_weekly_agreement_compares_the_champion_with_the_teacher(migrated_engine: Engine) -> None:
    seed(migrated_engine, n=40)
    add_rows(
        migrated_engine,
        [],
        [
            score_row(
                i,
                CREATED,
                model_version=CHAMPION,
                aspects=["content"] if i <= 30 else ["performance"],
            )
            for i in range(1, 41)
        ]
        + [score_row(i, CREATED, model_version="other", aspects=[]) for i in range(1, 41)],
    )
    run_spotcheck(
        migrated_engine, run_id=RUN_ID, client=FakeTeacher(), model_id="m",
        deadline=float("inf"), clock=lambda: 0.0,
    )  # fmt: skip
    today = datetime.now(UTC).date()

    with migrated_engine.connect() as connection:
        rows = weekly_agreement(connection, (today - timedelta(days=7), today), scored_as=CHAMPION)

    by_name = {row.metric: row for row in rows}
    assert by_name["macro_f1"].n == 40
    assert by_name["f1.content"].value == pytest.approx(2 * 30 / (2 * 30 + 10))
    assert by_name["macro_f1"].lo is not None
    assert all(row.family == "agreement" and row.scope == "week" for row in rows)


def reference(agreement: float = 0.75) -> Reference:
    return Reference(
        model_version=CHAMPION,
        languages_top=["english"],
        language_shares={"english": 1.0},
        length_shares={"bin4": 1.0},
        aspect_shares={c: 0.1 for c in ASPECTS},
        uncertain_share=0.0,
        thresholds={c: 0.5 for c in ASPECTS},
        teacher_agreement_macro_f1=agreement,
        sample_ids=[],
        n=1,
    )


def week_row(value: float, hi: float) -> MetricRow:
    return MetricRow(
        "agreement",
        "macro_f1",
        "week",
        date(2026, 11, 16),
        date(2026, 11, 22),
        value,
        value - 0.05,
        hi,
        700,
    )


def test_agreement_alert_uses_the_upper_bound() -> None:
    now = datetime(2026, 11, 23, 3, 0)
    rules = load_rules()
    low = {
        o.rule_id: o
        for o in teacher_outcomes(
            [week_row(0.6, 0.69)],
            reference(),
            last_label_at=now,
            first_label_at=now - timedelta(days=30),
            now=now,
            rules=rules,
        )
    }
    close = {
        o.rule_id: o
        for o in teacher_outcomes(
            [week_row(0.6, 0.71)],
            reference(),
            last_label_at=now,
            first_label_at=now - timedelta(days=30),
            now=now,
            rules=rules,
        )
    }

    assert low["agreement_dropped"].breached is True
    assert close["agreement_dropped"].breached is False


def test_teacher_unavailable_waits_a_week_after_the_first_label() -> None:
    now = datetime(2026, 11, 23, 3, 0)
    rules = load_rules()

    def unavailable(last: datetime | None, first: datetime | None) -> bool | None:
        outcomes = teacher_outcomes(
            [], reference(), last_label_at=last, first_label_at=first, now=now, rules=rules
        )
        return {o.rule_id: o for o in outcomes}["teacher_unavailable"].breached

    assert unavailable(None, None) is None  # never labeled yet: nothing to say
    assert unavailable(now - timedelta(days=8), now - timedelta(days=8)) is True
    assert unavailable(now - timedelta(days=2), now - timedelta(days=20)) is False
    assert unavailable(now - timedelta(days=8), now - timedelta(days=30)) is True


class SlowInvalidTeacher(FakeTeacher):
    """Answers every request with something that never validates, `seconds` each."""

    def __init__(self, seconds: float) -> None:
        super().__init__()
        self.now = 0.0
        self.seconds = seconds

    def generate(self, system: str, user: str, **_: Any) -> LLMResponse:
        self.calls += 1
        self.now += self.seconds
        return LLMResponse("not json", 100, 5, "STOP")


@pytest.mark.integration
def test_a_failing_batch_is_not_split_past_the_deadline(migrated_engine: Engine) -> None:
    # Final review C1 (18 Nov): a batch that never validates is relabeled one review at a time,
    # up to 40 requests; past the deadline that fallback must not start.
    seed(migrated_engine)
    teacher = SlowInvalidTeacher(seconds=60.0)

    outcome = run_spotcheck(
        migrated_engine, run_id=RUN_ID, client=teacher, model_id="m",
        deadline=30.0, clock=lambda: teacher.now,
    )  # fmt: skip

    assert teacher.calls == 2  # the batch and its repair, then nothing
    assert outcome.stopped == "deadline"


def test_the_spotcheck_client_never_retries_a_stalled_request(tmp_path: Path) -> None:
    from pydantic import SecretStr

    from patchpulse.config import Settings
    from patchpulse.pipeline.spotcheck import make_spotcheck_client

    settings = Settings.model_construct(gemini_api_key=SecretStr("k-test"))
    client = make_spotcheck_client(
        settings,
        None,  # type: ignore[arg-type]  # the client is built, never asked
        state_path=tmp_path / "s.json",
        clock=lambda: CREATED,
    )

    assert client.retry_errors is False  # type: ignore[attr-defined]


def teacher_settings(*, server: bool, google: bool) -> Any:
    from pydantic import SecretStr

    from patchpulse.config import Settings

    return Settings.model_construct(
        gemini_api_key=SecretStr("k-test") if google else None,
        llm_server_url="https://llm.example.net/v1" if server else None,
        llm_server_key=SecretStr("s-test") if server else None,
    )


def test_the_nightly_asks_the_server_first_and_keeps_google_as_the_backup(tmp_path: Path) -> None:
    import httpx2

    from patchpulse.labeling.openai_compat import OpenAICompatClient
    from patchpulse.pipeline.spotcheck import SERVER_TIMEOUT, spotcheck_teachers

    first, backup = spotcheck_teachers(
        teacher_settings(server=True, google=True), None,  # type: ignore[arg-type]
        state_path=tmp_path / "s.json", clock=lambda: CREATED,
    )  # fmt: skip

    assert first.model_id == SERVER_MODEL_ID
    assert isinstance(first.client, OpenAICompatClient)
    assert first.client.max_attempts == 1  # an outage is the spot-check's to wait out or hand over
    assert first.client.timeout == SERVER_TIMEOUT
    assert isinstance(SERVER_TIMEOUT, httpx2.Timeout)
    assert SERVER_TIMEOUT.connect == 10.0
    assert backup is not None
    assert backup.model_id == GOOGLE_MODEL_ID
    assert backup.client.retry_errors is False  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("server", "google", "first_model"),
    [(True, False, SERVER_MODEL_ID), (False, True, GOOGLE_MODEL_ID)],
    ids=["server only", "google only"],
)
def test_one_teacher_has_no_backup(
    tmp_path: Path, server: bool, google: bool, first_model: str
) -> None:
    from patchpulse.pipeline.spotcheck import spotcheck_teachers

    first, backup = spotcheck_teachers(
        teacher_settings(server=server, google=google), None,  # type: ignore[arg-type]
        state_path=tmp_path / "s.json", clock=lambda: CREATED,
    )  # fmt: skip

    assert (first.model_id, backup) == (first_model, None)


def test_no_teacher_names_both_settings(tmp_path: Path) -> None:
    from patchpulse.pipeline.spotcheck import spotcheck_teachers

    with pytest.raises(ValueError, match=r"PP_LLM_SERVER_URL.*PP_GEMINI_API_KEY"):
        spotcheck_teachers(
            teacher_settings(server=False, google=False), None,  # type: ignore[arg-type]
            state_path=tmp_path / "s.json", clock=lambda: CREATED,
        )  # fmt: skip
