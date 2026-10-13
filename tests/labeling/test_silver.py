"""The silver set (plan Task 10): ~30k teacher-labeled reviews the students train on."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fake_clock import FakeClock

from patchpulse.labeling.cache import LabelCache
from patchpulse.labeling.gemini import GeminiResponse, GeminiUnavailableError
from patchpulse.labeling.limits import QuotaExhaustedError
from patchpulse.labeling.silver import (
    label_silver,
    label_silver_until_done,
    load_silver,
    prevalence_report,
    sample_silver,
    split_holdout,
    write_silver_ids,
)
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.labeling.teacher import REVIEWS_MARKER, LabelRun, prompt_hash
from patchpulse.models import cli
from patchpulse.models.cli import main
from patchpulse.models.snapshot import Snapshot

SIZES = {1: 5_000, 2: 1_000, 3: 400}  # reviews per game: unequal, so the cap matters
LANGUAGES = ("english",) * 6 + ("schinese",) * 3 + ("french",)  # 60% / 30% / 10% in every game
MODEL = "gemma-4-26b-a4b-it"


def snapshot() -> Snapshot:
    rows = []
    review_id = 1
    for appid, size in SIZES.items():
        for i in range(size):
            empty = i % 50 == 7  # 2% have no usable text
            rows.append(
                {
                    "review_id": review_id,
                    "appid": appid,
                    "language": LANGUAGES[i % len(LANGUAGES)],
                    "voted_up": i % 5 != 0,
                    "text": ("   " if i % 2 else "[b][/b]") if empty else f"review {review_id}",
                }
            )
            review_id += 1
    games = pd.DataFrame({"appid": list(SIZES), "name": [f"Game {a}" for a in SIZES]})
    return Snapshot(reviews=pd.DataFrame(rows), games=games, patches=pd.DataFrame())


def test_silver_size_cap_and_exclusions() -> None:
    snap = snapshot()
    gold = set(range(1, 101))

    ids = sample_silver(snap, exclude_ids=gold, n=2_000, per_game_cap=900, seed=1)

    assert len(ids) == len(set(ids)) == 2_000
    assert not gold & set(ids)
    chosen = snap.reviews.set_index("review_id").loc[ids]
    assert not chosen["text"].isin(["   ", "[b][/b]"]).any()
    per_game = chosen["appid"].value_counts()
    assert per_game[1] == 900  # 5,000 of 6,400 reviews would earn ~1,560 without the cap
    assert per_game.sum() == 2_000


def test_language_mix_follows_the_snapshot() -> None:
    snap = snapshot()

    ids = sample_silver(snap, exclude_ids=set(), n=2_000, seed=2)

    chosen = snap.reviews.set_index("review_id").loc[ids]
    shares = chosen["language"].value_counts(normalize=True)
    assert shares["english"] == pytest.approx(0.6, abs=0.03)
    assert shares["schinese"] == pytest.approx(0.3, abs=0.03)
    assert shares["french"] == pytest.approx(0.1, abs=0.03)
    assert chosen["voted_up"].mean() == pytest.approx(0.8, abs=0.03)


def test_silver_sample_is_deterministic() -> None:
    snap = snapshot()

    first = sample_silver(snap, exclude_ids={5}, n=500, seed=3)

    assert sample_silver(snap, exclude_ids={5}, n=500, seed=3) == first
    assert sample_silver(snap, exclude_ids={5}, n=500, seed=4) != first


def test_holdout_is_a_deterministic_disjoint_tenth() -> None:
    ids = list(range(1, 1_001))

    train, holdout = split_holdout(ids, fraction=0.1, seed=5)

    assert len(holdout) == 100
    assert sorted(train + holdout) == ids
    assert split_holdout(ids, fraction=0.1, seed=5) == (train, holdout)


class Teacher:
    """Labels every review `performance`; raises the daily-quota error after `calls_left` calls."""

    def __init__(self, calls_left: int | None = None) -> None:
        self.calls_left = calls_left
        self.calls = 0

    def generate(
        self,
        system: str,
        user: str,
        *,
        json_schema: Mapping[str, Any] | None,
        expected_output_tokens: int = 512,
        max_output_tokens: int = 4096,
    ) -> GeminiResponse:
        if self.calls_left is not None and self.calls == self.calls_left:
            raise QuotaExhaustedError("daily quota")
        self.calls += 1
        aliases = [r["alias"] for r in json.loads(user.split(REVIEWS_MARKER, 1)[1])]
        labels = [{"alias": a, "aspects": ["performance"]} for a in aliases]
        return GeminiResponse(json.dumps({"labels": labels}), 1_000, 100, "STOP")


def silver_files(tmp_path: Path, snap: Snapshot, n: int) -> list[int]:
    ids = sample_silver(snap, exclude_ids=set(), n=n, seed=6)
    train, holdout = split_holdout(ids, seed=6)
    write_silver_ids(tmp_path, train, holdout)
    return ids


def test_quota_exhaustion_stops_cleanly_and_the_rerun_resumes(tmp_path: Path) -> None:
    snap = snapshot()
    ids = silver_files(tmp_path, snap, n=100)  # 5 batches of 20
    cache = LabelCache(tmp_path / "teacher_labels.sqlite")

    first = label_silver(Teacher(calls_left=2), cache, snap, tmp_path, model=MODEL)
    assert first == LabelRun(labeled=40, cached=0, remaining=60, stopped_for_quota=True)

    rerun = Teacher()
    second = label_silver(rerun, cache, snap, tmp_path, model=MODEL)
    assert second == LabelRun(labeled=60, cached=40, remaining=0, stopped_for_quota=False)
    assert rerun.calls == 3  # only the remaining batches

    silver = load_silver(tmp_path)
    assert len(silver.train) + len(silver.holdout) == 100
    assert sorted([*silver.train["review_id"], *silver.holdout["review_id"]]) == sorted(ids)
    assert list(silver.train.columns) == ["review_id", *ASPECTS]
    assert silver.train["performance"].all()
    assert not silver.train["content"].any()


def test_progress_is_logged_every_thousand_reviews(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    snap = snapshot()
    silver_files(tmp_path, snap, n=2_500)
    caplog.set_level(logging.INFO, logger="patchpulse.labeling.teacher")

    label_silver(Teacher(), LabelCache(tmp_path / "c.sqlite"), snap, tmp_path, model=MODEL)

    progress = [r.getMessage() for r in caplog.records if "labeled so far" in r.getMessage()]
    assert progress == [
        "1,000 of 2,500 labeled so far (0 were cached)",
        "2,000 of 2,500 labeled so far (0 were cached)",
    ]


def test_silver_labels_hold_no_text(tmp_path: Path) -> None:
    snap = snapshot()
    silver_files(tmp_path, snap, n=40)
    label_silver(Teacher(), LabelCache(tmp_path / "c.sqlite"), snap, tmp_path, model=MODEL)

    stored = pd.read_parquet(tmp_path / "silver" / "labels.parquet")

    assert "text" not in stored.columns
    assert set(stored.columns) == {"review_id", "split", "aspects", "model", "prompt_hash"}


def test_prevalence_counts_positives_per_aspect_and_language(tmp_path: Path) -> None:
    snap = snapshot()
    silver_files(tmp_path, snap, n=40)
    label_silver(Teacher(), LabelCache(tmp_path / "c.sqlite"), snap, tmp_path, model=MODEL)

    report = prevalence_report(tmp_path, snap)

    lines = report.splitlines()
    assert lines[0].startswith("aspect | all | english")
    assert "performance | 40 |" in report
    assert "content | 0 |" in report
    assert "review" not in report.replace("reviews (40)", "")  # counts only, no text


def models_yaml(tmp_path: Path, teacher: str | None) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "models.yaml").write_text(
        "candidates:\n"
        "  gemma:\n"
        "    id: gemma-4-26b-a4b-it\n"
        "    rpm: 30\n"
        "    tpm: 16000\n"
        "    rpd: 14400\n"
        "    json_mode: true\n"
        "    system_instruction: true\n"
        "    use_fraction: 0.5\n"
        "  server:\n"
        "    provider: openai_compat\n"
        "    id: gemma-31b\n"
        "    json_mode: true\n"
        "    system_instruction: true\n"
        f"teacher: {teacher or 'null'}\n",
        encoding="utf-8",
    )


def test_teacher_label_sends_gemini_one_request_at_a_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_env(monkeypatch, tmp_path)
    monkeypatch.setenv("PP_GEMINI_API_KEY", "AIza-test-key")
    models_yaml(tmp_path, teacher="gemma")

    assert main(["teacher", "label", "--set", "silver", "--workers", "4"]) == 2
    assert "one request at a time" in capsys.readouterr().err


def test_a_self_hosted_teacher_needs_the_server_address_and_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_env(monkeypatch, tmp_path)
    models_yaml(tmp_path, teacher="server")

    assert main(["teacher", "label", "--set", "silver", "--workers", "4"]) == 2
    error = capsys.readouterr().err
    assert "PP_LLM_SERVER_URL" in error
    assert "PP_LLM_SERVER_KEY" in error


def test_silver_labeling_can_run_in_parallel(tmp_path: Path) -> None:
    snap = snapshot()
    ids = silver_files(tmp_path, snap, n=100)

    run = label_silver(
        Teacher(), LabelCache(tmp_path / "c.sqlite"), snap, tmp_path, model=MODEL, workers=3
    )

    assert run.labeled == 100
    silver = load_silver(tmp_path)
    assert sorted([*silver.train["review_id"], *silver.holdout["review_id"]]) == sorted(ids)


def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in list(os.environ):
        if name.startswith(("PP_", "MLFLOW_")):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


def test_teacher_label_needs_a_chosen_teacher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_env(monkeypatch, tmp_path)
    monkeypatch.setenv("PP_GEMINI_API_KEY", "AIza-test-key")
    models_yaml(tmp_path, teacher=None)

    assert main(["teacher", "label", "--set", "silver", "--keep-going"]) == 2
    assert "teacher bakeoff" in capsys.readouterr().err


def test_silver_sample_command_refuses_to_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_env(monkeypatch, tmp_path)
    write_silver_ids(tmp_path / "data" / "local", [1, 2], [3])

    assert main(["silver", "sample", "--seed", "7"]) == 2
    assert "--force" in capsys.readouterr().err


class Flaky(Teacher):
    """Google is down for `outages` stretches: each one fails the next request."""

    def __init__(self, outages: int) -> None:
        super().__init__()
        self.outages = outages
        self.served = 0

    def generate(
        self,
        system: str,
        user: str,
        *,
        json_schema: Mapping[str, Any] | None,
        expected_output_tokens: int = 512,
        max_output_tokens: int = 4096,
    ) -> GeminiResponse:
        if self.outages and self.served % 2 == 1:  # every other batch, while outages last
            self.outages -= 1
            self.served += 1
            raise GeminiUnavailableError(503, "This model is currently experiencing high demand.")
        self.served += 1
        return super().generate(system, user, json_schema=json_schema)


def test_keep_going_resumes_after_outages_until_done(tmp_path: Path) -> None:
    snap = snapshot()
    silver_files(tmp_path, snap, n=100)  # 5 batches
    clock = FakeClock(datetime(2026, 10, 12, 12, 0, tzinfo=UTC))

    run = label_silver_until_done(
        Flaky(outages=2),
        LabelCache(tmp_path / "c.sqlite"),
        snap,
        tmp_path,
        model=MODEL,
        pause_s=900,
        max_s=6 * 3600,
        clock=clock,
        sleep=clock.sleep,
    )

    assert run == LabelRun(labeled=100, cached=0, remaining=0, stopped_for_quota=False)
    assert clock.slept == [900, 900]
    assert len(load_silver(tmp_path).train) + len(load_silver(tmp_path).holdout) == 100


def test_keep_going_gives_up_at_its_time_limit(tmp_path: Path) -> None:
    snap = snapshot()
    silver_files(tmp_path, snap, n=40)
    clock = FakeClock(datetime(2026, 10, 12, 12, 0, tzinfo=UTC))

    run = label_silver_until_done(
        DownAfter(1),
        LabelCache(tmp_path / "c.sqlite"),
        snap,
        tmp_path,
        model=MODEL,
        pause_s=900,
        max_s=3600,
        clock=clock,
        sleep=clock.sleep,
    )

    assert run.stopped_for_outage
    assert run.labeled == 20  # the first batch of each attempt succeeds, then Google fails
    assert sum(clock.slept) <= 3600


class DownAfter(Teacher):
    """Answers `ok` requests, then Google stays down."""

    def __init__(self, ok: int) -> None:
        super().__init__()
        self.ok = ok

    def generate(
        self,
        system: str,
        user: str,
        *,
        json_schema: Mapping[str, Any] | None,
        expected_output_tokens: int = 512,
        max_output_tokens: int = 4096,
    ) -> GeminiResponse:
        if self.ok == 0:
            raise GeminiUnavailableError(500, "Internal error encountered.")
        self.ok -= 1
        return super().generate(system, user, json_schema=json_schema)


def test_the_teacher_labels_the_gold_set_into_its_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The eval report scores the teacher on gold test too (plan Task 14)."""
    clean_env(monkeypatch, tmp_path)
    models_yaml(tmp_path, teacher="server")
    monkeypatch.setenv("PP_LLM_SERVER_URL", "https://llm.example/v1")
    monkeypatch.setenv("PP_LLM_SERVER_KEY", "test-key")
    data, gold = tmp_path / "data" / "local", tmp_path / "data" / "gold"
    data.mkdir(parents=True)
    gold.mkdir(parents=True)
    snap = snapshot()
    snap.reviews.to_parquet(data / "reviews.parquet", index=False)
    snap.games.to_parquet(data / "games.parquet", index=False)
    pd.DataFrame({"appid": [], "title": []}).to_parquet(data / "patches.parquet")
    gold_ids = [2, 3, 5, 8, 13, 21, 34]
    split = ["dev", "dev", "test", "test", "test", "test", "test"]
    pd.DataFrame({"review_id": gold_ids, "split": split}).to_csv(gold / "sample.csv", index=False)
    teacher = Teacher()
    monkeypatch.setattr(cli, "_client_for", lambda *_: teacher)

    assert main(["teacher", "label", "--set", "gold", "--workers", "2"]) == 0

    cached = LabelCache(data / "teacher_labels.sqlite").get_many(
        gold_ids, model="openai_compat/gemma-31b", prompt_hash=prompt_hash()
    )
    assert sorted(cached) == gold_ids
    assert teacher.calls == 1  # seven reviews fit one batch


def test_keep_going_is_for_the_silver_set_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_env(monkeypatch, tmp_path)

    assert main(["teacher", "label", "--set", "gold", "--keep-going"]) == 2
    assert "--keep-going" in capsys.readouterr().err
