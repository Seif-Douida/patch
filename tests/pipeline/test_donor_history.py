"""The comparison games' history: fetched once per game on a GitHub runner, merged into the dbt seed
on the laptop (Phase 4, design D4)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from fake_steam import FakeSteam

from patchpulse.ingest.day_counts import DayCounts
from patchpulse.ingest.games import Game, Role
from patchpulse.ingest.http import SteamUnavailableError
from patchpulse.pipeline.donor_history import (
    PAUSE_S,
    fetch_history,
    history_plan,
    merge_history,
    read_history_csv,
    write_history_csv,
)

APP = 1245620
D1 = date(2024, 10, 1)


def day(offset: int) -> date:
    return D1 + timedelta(days=offset)


def game(appid: int, role: Role = Role.DONOR, start: date = D1) -> Game:
    return Game(
        appid=appid, name=f"Game {appid}", genres=["Action"], role=role, backfill_start=start
    )


def test_fetch_writes_one_row_a_day(tmp_path: Path) -> None:
    steam = FakeSteam()
    for rid, (offset, up) in enumerate([(0, True), (0, False), (2, True)], start=1):
        steam.add_review(APP, rid, datetime(2024, 10, 1 + offset, 12, tzinfo=UTC), voted_up=up)

    rows = fetch_history(steam.http(), APP, day(0), day(2))
    write_history_csv(rows, tmp_path / "donor.csv")

    assert rows == [
        DayCounts(APP, day(0), 1, 1),
        DayCounts(APP, day(1), 0, 0),
        DayCounts(APP, day(2), 1, 0),
    ]
    assert (tmp_path / "donor.csv").read_text(encoding="utf-8").splitlines() == [
        "appid,day,positive,negative",
        "1245620,2024-10-01,1,1",
        "1245620,2024-10-02,0,0",
        "1245620,2024-10-03,1,0",
    ]
    assert read_history_csv(tmp_path / "donor.csv") == rows


def test_fetch_pauses_then_gives_up_on_429() -> None:
    steam = FakeSteam()
    steam.rate_limit_after = 1  # every request after the first answers HTTP 429
    pauses: list[float] = []

    with pytest.raises(SteamUnavailableError):
        fetch_history(steam.http(), APP, day(0), day(5), sleep=pauses.append)

    assert pauses == [PAUSE_S] * 3


def test_fetch_resumes_after_a_pause() -> None:
    steam = FakeSteam()
    steam.rate_limit_after = 2
    pauses: list[float] = []

    def wait(seconds: float) -> None:
        pauses.append(seconds)
        steam.rate_limit_after = None  # Steam lets us back in

    rows = fetch_history(steam.http(), APP, day(0), day(4), sleep=wait)

    assert [r.day for r in rows] == [day(n) for n in range(5)]
    assert pauses == [PAUSE_S]


def test_merge_refuses_gaps_and_duplicates(tmp_path: Path) -> None:
    write_history_csv(
        [DayCounts(1, day(0), 1, 0), DayCounts(1, day(2), 1, 0)],
        tmp_path / "donor-1" / "donor-1.csv",
    )
    write_history_csv([DayCounts(2, day(n), 1, 0) for n in (0, 1, 2, 2)], tmp_path / "donor-2.csv")
    write_history_csv(
        [DayCounts(3, day(n), -1 if n == 1 else 1, 0) for n in range(3)], tmp_path / "donor-3.csv"
    )

    with pytest.raises(ValueError, match="doesn't check out") as refused:
        merge_history(tmp_path, [game(1), game(2), game(3), game(4)], day(2))

    message = str(refused.value)
    assert "1: missing 2024-10-02" in message
    assert "2: duplicate 2024-10-03" in message
    assert "3: negative counts on 2024-10-02" in message
    assert "4: no donor-4.csv" in message


def test_merge_writes_the_seed_in_order(tmp_path: Path) -> None:
    write_history_csv([DayCounts(9, day(n), n, 1) for n in (1, 0)], tmp_path / "donor-9.csv")
    write_history_csv([DayCounts(2, day(n), 2, n) for n in (0, 1)], tmp_path / "x" / "donor-2.csv")
    tracked = game(553850, role=Role.BOTH)  # not a comparison game: no file needed

    rows = merge_history(tmp_path, [game(9), tracked, game(2)], day(1))

    assert [(r.appid, r.day) for r in rows] == [(2, day(0)), (2, day(1)), (9, day(0)), (9, day(1))]


def test_merge_starts_each_game_at_its_own_backfill_start(tmp_path: Path) -> None:
    write_history_csv([DayCounts(5, day(n), 1, 1) for n in (3, 4)], tmp_path / "donor-5.csv")

    rows = merge_history(tmp_path, [game(5, start=day(3))], day(4))

    assert [r.day for r in rows] == [day(3), day(4)]


def test_plan_lists_the_comparison_games() -> None:
    games = [game(9), game(553850, role=Role.BOTH), game(2, start=day(30))]

    lines = history_plan(games, last=None, today=date(2026, 10, 22))

    assert lines[0] == "last_day=2026-10-21"
    assert json.loads(lines[1].removeprefix("matrix=")) == [
        {"appid": 2, "first": "2024-10-31"},
        {"appid": 9, "first": "2024-10-01"},
    ]
    assert history_plan(games, last=date(2026, 10, 20), today=date(2026, 10, 22))[0] == (
        "last_day=2026-10-20"
    )
