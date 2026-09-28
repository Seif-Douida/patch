"""config/games.yaml is the source of truth for which games are tracked (spec §6.2)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from patchpulse.ingest.games import GAMES_FILE, GamesConfigError, Role, load_games

REPO_ROOT = Path(__file__).resolve().parents[2]

GAME = """
  - appid: {appid}
    name: Game {appid}
    genres: [RPG]
    role: {role}
    backfill_start: {start}
"""


def write(tmp_path: Path, *games: str) -> Path:
    path = tmp_path / "games.yaml"
    path.write_text("games:\n" + "".join(games), encoding="utf-8")
    return path


def game(appid: int = 1, role: str = "tracked", start: str = "2024-10-01") -> str:
    return GAME.format(appid=appid, role=role, start=start)


def test_games_yaml_is_valid_and_has_twelve_games() -> None:
    games = load_games(REPO_ROOT / GAMES_FILE)

    assert len(games) == 12
    appids = {g.appid for g in games}
    assert {553850, 949230, 1091500, 292030, 3751260} <= appids  # incident seeds + Seif's picks
    assert all(g.genres for g in games)


def test_games_released_after_the_default_start_backfill_from_launch() -> None:
    by_appid = {g.appid: g for g in load_games(REPO_ROOT / GAMES_FILE)}

    assert by_appid[949230].backfill_start == date(2023, 10, 24)  # Cities: Skylines II launch
    assert by_appid[3751260].backfill_start == date(2026, 9, 2)  # The Blood of Dawnwalker launch


def test_a_game_is_loaded_with_its_fields(tmp_path: Path) -> None:
    (loaded,) = load_games(write(tmp_path, game(7, role="both")), today=date(2026, 9, 28))

    assert loaded.appid == 7
    assert loaded.role is Role.BOTH
    assert loaded.backfill_start == date(2024, 10, 1)
    assert loaded.release_date is None


def test_duplicate_appids_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(GamesConfigError, match="duplicate appid 7"):
        load_games(write(tmp_path, game(7), game(7)), today=date(2026, 9, 28))


def test_unknown_role_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(GamesConfigError, match="role"):
        load_games(write(tmp_path, game(7, role="spectator")), today=date(2026, 9, 28))


def test_future_backfill_start_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(GamesConfigError, match="backfill_start"):
        load_games(write(tmp_path, game(7, start="2027-01-01")), today=date(2026, 9, 28))
