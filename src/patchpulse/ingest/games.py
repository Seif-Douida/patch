"""The tracked games, from config/games.yaml (spec §6.2: the file is the source of truth)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError

# Relative to the working directory: the repo root locally, /app in the jobs image.
GAMES_FILE = Path("config") / "games.yaml"


class GamesConfigError(ValueError):
    """games.yaml is missing, malformed or inconsistent; the message says where."""


class Role(StrEnum):
    TRACKED = "tracked"
    DONOR = "donor"
    BOTH = "both"


class Game(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    appid: PositiveInt
    name: str = Field(min_length=1)
    genres: list[str] = Field(min_length=1)
    role: Role
    backfill_start: date
    release_date: date | None = None
    developer: str | None = None
    publisher: str | None = None


def load_games(path: Path, *, today: date | None = None) -> list[Game]:
    today = today or datetime.now(UTC).date()
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise GamesConfigError(f"{path}: {error}") from error
    entries = document.get("games") if isinstance(document, dict) else None
    if not isinstance(entries, list) or not entries:
        raise GamesConfigError(f"{path}: expected a non-empty `games:` list")

    games: list[Game] = []
    seen: set[int] = set()
    for number, entry in enumerate(entries, start=1):
        try:
            game = Game.model_validate(entry)
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}"
                for issue in error.errors()
            )
            raise GamesConfigError(f"{path}: game #{number}: {problems}") from None
        if game.appid in seen:
            raise GamesConfigError(f"{path}: duplicate appid {game.appid}")
        if game.backfill_start > today:
            raise GamesConfigError(
                f"{path}: appid {game.appid}: backfill_start {game.backfill_start} is in the future"
            )
        seen.add(game.appid)
        games.append(game)
    return games
