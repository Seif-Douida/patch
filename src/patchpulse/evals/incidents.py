"""The golden incident set (spec §11.4): one YAML file per case in evals/incidents, researched from
public sources and reviewed by Seif. The model is strict, so a typo in a field fails the load,
named by its file, rather than dropping the field."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from patchpulse.ingest.games import GAMES_FILE, load_games

INCIDENTS_DIR = Path("evals") / "incidents"

Category = Literal[
    "patch", "pricing_sale", "platform_policy", "launch_performance", "off_topic", "no_change"
]
Direction = Literal["negative", "positive", "none"]


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class IncidentWindow(_Strict):
    start: dt.date
    end: dt.date


class PatchRef(_Strict):
    date: dt.date
    name: str = Field(min_length=1)


class Source(_Strict):
    url: str = Field(min_length=1)
    published: dt.date
    shows: str = Field(min_length=1)


class Incident(_Strict):
    id: str = Field(min_length=1)
    game: str = Field(min_length=1)
    appid: int
    window: IncidentWindow
    question: str = Field(min_length=1)
    primary_cause: Category
    secondary_causes: list[Category]
    direction: Direction
    patch: PatchRef | None = None
    key_facts: list[str] = Field(min_length=1)
    acceptable_evidence: list[str] = Field(min_length=1)
    sources: list[Source] = Field(min_length=1)
    confidence: Literal["high", "medium", "low"]
    researched: dt.date
    notes: str | None = None

    @model_validator(mode="after")
    def _window_runs_forward(self) -> Incident:
        if self.window.end < self.window.start:
            raise ValueError("the window ends before it starts")
        return self

    @property
    def control(self) -> bool:
        """A no-event control: the right answer is that nothing changed."""
        return self.primary_cause == "no_change"


def load_incidents(directory: Path = INCIDENTS_DIR) -> list[Incident]:
    """Every incident in `directory`, by id. Each file's name is its id, and each game is one the
    site tracks."""
    tracked = {game.appid for game in load_games(GAMES_FILE) if game.tracked}
    incidents = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            incident = Incident.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        except (yaml.YAMLError, ValidationError) as error:
            raise ValueError(f"{path.name}: {error}") from error
        if incident.id != path.stem:
            raise ValueError(f"{path.name}: its id {incident.id!r} differs from the file name")
        if incident.appid not in tracked:
            raise ValueError(f"{path.name}: appid {incident.appid} is not a tracked game")
        incidents.append(incident)
    return sorted(incidents, key=lambda incident: incident.id)
