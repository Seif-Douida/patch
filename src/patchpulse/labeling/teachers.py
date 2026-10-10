"""The teacher candidates, from config/models.yaml (plan Task 7; limits in docs/budget.md).

Each model's free-tier limits are read from Seif's AI Studio dashboard: they are per project, so
PatchPulse uses only `use_fraction` of each and leaves the rest to everything else on the project
(the Phase 5 agent shares it). `json_mode` and `system_instruction` record what the Gemini API
accepted for that model in the live smoke test; Gemma 3 refused system instructions there.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError, model_validator

# Relative to the working directory: the repo root.
MODELS_FILE = Path("config") / "models.yaml"


class ModelConfigError(ValueError):
    """models.yaml is missing, malformed or inconsistent; the message says where."""


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    rpm: PositiveInt
    tpm: PositiveInt
    rpd: PositiveInt
    json_mode: bool
    system_instruction: bool
    use_fraction: float = Field(gt=0, le=1)

    def share(self, limit: int) -> int:
        """PatchPulse's part of a free-tier limit: `use_fraction` of it, at least 1."""
        return max(1, math.floor(limit * self.use_fraction))


class TeacherModels(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates: dict[str, ModelConfig] = Field(min_length=1)
    # The bake-off winner (plan Task 9); None until then.
    teacher: str | None = None

    @model_validator(mode="after")
    def _teacher_is_a_candidate(self) -> TeacherModels:
        if self.teacher is not None and self.teacher not in self.candidates:
            raise ValueError(f"teacher {self.teacher!r} is not one of the candidates")
        return self


def load_model_config(path: Path = MODELS_FILE) -> TeacherModels:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        return TeacherModels.model_validate(document)
    except (OSError, yaml.YAMLError) as error:
        raise ModelConfigError(f"{path}: {error}") from error
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in issue['loc']) or 'file'}: {issue['msg']}"
            for issue in error.errors()
        )
        raise ModelConfigError(f"{path}: {problems}") from None
