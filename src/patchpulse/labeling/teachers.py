"""The teacher candidates, from config/models.yaml (plan Task 7; limits in docs/budget.md).

Two providers (ADR-020):

- `gemini` (the default): Google's free tier. Each model's limits are read from Seif's AI Studio
  dashboard and are per project, so PatchPulse uses only `use_fraction` of each and leaves the rest
  to everything else on the project (the Phase 5 agent shares it).
- `openai_compat`: a self-hosted, OpenAI-style server with no rate limits. Its address and key live
  in `.env` (PP_LLM_SERVER_URL, PP_LLM_SERVER_KEY), never in this public file.

`json_mode` and `system_instruction` record what the provider accepted for that model in a live
test; Gemma 3 refused system instructions on Gemini. A model's labels are cached under
`cache_name`, so the same model served by two providers never shares labels.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError, model_validator

# Relative to the working directory: the repo root.
MODELS_FILE = Path("config") / "models.yaml"


class ModelConfigError(ValueError):
    """models.yaml is missing, malformed or inconsistent; the message says where."""


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["gemini", "openai_compat"] = "gemini"
    id: str = Field(min_length=1)  # the name the provider's API expects
    # Free-tier limits: required for Gemini, absent for a self-hosted server.
    rpm: PositiveInt | None = None
    tpm: PositiveInt | None = None
    rpd: PositiveInt | None = None
    json_mode: bool
    system_instruction: bool
    use_fraction: float = Field(default=0.5, gt=0, le=1)

    @model_validator(mode="after")
    def _gemini_has_limits(self) -> ModelConfig:
        missing = [name for name in ("rpm", "tpm", "rpd") if getattr(self, name) is None]
        if self.provider == "gemini" and missing:
            raise ValueError(f"a Gemini model needs its free-tier limits: {', '.join(missing)}")
        return self

    @property
    def cache_name(self) -> str:
        return self.id if self.provider == "gemini" else f"{self.provider}/{self.id}"

    def share(self, limit: int | None) -> int:
        """PatchPulse's part of a free-tier limit: `use_fraction` of it, at least 1."""
        if limit is None:
            raise ValueError(f"{self.id} has no free-tier limits ({self.provider})")
        return max(1, math.floor(limit * self.use_fraction))


class AgentModels(BaseModel):
    """The Phase 5 agent's model chain (design D8): tried in order, each a Gemini candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chain: list[str] = Field(min_length=1)


class TeacherModels(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates: dict[str, ModelConfig] = Field(min_length=1)
    # The bake-off winner (plan Task 9); None until then.
    teacher: str | None = None
    agent: AgentModels | None = None

    @model_validator(mode="after")
    def _teacher_is_a_candidate(self) -> TeacherModels:
        if self.teacher is not None and self.teacher not in self.candidates:
            raise ValueError(f"teacher {self.teacher!r} is not one of the candidates")
        return self

    @model_validator(mode="after")
    def _agent_chain_is_gemini(self) -> TeacherModels:
        for name in self.agent.chain if self.agent else []:
            if name not in self.candidates:
                raise ValueError(f"agent model {name!r} is not a candidate")
            if self.candidates[name].provider != "gemini":
                raise ValueError(f"agent model {name!r} must be a Gemini model (free tier)")
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
