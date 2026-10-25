"""Pipeline settings, read from PP_-prefixed environment variables (and `.env` locally).

In Azure every value arrives as a Container Apps env var; secrets come from Container Apps
secrets, which Bicep fills from GitHub environment secrets. Nothing secret is stored in files.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PREFIX = "PP_"


class SettingsError(RuntimeError):
    """Raised when required settings are missing or invalid; names the env vars to set."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, extra="ignore")

    db_host: str
    db_port: int = 1433
    db_name: str
    db_user: str
    db_password: SecretStr
    db_encrypt: bool = True
    # Only the local and CI containers use a self-signed certificate.
    db_trust_cert: bool = False
    # Used by `migrate` only: that job connects as the admin and sets the writer's password.
    writer_password: SecretStr | None = None
    # Never rotate: author hashes are the deduplication key for review authors.
    author_hash_salt: SecretStr | None = None
    backfill_budget_minutes: int = 30
    # Phase 2 scoring (design D8): the champion baked into the jobs image, and the stage's
    # time budget. No model directory means the stage is skipped.
    model_dir: Path | None = None
    score_budget_minutes: int = 5
    # Phase 3's index stage (design D5): the free Qdrant cluster (its URL is treated as secret too:
    # with the key, it's the cluster's only door), the encoder baked into the jobs image, and the
    # stage's time budget. No URL or encoder means the stage is skipped; 0 minutes turns it off.
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    encoder_dir: Path | None = None
    index_budget_minutes: int = 5
    # Phase 4's effects stage (design D10): estimating the patches whose window ended or whose
    # inputs changed; what's left waits a night. 0 turns it off.
    effects_budget_minutes: int = 2
    # Phase 5's reports stage (design D10): the free-tier Gemini key the agent's models answer
    # with, and the stage's time budget. No key means the stage is skipped; 0 turns it off.
    gemini_api_key: SecretStr | None = None
    reports_budget_minutes: int = 3
    github_repository: str | None = None
    github_app_id: str | None = None
    github_app_private_key: SecretStr | None = None


class LabSettings(BaseSettings):
    """Phase 2's local work on Seif's machine (`patchpulse-ml`): the snapshot, labeling, training.

    MLflow's settings keep MLflow's own unprefixed names; `models.tracking` hands them to MLflow.
    Nothing here reaches Azure.
    """

    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, extra="ignore")

    # The Azure SQL server's host name (not secret); only `pull` needs it.
    pull_host: str | None = None
    pull_database: str = "patchpulse"
    # Review text lives here and is never committed (.gitignore).
    data_dir: Path = Path("data/local")
    # Review ids, splits and labels only: committed (spec C4).
    gold_dir: Path = Path("data/gold")
    gemini_api_key: SecretStr | None = None
    # A self-hosted, OpenAI-style teacher server (ADR-020): its /v1 address and key, not committed.
    llm_server_url: str | None = None
    llm_server_key: SecretStr | None = None
    # DagsHub's hosted MLflow (design D7).
    mlflow_tracking_uri: str | None = Field(default=None, validation_alias="MLFLOW_TRACKING_URI")
    mlflow_tracking_username: str | None = Field(
        default=None, validation_alias="MLFLOW_TRACKING_USERNAME"
    )
    mlflow_tracking_password: SecretStr | None = Field(
        default=None, validation_alias="MLFLOW_TRACKING_PASSWORD"
    )
    # Phase 3: the free Qdrant cluster (the upload, cloud measurements) and the local container
    # that holds the eval's collections.
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_local_url: str = "http://localhost:6333"


def scrub_secrets(detail: str, settings: Settings) -> str:
    """`detail` without the Qdrant cluster's URL, host or key, or the Gemini key: they're secrets,
    and a stage's error can carry them into logs and ops.data_quality_result."""
    url = settings.qdrant_url or ""
    found = [(url, "<qdrant>"), (urlsplit(url).hostname or "", "<qdrant>")]
    if settings.qdrant_api_key is not None:
        found.append((settings.qdrant_api_key.get_secret_value(), "<qdrant>"))
    if settings.gemini_api_key is not None:
        found.append((settings.gemini_api_key.get_secret_value(), "<gemini>"))
    for secret, label in sorted(found, key=lambda pair: len(pair[0]), reverse=True):
        if secret:
            detail = detail.replace(secret, label)
    return detail


def _load[T: BaseSettings](cls: type[T], env_file: str | None) -> T:
    try:
        return cls(_env_file=env_file)
    except ValidationError as error:
        names = sorted({f"{ENV_PREFIX}{str(issue['loc'][0]).upper()}" for issue in error.errors()})
        raise SettingsError(f"missing or invalid settings: {', '.join(names)}") from None


def get_settings(*, env_file: str | None = ".env") -> Settings:
    """Load settings; a missing or invalid value raises `SettingsError` naming its env var."""
    return _load(Settings, env_file)


def get_lab_settings(*, env_file: str | None = ".env") -> LabSettings:
    """Load the local Phase 2 settings, naming any invalid env var like `get_settings`."""
    return _load(LabSettings, env_file)
