"""Pipeline settings, read from PP_-prefixed environment variables (and `.env` locally).

In Azure every value arrives as a Container Apps env var; secrets come from Container Apps
secrets, which Bicep fills from GitHub environment secrets. Nothing secret is stored in files.
"""

from __future__ import annotations

from pathlib import Path

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
    # DagsHub's hosted MLflow (design D7).
    mlflow_tracking_uri: str | None = Field(default=None, validation_alias="MLFLOW_TRACKING_URI")
    mlflow_tracking_username: str | None = Field(
        default=None, validation_alias="MLFLOW_TRACKING_USERNAME"
    )
    mlflow_tracking_password: SecretStr | None = Field(
        default=None, validation_alias="MLFLOW_TRACKING_PASSWORD"
    )


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
