"""Pipeline settings, read from PP_-prefixed environment variables (and `.env` locally).

In Azure every value arrives as a Container Apps env var; secrets come from Container Apps
secrets, which Bicep fills from GitHub environment secrets. Nothing secret is stored in files.
"""

from __future__ import annotations

from pydantic import SecretStr, ValidationError
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


def get_settings(*, env_file: str | None = ".env") -> Settings:
    """Load settings; a missing or invalid value raises `SettingsError` naming its env var."""
    try:
        return Settings(_env_file=env_file)  # type: ignore[call-arg]
    except ValidationError as error:
        names = sorted({f"{ENV_PREFIX}{str(issue['loc'][0]).upper()}" for issue in error.errors()})
        raise SettingsError(f"missing or invalid settings: {', '.join(names)}") from None
