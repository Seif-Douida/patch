"""The API's settings (Phase 5, design D9), from PP_-prefixed environment variables.

In Azure each arrives as a Container Apps env var on `pp-api`; the secrets (the Gemini key,
Qdrant's URL and key, Langfuse's secret key) come from Container Apps secrets. Everything is
optional: without a value the API still answers /health and /reports, and Ask says it's
unavailable. The Qdrant URL is a secret too (with the key, it's the cluster's only door).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# The orphan branch the nightly job publishes the bundle and the reports to (a public repository:
# both are aggregates and public copies only).
AGENT_DATA_URL = "https://raw.githubusercontent.com/Seif-Douida/patch/refs/heads/agent-data"


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PP_", extra="ignore")

    gemini_api_key: SecretStr | None = None
    qdrant_url: SecretStr | None = None
    qdrant_api_key: SecretStr | None = None
    encoder_dir: Path | None = None
    agent_data_url: str = AGENT_DATA_URL
    # The site's origin, "https://<host>": the only one allowed to read the API from a browser.
    site_origin: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str | None = None
    # Ask's limits (design D9): questions per visitor an hour, and a day's questions in all. The
    # cap stays 15 until Task 17 measures the compute per wake.
    ask_per_visitor_hour: int = 3
    ask_daily_cap: int = 15
