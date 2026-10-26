"""Settings come from PP_-prefixed environment variables and never show secrets."""

from __future__ import annotations

from pathlib import Path

import pytest

from patchpulse.config import SettingsError, get_lab_settings, get_settings, scrub_secrets

REQUIRED = {
    "PP_DB_HOST": "sql.example.net",
    "PP_DB_NAME": "patchpulse",
    "PP_DB_USER": "pp_writer",
    "PP_DB_PASSWORD": "hunter2-db",
}


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    import os

    for name in list(os.environ):
        if name.startswith("PP_"):
            monkeypatch.delenv(name)
    return monkeypatch


def test_settings_read_pp_prefixed_env_vars(clean_env: pytest.MonkeyPatch) -> None:
    for name, value in REQUIRED.items():
        clean_env.setenv(name, value)
    clean_env.setenv("PP_DB_TRUST_CERT", "true")

    settings = get_settings(env_file=None)

    assert settings.db_host == "sql.example.net"
    assert settings.db_port == 1433
    assert settings.db_encrypt is True
    assert settings.db_trust_cert is True
    assert settings.db_password.get_secret_value() == "hunter2-db"
    assert settings.backfill_budget_minutes == 30


def test_secrets_are_not_shown_in_repr(clean_env: pytest.MonkeyPatch) -> None:
    for name, value in REQUIRED.items():
        clean_env.setenv(name, value)
    clean_env.setenv("PP_AUTHOR_HASH_SALT", "hunter2-salt")

    settings = get_settings(env_file=None)

    assert "hunter2" not in repr(settings)
    assert "hunter2" not in str(settings)


def test_settings_read_the_qdrant_variables(clean_env: pytest.MonkeyPatch) -> None:
    for name, value in REQUIRED.items():
        clean_env.setenv(name, value)

    assert get_settings(env_file=None).qdrant_url is None  # no Qdrant: the index stage is skipped
    assert get_settings(env_file=None).index_budget_minutes == 5
    clean_env.setenv("PP_QDRANT_URL", "https://cluster.example.cloud:6333")
    clean_env.setenv("PP_QDRANT_API_KEY", "hunter2-qdrant")
    clean_env.setenv("PP_ENCODER_DIR", "/app/encoder")
    clean_env.setenv("PP_INDEX_BUDGET_MINUTES", "0")

    settings = get_settings(env_file=None)
    lab = get_lab_settings(env_file=None)

    assert settings.qdrant_url == "https://cluster.example.cloud:6333"
    assert settings.qdrant_api_key is not None
    assert settings.qdrant_api_key.get_secret_value() == "hunter2-qdrant"
    assert settings.encoder_dir == Path("/app/encoder")
    assert settings.index_budget_minutes == 0
    assert "hunter2" not in repr(settings)
    assert lab.qdrant_url == "https://cluster.example.cloud:6333"
    assert lab.qdrant_api_key is not None
    assert lab.qdrant_local_url == "http://localhost:6333"
    assert "hunter2" not in repr(lab)


def test_missing_required_setting_names_the_env_var(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("PP_DB_NAME", "patchpulse")

    with pytest.raises(SettingsError) as error:
        get_settings(env_file=None)

    assert "PP_DB_HOST" in str(error.value)
    assert "PP_DB_PASSWORD" in str(error.value)
    assert "PP_DB_NAME" not in str(error.value)


def test_settings_read_the_agent_variables(clean_env: pytest.MonkeyPatch) -> None:
    # Phase 5 (design D10): no key means the reports stage is skipped; 0 minutes turns it off.
    for name, value in REQUIRED.items():
        clean_env.setenv(name, value)

    defaults = get_settings(env_file=None)
    assert defaults.gemini_api_key is None
    assert defaults.reports_budget_minutes == 3
    clean_env.setenv("PP_GEMINI_API_KEY", "hunter2-gemini")
    clean_env.setenv("PP_REPORTS_BUDGET_MINUTES", "0")

    settings = get_settings(env_file=None)

    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == "hunter2-gemini"
    assert settings.reports_budget_minutes == 0
    assert "hunter2" not in repr(settings)


def test_settings_read_the_langfuse_variables(clean_env: pytest.MonkeyPatch) -> None:
    # Phase 5 Task 18: the agent's traces. No keys: the reports aren't traced.
    for name, value in REQUIRED.items():
        clean_env.setenv(name, value)
    assert get_settings(env_file=None).langfuse_secret_key is None
    clean_env.setenv("PP_LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    clean_env.setenv("PP_LANGFUSE_SECRET_KEY", "hunter2-hunter2")
    clean_env.setenv("PP_LANGFUSE_HOST", "https://cloud.langfuse.com")

    settings = get_settings(env_file=None)
    lab = get_lab_settings(env_file=None)

    for loaded in (settings, lab):
        assert loaded.langfuse_public_key == "pk-lf-test"
        assert loaded.langfuse_secret_key is not None
        assert loaded.langfuse_secret_key.get_secret_value() == "hunter2-hunter2"
        assert loaded.langfuse_host == "https://cloud.langfuse.com"
        assert "hunter2" not in repr(loaded)
    assert scrub_secrets("refused hunter2-hunter2", settings) == "refused <langfuse>"
