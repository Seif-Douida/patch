"""Settings come from PP_-prefixed environment variables and never show secrets."""

from __future__ import annotations

import pytest

from patchpulse.config import SettingsError, get_settings

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


def test_missing_required_setting_names_the_env_var(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("PP_DB_NAME", "patchpulse")

    with pytest.raises(SettingsError) as error:
        get_settings(env_file=None)

    assert "PP_DB_HOST" in str(error.value)
    assert "PP_DB_PASSWORD" in str(error.value)
    assert "PP_DB_NAME" not in str(error.value)
