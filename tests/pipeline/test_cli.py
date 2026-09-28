"""The `patchpulse-pipeline` command the jobs image runs."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from sqlalchemy import Engine, text

from patchpulse.config import Settings
from patchpulse.db.engine import make_engine
from patchpulse.db.provision import WRITER_USER
from patchpulse.pipeline.cli import main


def use_settings(monkeypatch: pytest.MonkeyPatch, settings: Settings, **extra: str) -> None:
    for name in list(os.environ):
        if name.startswith("PP_") and name != "PP_TEST_SA_PASSWORD":
            monkeypatch.delenv(name)
    monkeypatch.setenv("PP_DB_HOST", settings.db_host)
    monkeypatch.setenv("PP_DB_NAME", settings.db_name)
    monkeypatch.setenv("PP_DB_USER", settings.db_user)
    monkeypatch.setenv("PP_DB_PASSWORD", settings.db_password.get_secret_value())
    monkeypatch.setenv("PP_DB_TRUST_CERT", "true")
    for name, value in extra.items():
        monkeypatch.setenv(name, value)


def test_help_lists_the_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["--help"])

    assert exited.value.code == 0
    out = capsys.readouterr().out
    for command in ("migrate", "nightly", "create-local-db"):
        assert command in out


def test_nightly_accepts_a_games_file_and_an_export_dir(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["nightly", "--help"])

    out = capsys.readouterr().out
    assert "--games-file" in out
    assert "--export-dir" in out


def test_http_request_logs_are_quiet() -> None:
    # One INFO line per Steam request would flood the shared 0.15 GB/day Log Analytics cap.
    import logging

    from patchpulse.pipeline.cli import configure_logging

    configure_logging()

    assert logging.getLogger("httpx2").getEffectiveLevel() >= logging.WARNING
    assert logging.getLogger("patchpulse").getEffectiveLevel() == logging.INFO


def test_missing_settings_exit_with_a_clear_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in list(os.environ):
        if name.startswith("PP_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)  # no .env here

    assert main(["nightly"]) == 2
    assert "PP_DB_HOST" in capsys.readouterr().err


@pytest.mark.integration
def test_migrate_creates_the_schema_and_the_writer(
    new_database: Callable[[], Settings], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    admin = new_database()
    use_settings(monkeypatch, admin, PP_WRITER_PASSWORD="Writer-Cli-1")
    monkeypatch.chdir(tmp_path)

    assert main(["migrate"]) == 0

    writer = make_engine(
        Settings.model_validate(
            {**admin.model_dump(), "db_user": WRITER_USER, "db_password": "Writer-Cli-1"}
        )
    )
    with writer.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM raw.game")).scalar_one() == 0


@pytest.mark.integration
def test_migrate_needs_the_writer_password(
    new_database: Callable[[], Settings],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    use_settings(monkeypatch, new_database())
    monkeypatch.chdir(tmp_path)

    assert main(["migrate"]) == 2
    assert "PP_WRITER_PASSWORD" in capsys.readouterr().err


@pytest.mark.integration
def test_create_local_db_is_idempotent(
    sql_server: Engine, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    name = "pp_test_cli_local"
    url = sql_server.url
    settings = Settings.model_validate(
        {
            "db_host": url.host,
            "db_name": name,
            "db_user": url.username,
            "db_password": url.password,
            "db_trust_cert": True,
        }
    )
    use_settings(monkeypatch, settings)
    monkeypatch.chdir(tmp_path)
    try:
        assert main(["create-local-db"]) == 0
        assert main(["create-local-db"]) == 0
        with sql_server.connect() as connection:
            assert connection.execute(text(f"SELECT DB_ID('{name}')")).scalar() is not None
    finally:
        with sql_server.connect() as connection:
            connection.execute(text(f"IF DB_ID('{name}') IS NOT NULL DROP DATABASE [{name}]"))
