"""The `patchpulse-pipeline` command the jobs image runs."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
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
    assert "--publish" in out


def test_publish_needs_the_github_settings_before_touching_the_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # In Azure a missing setting must fail the night loudly, not skip publishing quietly.
    for name in list(os.environ):
        if name.startswith("PP_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    for name in ("PP_DB_HOST", "PP_DB_NAME", "PP_DB_USER", "PP_DB_PASSWORD", "PP_AUTHOR_HASH_SALT"):
        monkeypatch.setenv(name, "unused")  # a connection attempt would hang on this host

    assert main(["nightly", "--publish"]) == 2
    err = capsys.readouterr().err
    for name in ("PP_GITHUB_REPOSITORY", "PP_GITHUB_APP_ID", "PP_GITHUB_APP_PRIVATE_KEY"):
        assert name in err


class DatabaseTouchedError(Exception):
    """Raised instead of connecting: the check under test has to come before the database."""


@pytest.fixture(scope="module")
def app_key() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ).decode()


def use_github_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, key: str) -> None:
    """Every setting `nightly --publish` needs, and a database that must not be reached."""
    for name in list(os.environ):
        if name.startswith("PP_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    for name in ("PP_DB_HOST", "PP_DB_NAME", "PP_DB_USER", "PP_DB_PASSWORD", "PP_AUTHOR_HASH_SALT"):
        monkeypatch.setenv(name, "unused")
    monkeypatch.setenv("PP_GITHUB_REPOSITORY", "Seif-Douida/patch")
    monkeypatch.setenv("PP_GITHUB_APP_ID", "123456")
    monkeypatch.setenv("PP_GITHUB_APP_PRIVATE_KEY", key)

    def no_database(*args: object, **kwargs: object) -> Engine:
        raise DatabaseTouchedError

    monkeypatch.setattr("patchpulse.pipeline.cli.make_engine", no_database)


def test_publish_rejects_a_key_without_its_pem_lines_before_touching_the_database(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    app_key: str,
) -> None:
    # 2026-10-06: the secret was pasted without its BEGIN/END lines. The night ran for 31 minutes,
    # then failed at publish with PyJWT's misleading "Could not parse the provided public key".
    body = "\n".join(app_key.strip().splitlines()[1:-1])
    use_github_app(monkeypatch, tmp_path, body)

    assert main(["nightly", "--publish"]) == 2
    err = capsys.readouterr().err
    assert "PP_GITHUB_APP_PRIVATE_KEY" in err
    assert "-----BEGIN" in err
    assert body.splitlines()[0] not in err  # the key is never echoed


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_publish_accepts_a_whole_pem_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, app_key: str, newline: str
) -> None:
    use_github_app(monkeypatch, tmp_path, app_key.replace("\n", newline))

    with pytest.raises(DatabaseTouchedError):
        main(["nightly", "--publish"])


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
