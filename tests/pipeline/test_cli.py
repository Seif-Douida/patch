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

    noisy = ("httpx2", "httpcore2", "httpx", "httpcore")
    for name in noisy:  # set here, not inherited: pytest leaves the root logger at WARNING
        logging.getLogger(name).setLevel(logging.NOTSET)

    configure_logging()

    # httpx is qdrant-client's transport: it logs each request's URL, and the cluster's URL is a
    # secret (20 Oct: the laptop's upload logged it).
    assert all(logging.getLogger(name).level >= logging.WARNING for name in noisy)
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


def test_donor_candidates_needs_no_database_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Phase 4: the candidate check (and the history fetch on GitHub runners) talk to Steam only.
    from datetime import date

    from patchpulse.causal.rules import GenreGroup
    from patchpulse.pipeline import cli
    from patchpulse.pipeline.donors import CandidateCheck

    for name in list(os.environ):
        if name.startswith("PP_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)  # no .env here
    games = tmp_path / "games.yaml"
    games.write_text(
        "games:\n  - {appid: 553850, name: Helldivers 2, genres: [Action], role: both, "
        "backfill_start: 2024-10-01}\n",
        encoding="utf-8",
    )
    check = CandidateCheck(
        1001,
        "Passes",
        ("Action",),
        GenreGroup.ACTION_RPG,
        date(2020, 3, 4),
        25.0,
        True,
        1,
        (date(2025, 3, 10),),
        "pass",
        (),
    )
    seen: list[list[int]] = []

    def fake_check(http: object, appids: list[int], tracked: list[object], **_: object) -> object:
        seen.append(list(appids))
        assert [g.appid for g in tracked] == [553850]  # type: ignore[attr-defined]
        return [check], {"t1": 3}

    monkeypatch.setattr(cli, "check_candidates", fake_check)

    assert main(["donor-candidates", "1001", "--games-file", str(games)]) == 0
    assert seen == [[1001]]
    out = capsys.readouterr().out
    assert "Passes" in out
    assert "1 bundles of tracked patches; 1 with fewer than 5 qualifying candidates" in out


def test_donor_history_plan_and_merge_need_no_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from datetime import date

    from patchpulse.ingest.day_counts import DayCounts
    from patchpulse.pipeline.donor_history import write_history_csv

    for name in list(os.environ):
        if name.startswith("PP_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    games = tmp_path / "games.yaml"
    games.write_text(
        "games:\n  - {appid: 1245620, name: Elden Ring, genres: [Action], role: donor, "
        "backfill_start: 2024-10-01}\n",
        encoding="utf-8",
    )
    write_history_csv(
        [DayCounts(1245620, date(2024, 10, d), 3, 1) for d in (1, 2)],
        tmp_path / "artifacts" / "donor-1245620" / "donor-1245620.csv",
    )

    assert main(["donor-history", "plan", "--games-file", str(games), "--last", "2024-10-02"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "last_day=2024-10-02",
        'matrix=[{"appid": 1245620, "first": "2024-10-01"}]',
    ]
    seed = tmp_path / "seed.csv"
    args = ["donor-history", "merge", str(tmp_path / "artifacts"), "--games-file", str(games)]
    assert main([*args, "--last", "2024-10-02", "--out", str(seed)]) == 0
    assert seed.read_text(encoding="utf-8").splitlines()[1:] == [
        "1245620,2024-10-01,3,1",
        "1245620,2024-10-02,3,1",
    ]
    assert main([*args, "--last", "2024-10-03", "--out", str(seed)]) == 1  # a day missing
