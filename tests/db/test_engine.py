"""Connecting to Azure SQL survives the free database resuming from auto-pause (spec §12)."""

from __future__ import annotations

import pytest
from mssql_python.exceptions import OperationalError as DriverOperationalError
from sqlalchemy.exc import OperationalError

from patchpulse.config import Settings
from patchpulse.db.engine import (
    connect_with_resume_retry,
    is_transient,
    make_engine,
    make_entra_engine,
)

# Microsoft's text for error 40613, returned while a paused serverless database resumes.
RESUMING = (
    "[Microsoft][SQL Server]Database 'patchpulse' on server 'sql-patchpulse' is not currently "
    "available. Please retry the connection later. If the problem persists, contact customer "
    "support, and provide them the session tracing ID of '0000'."
)
BUSY = "[Microsoft][SQL Server]The service is currently busy. Retry the request after 10 seconds."
LOGIN_FAILED = "[Microsoft][SQL Server]Login failed for user 'pp_writer'."


def db_error(ddbc_error: str) -> OperationalError:
    driver = DriverOperationalError("Invalid authorization specification", ddbc_error)
    return OperationalError("connect", None, driver)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.waits: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


class FlakyEngine:
    """Raises the given errors in order, then returns a connection."""

    def __init__(self, *errors: OperationalError) -> None:
        self.errors = list(errors)
        self.calls = 0

    def connect(self) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "connection"


def settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "db_host": "sql.example.net",
        "db_name": "patchpulse",
        "db_user": "pp_writer",
        "db_password": "hunter2-db",
    }
    values.update(overrides)
    return Settings.model_validate(values)


def test_connect_retries_while_the_database_resumes() -> None:
    clock = FakeClock()
    engine = FlakyEngine(db_error(RESUMING), db_error(RESUMING))

    connection = connect_with_resume_retry(engine, sleep=clock.sleep, clock=clock.monotonic)

    assert connection == "connection"
    assert engine.calls == 3
    assert clock.waits == sorted(clock.waits)  # backoff never shrinks


def test_non_transient_errors_are_not_retried() -> None:
    clock = FakeClock()
    engine = FlakyEngine(db_error(LOGIN_FAILED))

    with pytest.raises(OperationalError, match="Login failed"):
        connect_with_resume_retry(engine, sleep=clock.sleep, clock=clock.monotonic)

    assert engine.calls == 1
    assert clock.waits == []


def test_retry_gives_up_after_the_wait_budget() -> None:
    clock = FakeClock()
    engine = FlakyEngine(*[db_error(RESUMING) for _ in range(50)])

    with pytest.raises(OperationalError, match="not currently available"):
        connect_with_resume_retry(engine, max_wait_s=90, sleep=clock.sleep, clock=clock.monotonic)

    assert 60 <= clock.now <= 90


@pytest.mark.parametrize(
    ("message", "transient"),
    [
        (RESUMING, True),
        (BUSY, True),
        ("Client unable to establish connection because an error was encountered", True),
        (LOGIN_FAILED, False),
        ("[Microsoft][SQL Server]Invalid object name 'raw.game'.", False),
    ],
)
def test_transient_errors_are_recognised_by_their_documented_text(
    message: str, transient: bool
) -> None:
    assert is_transient(db_error(message)) is transient


def test_make_engine_builds_an_encrypted_mssqlpython_url() -> None:
    engine = make_engine(settings())

    assert engine.url.drivername == "mssql+mssqlpython"
    assert engine.url.query["Encrypt"] == "yes"
    assert engine.url.query["TrustServerCertificate"] == "no"
    assert "hunter2" not in str(engine.url)


def test_make_engine_trusts_the_certificate_only_when_asked() -> None:
    engine = make_engine(settings(db_trust_cert=True))

    assert engine.url.query["TrustServerCertificate"] == "yes"


def test_entra_engine_reuses_az_login_and_holds_no_password() -> None:
    # Phase 2's `pull` signs in as Seif through his `az login` (DefaultAzureCredential); device
    # code sign-in is blocked by the tenant's security defaults.
    engine = make_entra_engine("sql-x.database.windows.net", "patchpulse")

    assert engine.url.drivername == "mssql+mssqlpython"
    assert engine.url.query["Authentication"] == "ActiveDirectoryDefault"
    assert engine.url.query["Encrypt"] == "yes"
    assert engine.url.password is None
    assert engine.url.database == "patchpulse"
