"""SQLAlchemy engine for SQL Server / Azure SQL through Microsoft's mssql-python driver.

The free Azure SQL database pauses after 60 idle minutes. The first connection while it resumes
fails with error 40613, so connections retry transient errors for up to ~90 seconds (spec §12).
mssql-python reports only SQL Server's message text, not the error number, so transient errors
are recognised by Microsoft's documented wording.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL
from sqlalchemy.exc import DBAPIError

from patchpulse.config import Settings

# Documented Azure SQL transient conditions, matched case-insensitively on the message text.
_TRANSIENT_MARKERS = (
    "is not currently available",  # 40613: database resuming or failing over
    "please retry the connection later",  # 40613
    "the service is currently busy",  # 40501
    "the service has encountered an error processing your request",  # 40197
    "not enough resources to process request",  # 49918
    "too many create or update operations in progress",  # 49919
    "too many operations in progress",  # 49920
    "client unable to establish connection",  # driver: the server did not answer (yet)
)
_FIRST_WAIT_S = 2.0
_MAX_SINGLE_WAIT_S = 30.0


class SupportsConnect[T](Protocol):
    def connect(self) -> T: ...


def make_engine(settings: Settings) -> Engine:
    url = URL.create(
        "mssql+mssqlpython",
        username=settings.db_user,
        password=settings.db_password.get_secret_value(),
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
        query={
            "Encrypt": "yes" if settings.db_encrypt else "no",
            "TrustServerCertificate": "yes" if settings.db_trust_cert else "no",
        },
    )
    return create_engine(url, pool_pre_ping=True)


def make_entra_engine(host: str, database: str) -> Engine:
    """An engine that signs in as the person at the keyboard, for Phase 2's local `pull`.

    `ActiveDirectoryDefault` lets mssql-python use azure-identity's DefaultAzureCredential, which
    reuses a working `az login`. Device-code sign-in is blocked by the tenant's security defaults.
    """
    url = URL.create(
        "mssql+mssqlpython",
        host=host,
        port=1433,
        database=database,
        query={
            "Authentication": "ActiveDirectoryDefault",
            "Encrypt": "yes",
            "TrustServerCertificate": "no",
        },
    )
    return create_engine(url, pool_pre_ping=True)


def is_transient(error: BaseException) -> bool:
    if not isinstance(error, DBAPIError):
        return False
    message = str(error.orig).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


def connect_with_resume_retry[T](
    engine: SupportsConnect[T],
    *,
    max_wait_s: float = 90.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> T:
    """Connect, retrying transient errors with capped exponential backoff for `max_wait_s`."""
    started = clock()
    wait = _FIRST_WAIT_S
    while True:
        try:
            return engine.connect()
        except DBAPIError as error:
            if not is_transient(error) or clock() - started + wait > max_wait_s:
                raise
            sleep(wait)
            wait = min(wait * 2, _MAX_SINGLE_WAIT_S)
