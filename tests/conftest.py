"""Shared fixtures for integration tests against a real SQL Server.

Locally that's the compose container (`docker compose up -d sql`); in CI it's a service container.
Each test session works in its own throwaway database. Without a reachable server the tests skip
on a dev machine but fail in CI, so the integration suite can never silently not run.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from dotenv import dotenv_values
from sqlalchemy import Engine, text
from sqlalchemy.exc import DBAPIError

from patchpulse.config import Settings
from patchpulse.db.engine import make_engine
from patchpulse.db.local import create_local_database

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sa_password() -> str | None:
    return os.environ.get("PP_TEST_SA_PASSWORD") or dotenv_values(REPO_ROOT / ".env").get(
        "PP_TEST_SA_PASSWORD"
    )


def _unavailable(reason: str) -> None:
    if os.environ.get("CI"):
        pytest.fail(f"SQL Server is required in CI: {reason}")
    pytest.skip(f"local SQL Server not available ({reason}); run `docker compose up -d sql`")


def admin_settings(database: str) -> Settings:
    password = _sa_password()
    if password is None:
        _unavailable("PP_TEST_SA_PASSWORD is not set")
    return Settings.model_validate(
        {
            "db_host": os.environ.get("PP_TEST_DB_HOST", "localhost"),
            "db_name": database,
            "db_user": "sa",
            "db_password": password,
            "db_trust_cert": True,
        }
    )


@pytest.fixture(scope="session")
def sql_server() -> Iterator[Engine]:
    """An autocommit engine on `master`, for creating and dropping test databases."""
    engine = make_engine(admin_settings("master")).execution_options(isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except DBAPIError as error:
        _unavailable(str(error.orig).splitlines()[0])
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def new_database(sql_server: Engine) -> Iterator[Callable[[], Settings]]:
    """Factory: each call creates an empty database and returns admin settings for it."""
    created: list[str] = []

    def create() -> Settings:
        name = f"pp_test_{uuid.uuid4().hex[:12]}"
        create_local_database(sql_server, name)
        created.append(name)
        return admin_settings(name)

    yield create
    with sql_server.connect() as connection:
        for name in created:
            connection.execute(
                text(f"ALTER DATABASE [{name}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE")
            )
            connection.execute(text(f"DROP DATABASE [{name}]"))


@pytest.fixture(scope="session")
def sql_settings(new_database: Callable[[], Settings]) -> Settings:
    """Admin settings for one database shared by the whole session."""
    return new_database()


@pytest.fixture
def migrated_engine(new_database: Callable[[], Settings]) -> Iterator[Engine]:
    """An admin engine on a fresh database with every migration applied."""
    from patchpulse.db.migrate import upgrade_to_head

    engine = make_engine(new_database())
    upgrade_to_head(engine)
    yield engine
    engine.dispose()
