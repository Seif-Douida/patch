"""Shared fixtures for integration tests against a real SQL Server and a real Qdrant.

Locally those are the compose containers (`docker compose up -d sql qdrant`); in CI they're
containers on the runner. Each test session works in its own throwaway database, and each Qdrant
test in its own collections. Without a reachable server the tests skip on a dev machine but fail in
CI, so the integration suite can never silently not run.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from dotenv import dotenv_values
from sqlalchemy import Engine, text
from sqlalchemy.exc import DBAPIError

from patchpulse.config import Settings
from patchpulse.db.engine import make_engine
from patchpulse.db.local import create_local_database

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sa_password() -> str | None:
    return os.environ.get("PP_TEST_SA_PASSWORD") or dotenv_values(REPO_ROOT / ".env").get(
        "PP_TEST_SA_PASSWORD"
    )


def _unavailable(reason: str, *, server: str = "SQL Server", service: str = "sql") -> None:
    if os.environ.get("CI"):
        pytest.fail(f"{server} is required in CI: {reason}")
    pytest.skip(f"local {server} not available ({reason}); run `docker compose up -d {service}`")


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


@pytest.fixture(scope="session")
def qdrant() -> Iterator[QdrantClient]:
    """A client on the local Qdrant (compose locally, a container in CI), without an API key.
    PP_TEST_QDRANT_URL and PP_TEST_QDRANT_API_KEY point it elsewhere: Phase 3 ran the Qdrant
    tests once against the free cluster before the upload."""
    from qdrant_client import QdrantClient

    client = QdrantClient(
        url=os.environ.get("PP_TEST_QDRANT_URL", "http://localhost:6333"),
        api_key=os.environ.get("PP_TEST_QDRANT_API_KEY") or None,
        timeout=30,
    )
    try:
        client.get_collections()
    except Exception as error:  # refused, timed out: whatever the transport says
        _unavailable(type(error).__name__, server="Qdrant", service="qdrant")
    yield client
    client.close()


@pytest.fixture
def new_collection(qdrant: QdrantClient) -> Iterator[Callable[[], str]]:
    """Factory: each call returns an unused collection name; the collections and any aliases
    pointing at them are deleted after the test."""
    names: list[str] = []

    def create() -> str:
        names.append(f"pp_test_{uuid.uuid4().hex[:12]}")
        return names[-1]

    yield create
    from qdrant_client import models

    qdrant.update_collection_aliases(
        change_aliases_operations=[
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=a.alias_name))
            for a in qdrant.get_aliases().aliases
            if a.collection_name in names
        ]
    )
    for name in names:
        qdrant.delete_collection(name)
