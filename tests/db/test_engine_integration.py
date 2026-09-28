"""The engine reaches a real SQL Server (the local container, or the CI service container)."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from patchpulse.config import Settings
from patchpulse.db.engine import connect_with_resume_retry, make_engine

pytestmark = pytest.mark.integration


def test_engine_connects_to_the_local_container(sql_settings: Settings) -> None:
    engine = make_engine(sql_settings)

    with connect_with_resume_retry(engine) as connection:
        # SQLAlchemy types a text() row as zero columns, so the scalar needs an explicit type.
        database: str = connection.execute(text("SELECT DB_NAME()")).scalar_one()

    assert database == sql_settings.db_name
