"""Apply Alembic migrations through its Python API: no alembic.ini or working directory needed."""

from __future__ import annotations

from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, Engine

SCRIPT_LOCATION = "patchpulse.db:migrations"


def _config(connection: Connection) -> Config:
    config = Config()
    config.set_main_option("script_location", SCRIPT_LOCATION)
    config.attributes["connection"] = connection
    return config


def upgrade_to_head(engine: Engine) -> None:
    with engine.begin() as connection:
        command.upgrade(_config(connection), "head")


def downgrade_to_base(engine: Engine) -> None:
    """Drop everything Alembic owns. For tests; never run against production."""
    with engine.begin() as connection:
        command.downgrade(_config(connection), "base")
