"""Alembic environment: always runs on the connection handed over by patchpulse.db.migrate."""

from __future__ import annotations

from alembic import context

connection = context.config.attributes["connection"]
context.configure(connection=connection, transaction_per_migration=True)
with context.begin_transaction():
    context.run_migrations()
