"""The least-privilege database user the nightly job runs as (spec §12: `pp_writer`).

`pp-migrate` connects as the server admin, applies migrations and calls `provision_writer`; every
other job connects as `pp_writer`. The writer can read and write the raw and ops tables but not
change them, and can create and replace dbt's tables and views in stg, core and mart. It has no
server-level rights and cannot create schemas.

The user is contained in the database (it logs in with the database name, no server login), which
Azure SQL supports natively. A SQL Server container needs contained authentication switched on
first; see `patchpulse.db.local`.
"""

from __future__ import annotations

from sqlalchemy import Connection, text

WRITER_USER = "pp_writer"
_LOADED_BY_PIPELINE = ("raw", "ops")
_BUILT_BY_DBT = ("stg", "core", "mart")


def provision_writer(connection: Connection, *, password: str) -> None:
    """Create `pp_writer` or re-sync its password, then (re)apply its grants. Idempotent."""
    if not password:
        raise ValueError("the writer password must not be empty")
    exists = connection.execute(
        text("SELECT 1 FROM sys.database_principals WHERE name = :name"), {"name": WRITER_USER}
    ).first()
    verb = "ALTER" if exists else "CREATE"
    # DDL can't take bind parameters, so the literal is escaped; exec_driver_sql keeps SQLAlchemy
    # from reading ":word" in a password as a bind parameter.
    literal = password.replace("'", "''")
    connection.exec_driver_sql(f"{verb} USER [{WRITER_USER}] WITH PASSWORD = N'{literal}'")

    for schema in _LOADED_BY_PIPELINE:
        connection.exec_driver_sql(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON SCHEMA::[{schema}] TO [{WRITER_USER}]"
        )
    for schema in _BUILT_BY_DBT:
        connection.exec_driver_sql(
            f"GRANT SELECT, INSERT, UPDATE, DELETE, ALTER ON SCHEMA::[{schema}] TO [{WRITER_USER}]"
        )
    connection.exec_driver_sql(f"GRANT CREATE TABLE, CREATE VIEW TO [{WRITER_USER}]")
    # dbt-sqlserver reads view dependencies before replacing a view; Microsoft documents that this
    # catalog view needs both grants (metadata only, no data). Found by the end-to-end dbt test.
    connection.exec_driver_sql(f"GRANT VIEW DEFINITION TO [{WRITER_USER}]")
    connection.exec_driver_sql(
        f"GRANT SELECT ON sys.sql_expression_dependencies TO [{WRITER_USER}]"
    )
