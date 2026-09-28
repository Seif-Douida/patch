"""Local and CI only: prepare a database on the SQL Server container like Azure SQL would be.

Azure SQL supports contained database users out of the box. A SQL Server container needs the
server option switched on and the database set to partial containment. Neither can run inside a
transaction, and neither is allowed (or needed) on Azure SQL, so this lives outside the migrations.
"""

from __future__ import annotations

from sqlalchemy import Engine, text

_AZURE_SQL_DATABASE_EDITION = 5


def create_local_database(server: Engine, name: str) -> None:
    """Create `name` (if missing) with contained users enabled. `server` must be on `master`."""
    engine = server.execution_options(isolation_level="AUTOCOMMIT")
    with engine.connect() as connection:
        edition = connection.execute(text("SELECT SERVERPROPERTY('EngineEdition')")).scalar()
        if edition == _AZURE_SQL_DATABASE_EDITION:
            raise RuntimeError("create_local_database is for the local container, not Azure SQL")
        connection.exec_driver_sql("EXEC sp_configure 'contained database authentication', 1")
        connection.exec_driver_sql("RECONFIGURE")
        if connection.execute(text("SELECT DB_ID(:name)"), {"name": name}).scalar() is None:
            connection.exec_driver_sql(f"CREATE DATABASE [{name}]")
        connection.exec_driver_sql(f"ALTER DATABASE [{name}] SET CONTAINMENT = PARTIAL")
