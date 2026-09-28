"""Pipeline run bookkeeping in ops.pipeline_run. A run's id is also its data_version (spec §6.4)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from sqlalchemy import Connection, text


def start_run(connection: Connection, *, kind: str, image_version: str | None) -> int:
    run_id: int = connection.execute(
        text(
            "INSERT INTO ops.pipeline_run (kind, status, image_version) "
            "OUTPUT INSERTED.run_id VALUES (:kind, 'running', :image_version)"
        ),
        {"kind": kind, "image_version": image_version},
    ).scalar_one()
    return run_id


def finish_run(
    connection: Connection,
    run_id: int,
    *,
    status: str,
    stages: Mapping[str, Any],
    error: str | None,
) -> None:
    connection.execute(
        text(
            "UPDATE ops.pipeline_run SET finished_at = SYSUTCDATETIME(), status = :status, "
            "stages = :stages, error = :error WHERE run_id = :run_id"
        ),
        {
            "status": status,
            "stages": json.dumps(stages, sort_keys=True, default=str),
            "error": error[:4000] if error else None,
            "run_id": run_id,
        },
    )
