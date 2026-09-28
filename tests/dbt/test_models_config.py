"""Static checks on the dbt project: history is protected, statistics are pinned (spec §6.4)."""

from __future__ import annotations

import re
from pathlib import Path

DBT = Path(__file__).resolve().parents[2] / "dbt"


def config_block(sql: str) -> str:
    match = re.search(r"\{\{\s*config\((.*?)\)\s*\}\}", sql, flags=re.DOTALL)
    return match.group(1) if match else ""


def test_incremental_core_models_cannot_be_full_refreshed() -> None:
    # raw.steam_review keeps every version, but rebuilding millions of rows would burn the free
    # SQL allowance; `dbt build --full-refresh` must never rebuild core history by accident.
    incremental = [
        path
        for path in (DBT / "models" / "core").glob("*.sql")
        if "materialized='incremental'" in config_block(path.read_text(encoding="utf-8"))
    ]
    assert incremental, "expected incremental core models"
    for path in incremental:
        assert "full_refresh=false" in config_block(path.read_text(encoding="utf-8")), path.name


def test_wilson_interval_is_90_percent() -> None:
    macro = (DBT / "macros" / "wilson_bounds.sql").read_text(encoding="utf-8")
    assert "z=1.6449" in macro


def test_schemas_are_used_exactly_as_named() -> None:
    # Without this override dbt would build "dbo_core" instead of "core".
    macro = (DBT / "macros" / "generate_schema_name.sql").read_text(encoding="utf-8")
    assert "custom_schema_name | trim" in macro


def test_profile_reads_every_connection_setting_from_pp_env_vars() -> None:
    profile = (DBT / "profiles.yml").read_text(encoding="utf-8")
    for name in ("PP_DB_HOST", "PP_DB_NAME", "PP_DB_USER", "PP_DB_PASSWORD"):
        assert f"env_var('{name}')" in profile
    assert "backend: mssql-python" in profile
