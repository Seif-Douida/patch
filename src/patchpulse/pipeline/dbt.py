"""Run dbt as a subprocess and summarise its run_results.json.

dbt is never imported: it runs as the `dbt` executable next to this interpreter, with connection
settings passed as PP_ environment variables (dbt/profiles.yml reads them). Target files and logs
go to a caller-chosen directory, so the project folder stays read-only.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from patchpulse.config import Settings

# Relative to the working directory: the repo root locally, /app in the jobs image.
DBT_PROJECT_DIR = Path("dbt")
_FAILED = {"error", "fail"}


@dataclass(frozen=True)
class DbtResult:
    returncode: int
    counts: dict[str, int]  # node status -> count (success, pass, warn, error, fail, skipped)
    failures: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def dbt_executable() -> str:
    found = shutil.which("dbt", path=str(Path(sys.executable).parent)) or shutil.which("dbt")
    if found is None:
        raise FileNotFoundError("dbt executable not found; install the `pipeline` dependency group")
    return found


def dbt_env(settings: Settings) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PP_DB_HOST": settings.db_host,
            "PP_DB_PORT": str(settings.db_port),
            "PP_DB_NAME": settings.db_name,
            "PP_DB_USER": settings.db_user,
            "PP_DB_PASSWORD": settings.db_password.get_secret_value(),
            # dbt's as_bool filter only accepts Python literals: "True" / "False".
            "PP_DB_ENCRYPT": str(settings.db_encrypt),
            "PP_DB_TRUST_CERT": str(settings.db_trust_cert),
            "DO_NOT_TRACK": "1",
            "DBT_SEND_ANONYMOUS_USAGE_STATS": "false",
        }
    )
    return env


def _paths(project_dir: Path, target_path: Path) -> list[str]:
    return [
        "--project-dir",
        str(project_dir),
        "--profiles-dir",
        str(project_dir),
        "--target-path",
        str(target_path),
        "--log-path",
        str(target_path / "logs"),
    ]


def run_dbt(
    args: Sequence[str],
    *,
    settings: Settings,
    target_path: Path,
    project_dir: Path = DBT_PROJECT_DIR,
) -> DbtResult:
    """Run `dbt <args>` and summarise the nodes it ran."""
    started = time.monotonic()
    completed = subprocess.run(  # noqa: S603 - fixed executable, arguments from code
        [dbt_executable(), *args, *_paths(project_dir, target_path)],
        env=dbt_env(settings),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    counts: Counter[str] = Counter()
    failures: list[str] = []
    results_file = target_path / "run_results.json"
    if results_file.exists():
        for node in json.loads(results_file.read_text(encoding="utf-8")).get("results", []):
            counts[node["status"]] += 1
            if node["status"] in _FAILED:
                failures.append(f"{node['unique_id']}: {(node.get('message') or '')[:300]}")
    if completed.returncode != 0 and not failures:
        failures.append((completed.stdout + completed.stderr)[-2000:])
    return DbtResult(completed.returncode, dict(counts), failures, time.monotonic() - started)


def compile_inline(
    sql: str, *, settings: Settings, target_path: Path, project_dir: Path = DBT_PROJECT_DIR
) -> str:
    """Compile a Jinja SQL snippet against the project (macros included) and return plain SQL."""
    completed = subprocess.run(  # noqa: S603 - fixed executable, arguments from code
        [
            dbt_executable(),
            "--quiet",
            "compile",
            "--inline",
            sql,
            *_paths(project_dir, target_path),
        ],
        env=dbt_env(settings),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"dbt compile failed: {(completed.stdout + completed.stderr)[-2000:]}")
    return completed.stdout.strip()
