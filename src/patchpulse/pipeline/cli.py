"""`patchpulse-pipeline`: the command the jobs image runs (and developers run locally).

    migrate          Apply migrations and provision pp_writer. Connects as the server admin
                     (PP_DB_USER / PP_DB_PASSWORD) and needs PP_WRITER_PASSWORD. Run by pp-migrate.
    nightly          The nightly run, as pp_writer. Run by pp-nightly on its cron.
    create-local-db  Local and CI only: create PP_DB_NAME on the SQL Server container.

Exit codes: 0 success, 1 the run failed (details in ops.pipeline_run), 2 bad configuration.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx2

from patchpulse.config import Settings, SettingsError, get_settings
from patchpulse.db.engine import connect_with_resume_retry, make_engine
from patchpulse.db.local import create_local_database
from patchpulse.db.migrate import upgrade_to_head
from patchpulse.db.provision import provision_writer
from patchpulse.ingest.games import GAMES_FILE
from patchpulse.ingest.http import RateLimiter, SteamHttp
from patchpulse.pipeline.runner import NoPublisher, PipelineError, Publisher, run_nightly

log = logging.getLogger("patchpulse.pipeline")


class LocalDirPublisher:
    """Writes the export to a folder, for previewing the site locally."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def publish(self, files: Mapping[str, bytes], *, data_version: int) -> None:
        for relative, content in files.items():
            path = self.directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        log.info("wrote %d files (data_version %d) to %s", len(files), data_version, self.directory)


def configure_logging() -> None:
    """INFO per stage; the HTTP client's per-request lines are kept out of the logs, which share
    the 0.15 GB/day Log Analytics cap."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Explicit levels: basicConfig does nothing if something configured the root logger first.
    logging.getLogger("patchpulse").setLevel(logging.INFO)
    for noisy in ("httpx2", "httpcore2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _migrate(settings: Settings) -> int:
    if settings.writer_password is None:
        raise SettingsError("missing or invalid settings: PP_WRITER_PASSWORD")
    engine = make_engine(settings)
    connect_with_resume_retry(engine).close()  # wake a paused database first
    upgrade_to_head(engine)
    with engine.begin() as connection:
        provision_writer(connection, password=settings.writer_password.get_secret_value())
    log.info("migrations applied and %s provisioned", "pp_writer")
    return 0


def _nightly(settings: Settings, export_dir: Path | None, games_file: Path) -> int:
    engine = make_engine(settings)
    connect_with_resume_retry(engine).close()  # wake a paused database first
    publisher: Publisher = LocalDirPublisher(export_dir) if export_dir else NoPublisher()
    with httpx2.Client() as client, tempfile.TemporaryDirectory() as target:
        try:
            run_nightly(
                settings,
                engine=engine,
                steam=SteamHttp(client, limiter=RateLimiter()),
                now=datetime.now(UTC).replace(microsecond=0),
                publisher=publisher,
                target_root=Path(target),
                games_file=games_file,
            )
        except PipelineError as error:
            log.error("%s", error)
            return 1
    return 0


def _create_local_db(settings: Settings) -> int:
    server = make_engine(settings.model_copy(update={"db_name": "master"}))
    create_local_database(server, settings.db_name)
    log.info("database %s ready on %s", settings.db_name, settings.db_host)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="patchpulse-pipeline", description="PatchPulse nightly pipeline."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("migrate", help="apply migrations and provision pp_writer (admin)")
    nightly = commands.add_parser("nightly", help="run the nightly pipeline (pp_writer)")
    nightly.add_argument("--export-dir", type=Path, help="also write the site JSON here")
    nightly.add_argument(
        "--games-file", type=Path, default=GAMES_FILE, help=f"default: {GAMES_FILE}"
    )
    commands.add_parser("create-local-db", help="local/CI only: create PP_DB_NAME on the container")
    args = parser.parse_args(argv)

    configure_logging()
    try:
        settings = get_settings()
        if args.command == "migrate":
            return _migrate(settings)
        if args.command == "nightly":
            return _nightly(settings, args.export_dir, args.games_file)
        return _create_local_db(settings)
    except SettingsError as error:
        print(f"patchpulse-pipeline: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
