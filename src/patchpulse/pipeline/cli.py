"""`patchpulse-pipeline`: the command the jobs image runs (and developers run locally).

    migrate          Apply migrations and provision pp_writer. Connects as the server admin
                     (PP_DB_USER / PP_DB_PASSWORD) and needs PP_WRITER_PASSWORD. Run by pp-migrate.
    nightly          The nightly run, as pp_writer. Run by pp-nightly on its cron with --publish,
                     which sends the site data and the run's outcome to GitHub (ADR-018).
    create-local-db  Local and CI only: create PP_DB_NAME on the SQL Server container.

Exit codes: 0 success, 1 the run failed (details in ops.pipeline_run), 2 bad configuration.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import httpx2

from patchpulse.config import Settings, SettingsError, get_settings
from patchpulse.db.engine import connect_with_resume_retry, make_engine
from patchpulse.db.local import create_local_database
from patchpulse.db.migrate import upgrade_to_head
from patchpulse.db.provision import provision_writer
from patchpulse.ingest.games import GAMES_FILE, load_games
from patchpulse.ingest.http import RateLimiter, SteamHttp
from patchpulse.pipeline.donor_history import (
    fetch_history,
    history_plan,
    merge_history,
    write_history_csv,
)
from patchpulse.pipeline.donors import check_candidates, summary_lines
from patchpulse.pipeline.runner import NoPublisher, PipelineError, Publisher, run_nightly
from patchpulse.publish.github import GitHubPublisher, usable_private_key

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

    def publish_branch(self, branch: str, files: Mapping[str, bytes], *, data_version: int) -> None:
        """The agent's data goes to a sub-folder named after the branch."""
        LocalDirPublisher(self.directory / branch).publish(files, data_version=data_version)

    def report(self, *, run_id: int, status: str, error: str | None) -> None:
        log.info("run %d %s (not reported: local run)", run_id, status)

    def alert(self, *, title: str, body: str) -> None:
        log.warning("alert (not raised: local run): %s: %s", title, body)

    def resolve(self, *, title: str, comment: str) -> None:
        log.info("alert resolved (local run): %s", title)


def configure_logging() -> None:
    """INFO per stage; the HTTP client's per-request lines are kept out of the logs, which share
    the 0.15 GB/day Log Analytics cap."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Explicit levels: basicConfig does nothing if something configured the root logger first.
    logging.getLogger("patchpulse").setLevel(logging.INFO)
    # httpx (qdrant-client's transport) logs each request's URL, and the Qdrant cluster's URL is a
    # secret.
    for noisy in ("httpx2", "httpcore2", "httpx", "httpcore"):
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


def _github_publisher(settings: Settings, http: httpx2.Client) -> GitHubPublisher:
    repo = settings.github_repository
    app_id = settings.github_app_id
    key = settings.github_app_private_key
    if not (repo and app_id and key):
        required = {
            "PP_GITHUB_REPOSITORY": repo,
            "PP_GITHUB_APP_ID": app_id,
            "PP_GITHUB_APP_PRIVATE_KEY": key,
        }
        missing = ", ".join(name for name, value in required.items() if not value)
        raise SettingsError(f"missing or invalid settings: {missing}")
    if not usable_private_key(key.get_secret_value()):
        raise SettingsError(
            "PP_GITHUB_APP_PRIVATE_KEY is not an RSA private key in PEM format. Paste the whole "
            ".pem file, including its -----BEGIN and -----END lines"
        )
    return GitHubPublisher(app_id=app_id, private_key=key, repo=repo, http=http)


def _nightly(settings: Settings, export_dir: Path | None, games_file: Path, publish: bool) -> int:
    github = httpx2.Client()
    publisher: Publisher
    if publish:  # checked before the database is touched, so a bad setting fails fast
        publisher = _github_publisher(settings, github)
    else:
        publisher = LocalDirPublisher(export_dir) if export_dir else NoPublisher()
    engine = make_engine(settings)
    connect_with_resume_retry(engine).close()  # wake a paused database first
    with github, httpx2.Client() as client, tempfile.TemporaryDirectory() as target:
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


def _donor_candidates(appids: Sequence[int], games_file: Path) -> int:
    """Phase 4: check comparison-game candidates against Steam; prints, stores nothing."""
    tracked = [game for game in load_games(games_file) if game.tracked]
    with httpx2.Client() as client:
        checks, qualifying = check_candidates(
            SteamHttp(client, limiter=RateLimiter()),
            list(appids),
            tracked,
            today=datetime.now(UTC).date(),
        )
    for line in summary_lines(checks, qualifying):
        print(line)
    return 0


def _donor_history(args: argparse.Namespace) -> int:
    """Phase 4: the comparison games' history (Steam only; no database)."""
    if args.step == "plan":
        last = date.fromisoformat(args.last) if args.last else None
        today = datetime.now(UTC).date()
        for line in history_plan(load_games(args.games_file), last=last, today=today):
            print(line)
        return 0
    if args.step == "fetch":
        with httpx2.Client() as client:
            steam = SteamHttp(client, limiter=RateLimiter())
            rows = fetch_history(
                steam, args.appid, date.fromisoformat(args.first), date.fromisoformat(args.last)
            )
        write_history_csv(rows, args.out)
        log.info("game %d: %d days written to %s", args.appid, len(rows), args.out)
        return 0
    try:
        rows = merge_history(
            args.directory, load_games(args.games_file), date.fromisoformat(args.last)
        )
    except ValueError as error:
        print(f"patchpulse-pipeline: {error}", file=sys.stderr)
        return 1
    write_history_csv(rows, args.out)
    log.info(
        "%d rows for %d games written to %s", len(rows), len({r.appid for r in rows}), args.out
    )
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
    nightly.add_argument(
        "--publish",
        action="store_true",
        help="publish to the site-data branch and report the outcome on GitHub (Azure)",
    )
    commands.add_parser("create-local-db", help="local/CI only: create PP_DB_NAME on the container")
    candidates = commands.add_parser(
        "donor-candidates", help="Phase 4: check comparison-game candidates on Steam (no database)"
    )
    candidates.add_argument("appids", nargs="+", type=int)
    candidates.add_argument(
        "--games-file", type=Path, default=GAMES_FILE, help=f"default: {GAMES_FILE}"
    )
    history = commands.add_parser(
        "donor-history", help="Phase 4: the comparison games' daily counts, fetched once"
    ).add_subparsers(dest="step", required=True)
    plan = history.add_parser("plan", help="GITHUB_OUTPUT lines: the last day and the matrix")
    plan.add_argument("--games-file", type=Path, default=GAMES_FILE)
    plan.add_argument("--last", default="", help="YYYY-MM-DD; default yesterday (UTC)")
    fetch = history.add_parser("fetch", help="one game's days into a CSV (a runner per game)")
    fetch.add_argument("--appid", type=int, required=True)
    fetch.add_argument("--first", required=True)
    fetch.add_argument("--last", required=True)
    fetch.add_argument("--out", type=Path, required=True)
    merge = history.add_parser("merge", help="check the downloaded CSVs and write the dbt seed")
    merge.add_argument("directory", type=Path)
    merge.add_argument("--games-file", type=Path, default=GAMES_FILE)
    merge.add_argument("--last", required=True)
    merge.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    configure_logging()
    if args.command == "donor-candidates":  # Steam only: no database settings needed
        return _donor_candidates(args.appids, args.games_file)
    if args.command == "donor-history":
        return _donor_history(args)
    try:
        settings = get_settings()
        if args.command == "migrate":
            return _migrate(settings)
        if args.command == "nightly":
            return _nightly(settings, args.export_dir, args.games_file, args.publish)
        return _create_local_db(settings)
    except SettingsError as error:
        print(f"patchpulse-pipeline: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
