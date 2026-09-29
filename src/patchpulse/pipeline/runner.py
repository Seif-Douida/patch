"""The nightly run (spec §5, plan Decision 3): ingest → classify → dbt → export → publish.

1. Start an ops.pipeline_run; its id is the data_version.
2. Upsert the games from config/games.yaml.
3. Per game: reviews in the incremental window (the last 3 days to now), then news, classified.
   A game that fails is recorded and skipped; the others carry on.
4. Prices for every game, in batches.
5. Backfill: history older than each game's coverage, newest first, round-robin across games
   until the time budget runs out, so every game's history fills in together. If Steam stops
   answering (a rate limit), backfill stops for the night with a warning; the run still succeeds.
6. dbt build (models and data tests). Only if it succeeds is each healthy game's coverage
   advanced, so a failed night leaves no gap: the next one re-reads from the old watermark.
7. Export the site JSON and publish it; purge raw payloads older than 30 days; finish the run.

Raw data is committed page by page, so what was fetched is kept even when a later stage fails.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import Connection, Engine

from patchpulse.config import Settings, SettingsError
from patchpulse.export.site import build_site_export
from patchpulse.ingest.coverage import Coverage, Windows, advance, plan_windows
from patchpulse.ingest.games import GAMES_FILE, Game, load_games
from patchpulse.ingest.http import SteamError, SteamHttp, SteamUnavailableError
from patchpulse.ingest.load import (
    purge_raw_pages,
    read_coverage,
    record_dq,
    record_page,
    upsert_classifications,
    upsert_games,
    upsert_news,
    upsert_prices,
    upsert_reviews,
    write_coverage,
)
from patchpulse.ingest.news import fetch_news
from patchpulse.ingest.patches import classify
from patchpulse.ingest.prices import PriceFetch, fetch_prices
from patchpulse.ingest.reviews import ReviewPage, StopReason, fetch_review_pages
from patchpulse.pipeline.dbt import DbtResult, run_dbt
from patchpulse.pipeline.runs import finish_run, start_run

log = logging.getLogger("patchpulse.pipeline")

RAW_RETENTION = timedelta(days=30)


class PipelineError(RuntimeError):
    """The run failed; ops.pipeline_run has the details."""


class Publisher(Protocol):
    def publish(self, files: Mapping[str, bytes], *, data_version: int) -> None: ...


class NoPublisher:
    """Local runs: the export is built but not pushed anywhere."""

    def publish(self, files: Mapping[str, bytes], *, data_version: int) -> None:
        log.info("publish skipped (local run): %d files, data_version %d", len(files), data_version)


class DbtRunner(Protocol):
    def __call__(
        self, args: Sequence[str], *, settings: Settings, target_path: Path
    ) -> DbtResult: ...


class Exporter(Protocol):
    def __call__(
        self, connection: Connection, *, data_version: int, generated_at: datetime
    ) -> dict[str, bytes]: ...


@dataclass(frozen=True)
class RunResult:
    run_id: int
    status: str
    stages: dict[str, Any]


class _Run:
    """The state of one run: counters for ops.pipeline_run.stages and per-game failures."""

    def __init__(self, engine: Engine, run_id: int, salt: str) -> None:
        self.engine = engine
        self.run_id = run_id
        self.salt = salt
        self.stages: dict[str, Any] = {"durations_s": {}}
        self.errors: dict[int, str] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.monotonic()
        yield
        self.stages["durations_s"][name] = round(time.monotonic() - started, 1)
        log.info("stage %s done in %.1f s", name, self.stages["durations_s"][name])

    def fail_game(self, appid: int, error: Exception) -> None:
        self.errors[appid] = f"{appid}: {type(error).__name__}: {error}"
        log.warning("game %d failed: %s", appid, error)
        with self.engine.begin() as connection:
            record_dq(
                connection,
                run_id=self.run_id,
                check="game_failed",
                appid=appid,
                severity="error",
                detail=str(error),
            )

    def store_page(self, page: ReviewPage, counters: dict[str, int]) -> None:
        """Commit one page of reviews and its data-quality findings."""
        appid = page.raw.appid
        with self.engine.begin() as connection:
            record_page(connection, page.raw, run_id=self.run_id)
            counters["new_versions"] += upsert_reviews(connection, page.records, run_id=self.run_id)
            for check, detail in (
                ("schema_drift", ", ".join(sorted(page.drift))),
                ("invalid_review", f"{page.invalid} unusable review(s)" if page.invalid else ""),
                (
                    "cursor_repeat",
                    "Steam repeated a cursor" if page.next_stop is StopReason.CURSOR_REPEAT else "",
                ),
            ):
                if detail:
                    record_dq(
                        connection,
                        run_id=self.run_id,
                        check=check,
                        appid=appid,
                        severity="warn",
                        detail=detail,
                    )
        counters["pages"] += 1
        counters["reviews"] += len(page.records)


def run_nightly(
    settings: Settings,
    *,
    engine: Engine,
    steam: SteamHttp,
    now: datetime,
    publisher: Publisher,
    target_root: Path,
    games_file: Path = GAMES_FILE,
    dbt: DbtRunner = run_dbt,
    export: Exporter = build_site_export,
    clock: Callable[[], float] = time.monotonic,
) -> RunResult:
    if settings.author_hash_salt is None:
        raise SettingsError("missing or invalid settings: PP_AUTHOR_HASH_SALT")
    with engine.begin() as connection:
        run_id = start_run(connection, kind="nightly", image_version=os.environ.get("APP_VERSION"))
    run = _Run(engine, run_id, settings.author_hash_salt.get_secret_value())
    log.info("nightly run %d started", run_id)
    try:
        _run_stages(
            run, settings, steam, now, publisher, target_root, games_file, dbt, export, clock
        )
    except Exception as error:
        with engine.begin() as connection:
            finish_run(connection, run_id, status="failed", stages=run.stages, error=str(error))
        if isinstance(error, PipelineError):
            raise
        raise PipelineError(f"run {run_id} failed: {error}") from error

    status = "failed" if run.errors else "succeeded"
    error_text = "; ".join(run.errors.values()) or None
    with engine.begin() as connection:
        finish_run(connection, run_id, status=status, stages=run.stages, error=error_text)
    log.info("nightly run %d %s", run_id, status)
    if run.errors:
        raise PipelineError(f"run {run_id}: {len(run.errors)} game(s) failed: {error_text}")
    return RunResult(run_id, status, run.stages)


def _run_stages(
    run: _Run,
    settings: Settings,
    steam: SteamHttp,
    now: datetime,
    publisher: Publisher,
    target_root: Path,
    games_file: Path,
    dbt: DbtRunner,
    export: Exporter,
    clock: Callable[[], float],
) -> None:
    games = load_games(games_file, today=now.date())
    with run.engine.begin() as connection:
        upsert_games(connection, games, run_id=run.run_id)
        plans = {g.appid: _plan(connection, g, now) for g in games}

    with run.stage("reviews"):
        counters = {"pages": 0, "reviews": 0, "new_versions": 0}
        for game in games:
            windows = plans[game.appid][1]
            start, end = windows.incremental
            try:
                for page in fetch_review_pages(
                    steam, game.appid, start=start, end=end, salt=run.salt
                ):
                    run.store_page(page, counters)
            except SteamError as error:
                run.fail_game(game.appid, error)
        run.stages["reviews"] = counters

    with run.stage("news"):
        items_total = new_total = 0
        for game in games:
            if game.appid in run.errors:
                continue
            try:
                fetched = fetch_news(steam, game.appid)
            except SteamError as error:
                run.fail_game(game.appid, error)
                continue
            with run.engine.begin() as connection:
                record_page(connection, fetched.raw, run_id=run.run_id)
                new_total += upsert_news(connection, fetched.items, run_id=run.run_id)
                upsert_classifications(
                    connection, {i.gid: classify(i) for i in fetched.items}, run_id=run.run_id
                )
            items_total += len(fetched.items)
        run.stages["news"] = {"items": items_total, "new": new_total}

    with run.stage("prices"):
        try:
            prices = fetch_prices(steam, [g.appid for g in games])
        except SteamError as error:
            # A missing day of prices is a gap in one snapshot, not a reason to lose the night.
            run.stages["prices"] = {"ok": 0, "error": f"{type(error).__name__}: {error}"}
            with run.engine.begin() as connection:
                record_dq(
                    connection,
                    run_id=run.run_id,
                    check="prices_unavailable",
                    appid=None,
                    severity="warn",
                    detail=str(error),
                )
        else:
            _store_prices(run, prices, now)

    with run.stage("backfill"):
        reached = _backfill(run, settings, steam, games, plans, clock)

    with run.stage("dbt"):
        result = dbt(["build"], settings=settings, target_path=target_root / f"run-{run.run_id}")
        run.stages["dbt"] = result.counts
        if not result.ok:
            raise PipelineError("dbt build failed: " + " | ".join(result.failures)[:3000])

    with run.engine.begin() as connection:
        for game in games:
            if game.appid in run.errors:
                continue
            previous, windows = plans[game.appid]
            coverage = advance(previous, windows, backfill_reached=reached.get(game.appid))
            write_coverage(connection, game.appid, coverage, run_id=run.run_id)

    with run.stage("export"):
        with run.engine.connect() as connection:
            files = export(connection, data_version=run.run_id, generated_at=now)
        run.stages["export"] = {"files": len(files)}

    with run.stage("publish"):
        publisher.publish(files, data_version=run.run_id)

    with run.engine.begin() as connection:
        run.stages["purged_pages"] = purge_raw_pages(connection, older_than=now - RAW_RETENTION)


def _store_prices(run: _Run, prices: PriceFetch, now: datetime) -> None:
    with run.engine.begin() as connection:
        for raw in prices.raws:
            record_page(connection, raw, run_id=run.run_id)
        upsert_prices(connection, prices.snapshots, snapshot_date=now.date(), run_id=run.run_id)
        for appid in sorted(prices.failed):
            record_dq(
                connection,
                run_id=run.run_id,
                check="price_unavailable",
                appid=appid,
                severity="warn",
                detail="appdetails answered success: false",
            )
    run.stages["prices"] = {
        "ok": len(prices.snapshots) - len(prices.failed),
        "failed": sorted(prices.failed),
    }


def _plan(connection: Connection, game: Game, now: datetime) -> tuple[Coverage | None, Windows]:
    previous = read_coverage(connection, game.appid)
    return previous, plan_windows(previous, backfill_start=game.backfill_start, now=now)


def _backfill(
    run: _Run,
    settings: Settings,
    steam: SteamHttp,
    games: Sequence[Game],
    plans: Mapping[int, tuple[Coverage | None, Windows]],
    clock: Callable[[], float],
) -> dict[int, datetime | None]:
    """Fill history round-robin until the budget runs out; returns how far each game got."""
    deadline = clock() + settings.backfill_budget_minutes * 60
    counters = {"pages": 0, "reviews": 0, "new_versions": 0}
    pagers: dict[int, Iterator[ReviewPage]] = {}
    windows: dict[int, tuple[datetime, datetime]] = {}
    for game in games:
        window = plans[game.appid][1].backfill
        if window is None or game.appid in run.errors:
            continue
        windows[game.appid] = window
        pagers[game.appid] = fetch_review_pages(
            steam,
            game.appid,
            start=window[0],
            end=window[1],
            salt=run.salt,
            deadline=deadline,
            clock=clock,
        )

    oldest: dict[int, datetime] = {}
    reached: dict[int, datetime | None] = {}
    complete: list[int] = []
    stopped: str | None = None
    while pagers:
        for appid in list(pagers):
            try:
                page = next(pagers[appid], None)
            except SteamUnavailableError as error:
                # Steam is refusing us (a rate limit or an outage): stop backfill for every game
                # tonight. It is best effort; each game resumes from its oldest review next night.
                stopped = f"{type(error).__name__}: {error}"
                log.warning("backfill stopped: %s", error)
                with run.engine.begin() as connection:
                    record_dq(
                        connection,
                        run_id=run.run_id,
                        check="backfill_stopped",
                        appid=appid,
                        severity="warn",
                        detail=str(error),
                    )
                for unfinished in pagers:
                    reached[unfinished] = oldest.get(unfinished)
                pagers.clear()
                break
            except SteamError as error:
                run.fail_game(appid, error)
                del pagers[appid]
                continue
            if page is not None:
                run.store_page(page, counters)
                if page.records:
                    earliest = min(r.timestamp_created for r in page.records)
                    oldest[appid] = min(oldest.get(appid, earliest), earliest)
            if page is None or page.next_stop is not None:
                if page is not None and page.next_stop in {
                    StopReason.EMPTY,
                    StopReason.WINDOW_PASSED,
                }:
                    reached[appid] = windows[appid][0]  # the whole window is covered
                    complete.append(appid)
                else:  # deadline or a repeated cursor: resume from the oldest review next night
                    reached[appid] = oldest.get(appid)
                del pagers[appid]
    run.stages["backfill"] = {
        **counters,
        "budget_minutes": settings.backfill_budget_minutes,
        "complete": sorted(complete),
        "stopped": stopped,
    }
    return reached
