"""`patchpulse-ml causal benchmark` (Phase 4, Task 11): the benchmark on Seif's machine.

Tracked games' daily series come from the pulled snapshot (each review's date and thumbs; never
its text), comparison games' from the dbt seed, sale days from the sale calendar seed, patch
dates from the snapshot and genre groups from config/games.yaml. The real run writes
reports/causal/benchmark.md, logs the MLflow experiment `causal` and writes config/causal.yaml; a
dry run (20 trials a cell by default) writes only data/local/causal-benchmark-dry-run.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from patchpulse.causal.benchmark import (
    choose_default,
    placebo_in_space,
    quiet_windows,
    render_benchmark_report,
    run_trials,
    summarise,
)
from patchpulse.causal.config import CAUSAL_CONFIG_PATH, CausalConfig, write_causal_config
from patchpulse.causal.estimate import EFFECTS_VERSION
from patchpulse.causal.series import DailySeries, from_counts
from patchpulse.config import LabSettings
from patchpulse.ingest.games import GAMES_FILE, load_games
from patchpulse.models.tracking import tracking_run

log = logging.getLogger("patchpulse.causal")

SEED_PATH = Path("dbt/seeds/donor_daily_counts.csv")
CALENDAR_PATH = Path("dbt/seeds/steam_sale_calendar.csv")
REPORT_PATH = Path("reports/causal/benchmark.md")
DRY_RUN_REPORT = "causal-benchmark-dry-run.md"
DRY_RUN_PER_CELL = 20


@dataclass(frozen=True)
class BenchmarkPaths:
    data_dir: Path
    seed: Path = SEED_PATH
    games: Path = GAMES_FILE
    calendar: Path = CALENDAR_PATH
    report: Path = REPORT_PATH
    config: Path = CAUSAL_CONFIG_PATH


def series_from_reviews(
    reviews: pd.DataFrame, appids: Iterable[int], *, last_day: date
) -> dict[int, DailySeries]:
    """Each game's positive and total reviews per day from its first review to `last_day`; a day
    without reviews is an observed 0. Reads only appid, date and voted_up."""
    wanted = set(appids)
    frame = reviews.loc[reviews["appid"].isin(wanted), ["appid", "date", "voted_up"]]
    grouped = frame.groupby(["appid", "date"])["voted_up"].agg(["sum", "count"]).reset_index()
    counts: dict[int, dict[date, tuple[int, int]]] = defaultdict(dict)
    for appid, day, positive, total in zip(
        grouped["appid"], grouped["date"], grouped["sum"], grouped["count"], strict=True
    ):
        counts[int(appid)][pd.Timestamp(day).date()] = (int(positive), int(total))
    return {
        appid: from_counts(appid, min(days), last_day, days, missing_is_zero=True)
        for appid, days in counts.items()
    }


def series_from_seed(path: Path) -> dict[int, DailySeries]:
    """Comparison games' daily counts; a day the seed lacks is unobserved."""
    counts: dict[int, dict[date, tuple[int, int]]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            positive, negative = int(row["positive"]), int(row["negative"])
            counts[int(row["appid"])][date.fromisoformat(row["day"])] = (
                positive,
                positive + negative,
            )
    return {
        appid: from_counts(appid, min(days), max(days), days, missing_is_zero=False)
        for appid, days in counts.items()
    }


def sale_days(path: Path) -> frozenset[date]:
    days: set[date] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            day, end = date.fromisoformat(row["start_date"]), date.fromisoformat(row["end_date"])
            while day <= end:
                days.add(day)
                day += timedelta(days=1)
    return frozenset(days)


def run_benchmark(paths: BenchmarkPaths, *, per_cell: int, seed: int, dry_run: bool) -> int:
    began = time.monotonic()
    games = load_games(paths.games)
    groups = {g.appid: list(g.genres) for g in games}
    tracked = [g.appid for g in games if g.tracked]
    reviews = pd.read_parquet(
        paths.data_dir / "reviews.parquet", columns=["appid", "date", "voted_up"]
    )
    last_day = pd.Timestamp(reviews["date"].max()).date() - timedelta(
        days=1
    )  # the pull's day is partial
    series = series_from_reviews(reviews, tracked, last_day=last_day)
    donors = (
        {a: s for a, s in series_from_seed(paths.seed).items() if a in groups}
        if paths.seed.exists()
        else {}
    )
    patches = pd.read_parquet(paths.data_dir / "patches.parquet")
    patch_days: dict[int, list[date]] = defaultdict(list)
    for appid, day in zip(patches["appid"], patches["date"], strict=True):
        patch_days[int(appid)].append(pd.Timestamp(day).date())
    sales = sale_days(paths.calendar)
    windows = quiet_windows(series, patch_days, sales)
    if not windows:
        log.error("no quiet windows in the snapshot")
        return 1
    trials = run_trials(
        windows, series, donors, groups, patch_days, sale_days=sales, per_cell=per_cell, seed=seed
    )
    tracked_dates = sorted({d for a in tracked for d in patch_days.get(a, [])})
    placebos = placebo_in_space(
        donors, tracked_dates, groups, patch_days, seed=seed, sale_days=sales
    )
    choice = choose_default(trials)
    state = json.loads((paths.data_dir / "pull_state.json").read_text(encoding="utf-8"))
    meta: dict[str, Any] = {
        "trials per cell": per_cell,
        "seed": seed,
        "snapshot (max run id)": state.get("max_run_id"),
        "effects version": EFFECTS_VERSION,
        "tracked games": len(series),
        "comparison games": len(donors),
        "quiet windows": len(windows),
        "dry run": dry_run,
    }
    report = render_benchmark_report(summarise(trials), choice, trials, placebos, meta)
    target = paths.data_dir / DRY_RUN_REPORT if dry_run else paths.report
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
    log.info(
        "benchmark: %d trials, %d placebos, default %s (%s) in %.0f s; report %s",
        len(trials),
        len(placebos),
        choice.method,
        choice.reason,
        time.monotonic() - began,
        target,
    )
    if dry_run:
        return 0
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    with tracking_run("causal", params={**meta, "default_method": choice.method}):
        import mlflow

        mlflow.log_metrics(
            {
                f"{method}_{name}": value
                for method, row in choice.table.items()
                for name, value in row.items()
                if not np.isnan(value)
            }
        )
        mlflow.log_artifact(str(target))
    write_causal_config(
        CausalConfig(choice.method, EFFECTS_VERSION, digest, validated=choice.validated),
        paths.config,
    )
    return 0


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    causal = commands.add_parser("causal", help="Phase 4: the causal benchmark").add_subparsers(
        dest="causal_command", required=True
    )
    benchmark = causal.add_parser(
        "benchmark", help="injected effects on quiet windows; the default method (ADR-007)"
    )
    benchmark.add_argument("--per-cell", type=int, default=200)
    benchmark.add_argument("--seed", type=int, default=20261021)
    benchmark.add_argument(
        "--dry-run",
        action="store_true",
        help=f"{DRY_RUN_PER_CELL} a cell unless --per-cell; local report only",
    )


def run(args: argparse.Namespace, settings: LabSettings) -> int:
    per_cell = args.per_cell
    if args.dry_run and per_cell == 200:
        per_cell = DRY_RUN_PER_CELL
    return run_benchmark(
        BenchmarkPaths(data_dir=settings.data_dir),
        per_cell=per_cell,
        seed=args.seed,
        dry_run=args.dry_run,
    )
