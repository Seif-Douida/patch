"""The comparison games' history, fetched once (Phase 4, design D4).

`.github/workflows/donor-history.yml` runs `donor-history plan`, then `donor-history fetch` for each
comparison game on its own GitHub runner (each runner has its own address, so Steam's per-address
limit applies per game; the spike of 21 Oct 2026 saw no refusal). Each run uploads one CSV;
`donor-history merge` checks every game has every day exactly once and writes the dbt seed
`donor_daily_counts.csv`. Counts only: no review text.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path

from patchpulse.ingest.day_counts import DayCounts, fetch_day_counts
from patchpulse.ingest.games import Game
from patchpulse.ingest.http import SteamError, SteamHttp, SteamUnavailableError

PAUSE_S = 300.0  # Steam's window is about 5 minutes
MAX_PAUSES = 3
MAX_ERRORS_A_DAY = 3
HEADER = ("appid", "day", "positive", "negative")


def fetch_history(
    http: SteamHttp,
    appid: int,
    first: date,
    last: date,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> list[DayCounts]:
    """One request per day from `first` to `last`. When Steam keeps refusing (the client already
    paused once), waits 5 minutes and carries on, at most 3 times; then raises."""
    rows: list[DayCounts] = []
    pauses = errors = 0
    day = first
    while day <= last:
        try:
            rows.append(fetch_day_counts(http, appid, day))
        except SteamUnavailableError:
            if pauses >= MAX_PAUSES:
                raise
            pauses += 1
            sleep(PAUSE_S)
            http.refused = False
            continue
        except SteamError:
            errors += 1
            if errors >= MAX_ERRORS_A_DAY:
                raise
            continue
        errors = 0
        day += timedelta(days=1)
    return rows


def write_history_csv(rows: Sequence[DayCounts], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(HEADER)
        for row in rows:
            writer.writerow((row.appid, row.day.isoformat(), row.positive, row.negative))


def read_history_csv(path: Path) -> list[DayCounts]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            DayCounts(
                int(r["appid"]),
                date.fromisoformat(r["day"]),
                int(r["positive"]),
                int(r["negative"]),
            )
            for r in csv.DictReader(handle)
        ]


def merge_history(directory: Path, games: Sequence[Game], last_day: date) -> list[DayCounts]:
    """Every comparison game's rows, by game and day, once each check passes: one file per game
    (found anywhere under `directory`, as artifact downloads nest them), every day from its
    backfill_start to `last_day` exactly once, no negative counts. Raises naming every problem."""
    problems: list[str] = []
    merged: list[DayCounts] = []
    for game in sorted((g for g in games if g.donor), key=lambda g: g.appid):
        name = f"donor-{game.appid}.csv"
        found = sorted(directory.rglob(name))
        if not found:
            problems.append(f"{game.appid}: no {name}")
            continue
        rows = read_history_csv(found[0])
        by_day: dict[date, list[DayCounts]] = defaultdict(list)
        for row in rows:
            by_day[row.day].append(row)
        day = game.backfill_start
        while day <= last_day:
            seen = by_day.pop(day, [])
            if not seen:
                problems.append(f"{game.appid}: missing {day}")
            elif len(seen) > 1:
                problems.append(f"{game.appid}: duplicate {day}")
            elif seen[0].positive < 0 or seen[0].negative < 0 or seen[0].appid != game.appid:
                problems.append(f"{game.appid}: negative counts on {day}")
            else:
                merged.append(seen[0])
            day += timedelta(days=1)
        problems.extend(f"{game.appid}: {extra} is outside the history" for extra in sorted(by_day))
    if problems:
        raise ValueError("the donor history doesn't check out:\n" + "\n".join(problems))
    return merged


def history_plan(games: Sequence[Game], *, last: date | None, today: date) -> list[str]:
    """GITHUB_OUTPUT lines for the workflow: the last day (default yesterday, UTC) and the matrix
    of comparison games with their first day."""
    last_day = last or today - timedelta(days=1)
    matrix = [
        {"appid": g.appid, "first": g.backfill_start.isoformat()}
        for g in sorted((g for g in games if g.donor), key=lambda g: g.appid)
    ]
    return [f"last_day={last_day.isoformat()}", f"matrix={json.dumps(matrix)}"]
