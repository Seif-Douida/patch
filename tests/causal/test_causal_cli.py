"""`patchpulse-ml causal benchmark`: the benchmark on the laptop's snapshot (Phase 4, Task 11)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from patchpulse.causal import cli as causal_cli
from patchpulse.causal.cli import (
    BenchmarkPaths,
    run_benchmark,
    series_from_reviews,
    series_from_seed,
)
from patchpulse.causal.config import CausalConfig, read_causal_config, write_causal_config

FIRST = date(2026, 3, 2)
DAYS = 150
SECRET = "ZEBRA review text that must never leave data/local"


def toy_reviews(appid: int, per_day: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    rid = appid * 100_000
    for offset in range(DAYS):
        for _ in range(per_day):
            rid += 1
            rows.append(
                {
                    "review_id": rid,
                    "appid": appid,
                    "date": FIRST + timedelta(days=offset),
                    "voted_up": bool(rng.random() < 0.8),
                    "text": SECRET,
                }
            )
    return pd.DataFrame(rows)


def test_series_built_from_the_snapshot_match_the_counts() -> None:
    reviews = pd.DataFrame(
        {
            "appid": [1, 1, 1, 2],
            "date": [FIRST, FIRST, FIRST + timedelta(days=2), FIRST],
            "voted_up": [True, False, True, True],
        }
    )

    series = series_from_reviews(reviews, [1], last_day=FIRST + timedelta(days=3))

    one = series[1]
    assert (one.first_day, list(one.totals), list(one.positives)) == (
        FIRST,
        [2, 0, 1, 0],
        [1, 0, 1, 0],
    )
    assert one.observed.all()  # a tracked game's day without reviews is an observed 0
    assert set(series) == {1}


def test_seed_days_not_fetched_are_unobserved(tmp_path: Path) -> None:
    seed = tmp_path / "seed.csv"
    seed.write_text(
        "appid,day,positive,negative\n7,2026-03-02,10,2\n7,2026-03-04,8,1\n", encoding="utf-8"
    )

    (donor,) = series_from_seed(seed).values()

    assert list(donor.totals) == [12, 0, 9]
    assert list(donor.observed) == [True, False, True]


def test_the_config_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "causal.yaml"
    path.write_text(
        "default_method: its\neffects_version: e1\nreport_sha256: null\n", encoding="utf-8"
    )

    unset = read_causal_config(path)
    write_causal_config(CausalConfig("synth", "e1", "ab" * 32), path)

    assert (unset.default_method, unset.benchmarked) == ("its", False)
    assert read_causal_config(path) == CausalConfig("synth", "e1", "ab" * 32)
    assert read_causal_config(path).benchmarked


def toy_data(tmp_path: Path) -> BenchmarkPaths:
    data = tmp_path / "data"
    data.mkdir()
    reviews = pd.concat([toy_reviews(1, 60, 1), toy_reviews(2, 40, 2)], ignore_index=True)
    reviews.to_parquet(data / "reviews.parquet")
    pd.DataFrame({"appid": [1, 2], "name": ["A", "B"], "genres": ["Action", "Action"]}).to_parquet(
        data / "games.parquet"
    )
    pd.DataFrame(
        {
            "appid": [1],
            "date": [FIRST + timedelta(days=120)],
            "title": ["Patch 1.1"],
            "patch_type": ["patch"],
        }
    ).to_parquet(data / "patches.parquet")
    (data / "pull_state.json").write_text(json.dumps({"max_run_id": 20}), encoding="utf-8")
    rng = np.random.default_rng(5)
    lines = ["appid,day,positive,negative"]
    for appid in range(101, 107):
        for offset in range(DAYS):
            total = int(rng.poisson(60))
            positive = int(rng.binomial(total, 0.75))
            lines.append(f"{appid},{FIRST + timedelta(days=offset)},{positive},{total - positive}")
    seed = tmp_path / "donor_daily_counts.csv"
    seed.write_text("\n".join(lines) + "\n", encoding="utf-8")
    games = tmp_path / "games.yaml"
    entries = [
        f"  - {{appid: {a}, name: G{a}, genres: [Action], role: both, backfill_start: 2026-03-02}}"
        for a in (1, 2)
    ]
    entries += [
        f"  - {{appid: {a}, name: D{a}, genres: [Action], role: donor, backfill_start: 2026-03-02}}"
        for a in range(101, 107)
    ]
    games.write_text("games:\n" + "\n".join(entries) + "\n", encoding="utf-8")
    calendar = tmp_path / "calendar.csv"
    calendar.write_text(
        "sale_name,start_date,end_date,source_url\nSale,2026-04-20,2026-04-22,x\n", encoding="utf-8"
    )
    return BenchmarkPaths(
        data_dir=data,
        seed=seed,
        games=games,
        calendar=calendar,
        report=tmp_path / "reports" / "causal" / "benchmark.md",
        config=tmp_path / "config" / "causal.yaml",
    )


def test_dry_run_writes_only_local_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = toy_data(tmp_path)

    def no_tracking(*args: object, **kwargs: object) -> object:
        raise AssertionError("a dry run must not log to MLflow")

    monkeypatch.setattr(causal_cli, "tracking_run", no_tracking)

    assert run_benchmark(paths, per_cell=2, seed=3, dry_run=True) == 0

    dry = paths.data_dir / "causal-benchmark-dry-run.md"
    assert dry.exists()
    assert not paths.report.exists()
    assert not paths.config.exists()
    report = dry.read_text(encoding="utf-8")
    assert "Default method:" in report


def test_report_has_no_review_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = toy_data(tmp_path)
    monkeypatch.setattr(causal_cli, "tracking_run", None)

    run_benchmark(paths, per_cell=1, seed=4, dry_run=True)

    assert "ZEBRA" not in (paths.data_dir / "causal-benchmark-dry-run.md").read_text("utf-8")
