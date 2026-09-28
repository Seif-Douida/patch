"""The static JSON the dashboard is built from (spec §5: the web tier never queries SQL)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import Any

from patchpulse.export.site import (
    ATTRIBUTION,
    SCHEMA_VERSION,
    DayRow,
    GameRow,
    PatchRow,
    shape_site_export,
)

GENERATED = datetime(2026, 9, 29, 3, 30, tzinfo=UTC)
HD2 = GameRow(appid=553850, name="Helldivers 2", genres="Action")
CS2 = GameRow(appid=949230, name="Cities: Skylines II", genres="Simulation")


def day(appid: int, when: date, n: int, positive: int, **extra: Any) -> DayRow:
    share = positive / n if n else None
    return DayRow(
        appid=appid,
        date=when,
        n_reviews=n,
        n_positive=positive,
        pos_share=share,
        pos_lo90=extra.get("lo", share),
        pos_hi90=extra.get("hi", share),
        discount_pct=extra.get("discount"),
        is_sale=extra.get("sale", False),
    )


def patch(appid: int, when: date, title: str, patch_type: str = "patch") -> PatchRow:
    return PatchRow(
        appid=appid,
        published_at=datetime(when.year, when.month, when.day, 9, tzinfo=UTC),
        title=title,
        url=f"https://example.com/{appid}",
        patch_type=patch_type,
        is_treatment_candidate=patch_type in {"patch", "major_update"},
    )


def export(games: list[GameRow], days: list[DayRow], patches: list[PatchRow]) -> dict[str, bytes]:
    return shape_site_export(games, days, patches, data_version=42, generated_at=GENERATED)


def load(files: dict[str, bytes], path: str) -> Any:
    return json.loads(files[path])


D1 = date(2026, 9, 20)


def test_export_is_deterministic() -> None:
    days = [day(553850, D1, 3, 2), day(553850, D1 + timedelta(days=1), 1, 1)]
    patches = [patch(553850, D1, "7.1.1"), patch(553850, D1, "7.1.0")]

    first = export([HD2, CS2], days, patches)
    shuffled = export([CS2, HD2], list(reversed(days)), list(reversed(patches)))

    assert first == shuffled


def test_export_includes_attribution_and_data_version() -> None:
    index = load(export([HD2], [day(553850, D1, 3, 2)], []), "index.json")

    assert index["attribution"] == ATTRIBUTION == "Data from Steam"
    assert index["data_version"] == 42
    assert index["schema_version"] == SCHEMA_VERSION
    assert index["generated_at"] == "2026-09-29T03:30:00Z"


def test_game_file_lists_only_that_games_patches() -> None:
    files = export(
        [HD2, CS2],
        [day(553850, D1, 1, 1), day(949230, D1, 1, 0)],
        [patch(553850, D1, "HD2 patch"), patch(949230, D1, "CS2 patch")],
    )

    assert [p["title"] for p in load(files, "games/553850.json")["patches"]] == ["HD2 patch"]
    assert [p["title"] for p in load(files, "games/949230.json")["patches"]] == ["CS2 patch"]


def test_floats_are_rounded_and_dates_are_iso() -> None:
    files = export([HD2], [day(553850, D1, 3, 2, lo=0.312345678, hi=0.912345678)], [])

    (point,) = load(files, "games/553850.json")["series"]
    assert point == {
        "date": "2026-09-20",
        "n": 3,
        "pos": 2,
        "share": 0.6667,
        "lo": 0.3123,
        "hi": 0.9123,
        "sale": False,
        "discount": None,
    }


def test_latest_summary_covers_the_last_30_days() -> None:
    last = D1 + timedelta(days=40)
    days = [
        day(553850, D1, 100, 0),
        day(553850, last - timedelta(days=5), 3, 3),
        day(553850, last, 1, 0),
    ]

    (game,) = load(export([HD2], days, []), "index.json")["games"]

    assert game["latest"] == {"days": 30, "to": last.isoformat(), "n": 4, "share": 0.75}
    assert game["first_date"] == D1.isoformat()


def test_game_without_data_yet_is_listed_without_a_summary() -> None:
    (game,) = load(export([HD2], [], []), "index.json")["games"]

    assert game["latest"] is None
    assert "games/553850.json" in export([HD2], [], [])


def test_files_end_with_a_newline_and_are_utf8() -> None:
    files = export([CS2], [], [patch(949230, D1, "Patch 1.6.2f1 — Autumn Breeze")])

    assert all(content.endswith(b"\n") for content in files.values())
    assert "—" in files["games/949230.json"].decode("utf-8")
