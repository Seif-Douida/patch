"""The comparison-game candidate check (Phase 4, design D3): Steam's facts against the rules,
and how many candidates qualify for each of the tracked games' patches."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from fake_steam import FakeSteam

from patchpulse.causal.rules import GenreGroup
from patchpulse.ingest.games import Game, Role
from patchpulse.pipeline.donors import check_candidates

TODAY = date(2026, 10, 21)
TRACKED = Game(
    appid=553850,
    name="Helldivers 2",
    genres=["Action"],
    role=Role.BOTH,
    backfill_start=date(2024, 10, 1),
)
PASSES, QUIET, LATE, SHORT_NEWS, SIM = 1001, 1002, 1003, 1004, 1005


def noon(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 12, tzinfo=UTC)


def details(name: str, genres: list[str], released: str) -> dict[str, object]:
    return {
        "name": name,
        "genres": [{"id": str(i), "description": g} for i, g in enumerate(genres)],
        "release_date": {"coming_soon": False, "date": released},
    }


def steam_world() -> FakeSteam:
    steam = FakeSteam()
    steam.add_news(TRACKED.appid, "t1", "Patch 1.2", noon(date(2025, 3, 1)))
    steam.add_news(TRACKED.appid, "t2", "Patch 1.3", noon(date(2025, 6, 1)))
    rid = 0
    for appid, per_day in ((PASSES, 25), (QUIET, 5), (LATE, 30), (SHORT_NEWS, 30), (SIM, 30)):
        for offset in range(1, 91):
            for _ in range(per_day):
                rid += 1
                steam.add_review(appid, rid, noon(TODAY - timedelta(days=offset)))
    steam.details[PASSES] = details("Passes", ["Action", "RPG"], "Mar 4, 2020")
    steam.details[QUIET] = details("Quiet", ["Action"], "Mar 4, 2020")
    steam.details[LATE] = details("Late", ["Action"], "Jan 15, 2025")
    steam.details[SHORT_NEWS] = details("Short news", ["Action"], "Mar 4, 2020")
    steam.details[SIM] = details("Sim", ["Simulation", "Strategy"], "Mar 4, 2020")
    steam.add_news(PASSES, "p1", "Patch 2.1", noon(date(2025, 3, 10)))  # inside t1's span
    steam.add_news(PASSES, "p0", "Patch 2.0", noon(date(2023, 5, 1)))
    for n in range(100):  # a busy feed: its 100 latest items start in 2025
        steam.add_news(SHORT_NEWS, f"s{n}", f"Patch 3.{n}", noon(date(2025, 7, 1) + timedelta(n)))
    return steam


def test_check_applies_each_rule() -> None:
    checks, _ = check_candidates(
        steam_world().http(), [PASSES, QUIET, LATE, SHORT_NEWS, SIM], [TRACKED], today=TODAY
    )
    by_id = {c.appid: c for c in checks}

    assert by_id[PASSES].verdict == "pass"
    assert by_id[PASSES].mean_per_day_90d == 25.0
    assert by_id[PASSES].group is GenreGroup.ACTION_RPG
    assert by_id[PASSES].treatment_patches == 1  # 2.1; 2.0 predates the history
    assert by_id[QUIET].reasons == ("under 20 reviews a day (5.0)",)
    assert by_id[LATE].reasons == ("released after 2024-09-02 (2025-01-15)",)
    assert by_id[SHORT_NEWS].reasons == ("its 100 latest news items start after 2024-10-01",)
    assert by_id[SIM].verdict == "pass"
    assert by_id[SIM].group is GenreGroup.SIM_STRATEGY


def test_unknown_candidate_fails_with_a_reason() -> None:
    checks, _ = check_candidates(steam_world().http(), [999], [TRACKED], today=TODAY)

    assert [(c.appid, c.verdict, c.reasons) for c in checks] == [
        (999, "fail", ("not found on Steam",))
    ]


def test_qualifying_counts_per_bundle() -> None:
    _, qualifying = check_candidates(
        steam_world().http(), [PASSES, QUIET, SIM], [TRACKED], today=TODAY
    )

    # t1: the only passing Action candidate patched inside its span; t2: it qualifies. The
    # Simulation candidate is another group; the quiet one failed the check.
    assert qualifying == {"t1": 0, "t2": 1}
