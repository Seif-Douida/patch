"""The static JSON the dashboard is built from (spec §5: the web tier never queries SQL)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from patchpulse.export.site import (
    ATTRIBUTION,
    MIN_ASPECT_F1,
    SCHEMA_VERSION,
    AspectDayRow,
    AspectModel,
    Aspects,
    DayRow,
    EffectRow,
    Effects,
    GameRow,
    PatchRow,
    read_aspect_model,
    shape_site_export,
)
from patchpulse.labeling.taxonomy import ASPECTS

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


def patch(
    appid: int, when: date, title: str, patch_type: str = "patch", patch_id: str | None = None
) -> PatchRow:
    return PatchRow(
        appid=appid,
        published_at=datetime(when.year, when.month, when.day, 9, tzinfo=UTC),
        title=title,
        url=f"https://example.com/{appid}",
        patch_type=patch_type,
        is_treatment_candidate=patch_type in {"patch", "major_update"},
        patch_id=patch_id or f"{appid}-{when}-{title}",
    )


MODEL = AspectModel(family="s0", version="20261012T223218Z", test_macro_f1=0.61391133)


def export(
    games: list[GameRow],
    days: list[DayRow],
    patches: list[PatchRow],
    aspects: Aspects | None = None,
    effects: Effects | None = None,
) -> dict[str, bytes]:
    return shape_site_export(
        games,
        days,
        patches,
        data_version=42,
        generated_at=GENERATED,
        aspects=aspects,
        effects=effects,
    )


def mentions(**counts: int) -> tuple[int, ...]:
    return tuple(counts.get(code, 0) for code in ASPECTS)


def aspect_day(
    appid: int,
    when: date,
    scored: tuple[int, int],
    neg: tuple[int, ...] = mentions(),
    pos: tuple[int, ...] = mentions(),
) -> AspectDayRow:
    return AspectDayRow(
        appid, when, n_neg_scored=scored[0], n_pos_scored=scored[1], neg=neg, pos=pos
    )


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


def test_export_leaves_out_comparison_games() -> None:
    # Phase 4: comparison games (role donor) are stored as daily counts only, never shown.
    donor = GameRow(appid=1245620, name="Elden Ring", genres="Action, RPG", role="donor")
    both = GameRow(appid=949230, name="Cities: Skylines II", genres="Simulation", role="both")
    days = [day(1245620, D1, 50, 40), day(553850, D1, 10, 8)]
    patches = [patch(1245620, D1, "Patch 1.16")]

    files = export([HD2, donor, both], days, patches)

    assert "games/1245620.json" not in files
    assert {g["appid"] for g in load(files, "index.json")["games"]} == {553850, 949230}


def test_files_end_with_a_newline_and_are_utf8() -> None:
    files = export([CS2], [], [patch(949230, D1, "Patch 1.6.2f1 — Autumn Breeze")])

    assert all(content.endswith(b"\n") for content in files.values())
    assert "—" in files["games/949230.json"].decode("utf-8")


def test_export_v2_shape_is_deterministic() -> None:
    days = [day(553850, D1, 5, 3), day(553850, D1 + timedelta(days=1), 4, 1)]
    aspect_days = [
        aspect_day(553850, D1, (2, 3), neg=mentions(performance=2), pos=mentions(content=1)),
        aspect_day(
            553850, D1 + timedelta(days=1), (3, 1), neg=mentions(performance=1, monetization=2)
        ),
    ]

    first = export([HD2, CS2], days, [], Aspects(MODEL, aspect_days))
    shuffled = export([CS2, HD2], days[::-1], [], Aspects(MODEL, aspect_days[::-1]))

    assert first == shuffled
    game = load(first, "games/553850.json")
    assert game["schema_version"] == SCHEMA_VERSION == 3
    assert game["aspects"]["codes"] == list(ASPECTS)
    assert game["aspects"]["days"][0] == {
        "date": "2026-09-20",
        "scored": [2, 3],
        "neg": [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "pos": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    }
    assert [d["date"] for d in game["aspects"]["days"]] == ["2026-09-20", "2026-09-21"]
    index = load(first, "index.json")
    assert index["aspect_model"] == {
        "family": "s0",
        "version": "20261012T223218Z",
        "test_macro_f1": 0.6139,
        "aspect_f1": {},
    }
    hd2 = next(g for g in index["games"] if g["appid"] == 553850)
    assert hd2["top_complaint_30d"] == {"code": "performance", "share": 0.6, "n": 5}
    cs2 = next(g for g in index["games"] if g["appid"] == 949230)
    assert cs2["top_complaint_30d"] is None
    assert load(first, "games/949230.json")["aspects"] == {"codes": list(ASPECTS), "days": []}


def test_top_complaint_covers_the_last_30_days_and_skips_off_topic() -> None:
    # Off-topic reviews (memes, jokes) aren't complaints about the game.
    last = D1 + timedelta(days=40)
    days = [day(553850, D1, 10, 0), day(553850, last, 10, 0)]
    aspect_days = [
        aspect_day(553850, D1, (10, 0), neg=mentions(monetization=10)),
        aspect_day(553850, last, (10, 0), neg=mentions(off_topic=6, stability_bugs=3)),
    ]

    (game,) = load(export([HD2], days, [], Aspects(MODEL, aspect_days)), "index.json")["games"]

    assert game["top_complaint_30d"] == {"code": "stability_bugs", "share": 0.3, "n": 10}


def test_without_a_model_the_aspect_fields_are_null() -> None:
    files = export([HD2], [day(553850, D1, 3, 2)], [])

    index = load(files, "index.json")
    assert index["schema_version"] == 3
    assert index["aspect_model"] is None
    assert index["games"][0]["top_complaint_30d"] is None
    assert load(files, "games/553850.json")["aspects"] is None


def test_aspect_model_is_read_from_model_json(tmp_path: Path) -> None:
    (tmp_path / "model.json").write_text(
        json.dumps({"family": "s0", "version": "v1", "metrics": {"test_macro_f1": 0.6}}),
        encoding="utf-8",
    )

    assert read_aspect_model(tmp_path) == AspectModel("s0", "v1", 0.6)
    (tmp_path / "model.json").write_text(
        json.dumps(
            {
                "family": "s0",
                "version": "v1",
                "metrics": {"test_macro_f1": 0.6, "test_f1_performance": 0.87, "other": 1.0},
            }
        ),
        encoding="utf-8",
    )
    assert read_aspect_model(tmp_path) == AspectModel("s0", "v1", 0.6, {"performance": 0.87})
    assert read_aspect_model(None) is None
    # An unreadable model.json loses the model block for a night; it never fails the export.
    assert read_aspect_model(tmp_path / "missing") is None


def test_weak_aspects_are_named_and_left_out_of_the_top_complaint() -> None:
    # An aspect the model detects with F1 under MIN_ASPECT_F1 on gold test isn't a headline.
    model = AspectModel("s0", "v1", 0.61, {"ux_controls": 0.48, "performance": 0.87})
    days = [day(553850, D1, 10, 0)]
    aspect_days = [aspect_day(553850, D1, (10, 0), neg=mentions(ux_controls=6, performance=3))]

    files = export([HD2], days, [], Aspects(model, aspect_days))

    index = load(files, "index.json")
    assert index["aspect_model"]["aspect_f1"] == {"performance": 0.87, "ux_controls": 0.48}
    assert MIN_ASPECT_F1 == 0.5
    assert index["games"][0]["top_complaint_30d"] == {"code": "performance", "share": 0.3, "n": 10}


TOO_FEW = "too few comparison games without their own patch (3 of 5 needed)"
NO_TRACK = "the comparison games didn't track the game before the patch"


def effect_row(bundle_id: str, horizon: int, method: str, **values: Any) -> EffectRow:
    return EffectRow(
        bundle_id=bundle_id,
        horizon=horizon,
        effect=values.get("effect"),
        lo90=values.get("lo90"),
        hi90=values.get("hi90"),
        method=method,
        flags=frozenset(values.get("flags", ())),
        reason=values.get("reason"),
    )


def effects(
    *rows: EffectRow, method: str = "synth", report: str | None = None, validated: bool = True
) -> Effects:
    return Effects(
        default_method=method,
        effects_version="e2",
        benchmark_report=report,
        validated=validated,
        rows=list(rows),
    )


def horizon(days: int, ends: str, **values: Any) -> dict[str, Any]:
    return {
        "days": days,
        "ends": ends,
        "effect": values.get("effect"),
        "lo90": values.get("lo90"),
        "hi90": values.get("hi90"),
        "method": values.get("method"),
        "flags": values.get("flags", []),
        "reason": values.get("reason"),
    }


def test_patch_effects_in_schema_3() -> None:
    # The night of 29 Sep: the last complete day is 28 Sep.
    patches = [
        patch(553850, date(2026, 8, 1), "Patch 1.0", patch_id="a"),
        patch(553850, date(2026, 8, 3), "Patch 1.0.1", patch_id="b"),  # joins a's bundle
        patch(553850, date(2026, 8, 3), "Hotfix", "hotfix", patch_id="h"),  # not a candidate
        patch(553850, date(2026, 8, 20), "Patch 1.1", patch_id="c"),  # d starts 10 days later
        patch(553850, date(2026, 8, 30), "Patch 1.2", patch_id="d"),  # nothing stored yet
        patch(553850, date(2026, 9, 25), "Patch 1.3", patch_id="e"),  # 7 days end 1 Oct
    ]
    served = effects(
        effect_row(
            "a",
            7,
            "synth",
            effect=-4.123456,
            lo90=-6.0,
            hi90=-2.2,
            flags={"volume_spike", "confounded"},
        ),
        # Synthetic control had none: interrupted time series, with synthetic control's reason.
        effect_row("a", 14, "its", effect=-3.5, lo90=-5.0, hi90=-2.0, reason=TOO_FEW),
        effect_row("c", 7, "synth", reason=NO_TRACK),
    )

    game = load(export([HD2], [], patches, effects=served), "games/553850.json")

    blocks = {p["title"]: p["effect"] for p in game["patches"]}
    estimated = {
        "start": "2026-08-01",
        "horizons": [
            horizon(
                7,
                "2026-08-07",
                effect=-4.1235,
                lo90=-6.0,
                hi90=-2.2,
                method="synth",
                flags=["confounded", "volume_spike"],
            ),
            horizon(
                14, "2026-08-14", effect=-3.5, lo90=-5.0, hi90=-2.0, method="its", reason=TOO_FEW
            ),
        ],
    }
    assert blocks["Patch 1.0"] == estimated
    assert blocks["Patch 1.0.1"] == estimated  # one bundle, one estimate
    assert blocks["Hotfix"] is None
    assert blocks["Patch 1.1"] == {
        "start": "2026-08-20",
        "horizons": [
            horizon(7, "2026-08-26", method="synth", reason=NO_TRACK),
            horizon(14, "2026-09-02", reason="another_patch_within_14_days"),
        ],
    }
    assert blocks["Patch 1.2"] == {
        "start": "2026-08-30",
        "horizons": [
            horizon(7, "2026-09-05", reason="not_estimated_yet"),
            horizon(14, "2026-09-12", reason="not_estimated_yet"),
        ],
    }
    assert blocks["Patch 1.3"] == {
        "start": "2026-09-25",
        "horizons": [
            horizon(7, "2026-10-01", reason="window_not_complete"),
            horizon(14, "2026-10-08", reason="window_not_complete"),
        ],
    }


def test_index_names_the_default_method() -> None:
    report = "reports/causal/benchmark.md"
    patches = [patch(553850, D1, "Patch 1.0")]

    with_effects = export([HD2], [], patches, effects=effects(method="synth", report=report))
    unvalidated = export(
        [HD2], [], patches, effects=effects(method="its", report=report, validated=False)
    )
    without = export([HD2], [], patches)

    assert load(with_effects, "index.json")["causal"] == {
        "default_method": "synth",
        "effects_version": "e2",
        "benchmark_report": report,
        "validated": True,
    }
    assert load(unvalidated, "index.json")["causal"]["validated"] is False  # ADR-007 rule 3
    # Without the effects input (no stored estimates read), no causal block and no estimates.
    assert load(without, "index.json")["causal"] is None
    assert [p["effect"] for p in load(without, "games/553850.json")["patches"]] == [None]
