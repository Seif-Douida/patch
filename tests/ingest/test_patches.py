"""Rules-first patch classifier (spec §6.1; plan Decision 2: the LLM fallback waits for Phase 2).

The titles are real Steam news items collected on 2026-09-28, and for rules-v2 on 2026-11-08
(every official post of the tracked games since their data start, and the comparison games'
since 1 Oct 2024). A synthetic title says so.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from patchpulse.ingest.models import NewsItem
from patchpulse.ingest.patches import CLASSIFIER_VERSION, PatchType, classify

OFFICIAL = "steam_community_announcements"
NO_MANS_SKY, HELLDIVERS_2, DEEP_ROCK_GALACTIC = 275850, 553850, 548430


def news(
    title: str, tags: list[str] | None = None, feedname: str = OFFICIAL, appid: int = 1
) -> NewsItem:
    return NewsItem(
        gid="1",
        appid=appid,
        title=title,
        url="https://example.com",
        contents="",
        published_at=datetime(2026, 9, 28, tzinfo=UTC),
        feedname=feedname,
        feedlabel="Community Announcements",
        tags=tags or [],
    )


@pytest.mark.parametrize(
    ("title", "tags", "expected"),
    [
        ("Devoid of Liberty: 7.1.1", ["patchnotes"], PatchType.PATCH),
        ("Devoid of Liberty: 7.1.0", [], PatchType.PATCH),
        ("Patch 1.6.2f1 - Autumn Breeze", [], PatchType.PATCH),
        ("Spring Cleaning - Patch 1.5.7f1", [], PatchType.PATCH),
        ("Devoid of Liberty: 7.0.0", [], PatchType.MAJOR_UPDATE),
        (
            "HELLDIVERS 2 Devoid of Liberty Update Out Now",
            ["vo_marketing_message"],
            PatchType.MAJOR_UPDATE,
        ),
        (
            "See what's new in The Witcher 3: Wild Hunt — Remastered!",
            ["patchnotes"],
            PatchType.MAJOR_UPDATE,
        ),
        (
            "Major Update #8: Wide Valley, Cows, Cider, Advanced Construction & More",
            ["workshop"],
            PatchType.MAJOR_UPDATE,
        ),
        (
            "A complete overhaul to space arrives in the form of No Man's Sky COSMOS (7.0)",
            [],
            PatchType.MAJOR_UPDATE,
        ),
        ("Hotfix - Patch 1.5.10f1", ["patchnotes"], PatchType.HOTFIX),
        ("The Blood of Dawnwalker - Hotfix 1.0.5", ["patchnotes"], PatchType.HOTFIX),
        ("Save 35% on Manor Lords", ["hide_store"], PatchType.MARKETING),
        (
            "Manor Lords News Roundup at the Hooded Horse Summer Sale",
            ["hide_store"],
            PatchType.MARKETING,
        ),
        ("Cities: Skylines II Merch is here!", [], PatchType.MARKETING),
        (
            "Vote for The Blood of Dawnwalker at the 2026 Golden Joystick Awards!",
            [],
            PatchType.MARKETING,
        ),
        ("The Blood of Dawnwalker - 1 million copies sold", [], PatchType.MARKETING),
        ('"Our Journey Continues" 10th Anniversary Expedition', [], PatchType.EVENT),
        ("Celebrating 10 Years (!) of No Man's Sky", [], PatchType.EVENT),
        # rules-v2: a version reads as a build (3+ parts, a build suffix, or marked as a version)
        ("Machinery of Oppression 6.0.3", ["patchnotes"], PatchType.PATCH),
        ("01.002.105", ["patchnotes"], PatchType.PATCH),
        ("1.5.4f1 - First Frost", [], PatchType.PATCH),
        ("Update Summary (Ver.1.030)", [], PatchType.PATCH),
        ("v1.0 Launch Update - Patch Notes", [], PatchType.MAJOR_UPDATE),
        # Capcom's word for a release: "notice" is not an announcement word
        ("Notice of Update Distribution", ["patchnotes"], PatchType.PATCH),
    ],
)
def test_official_posts_are_classified_by_rules(
    title: str, tags: list[str], expected: PatchType
) -> None:
    result = classify(news(title, tags))

    assert result.patch_type is expected
    assert not result.ambiguous


@pytest.mark.parametrize(
    "title",
    [
        "New BETA version is available for testing (0.8.110)",
        "Small Beta Hotfix 0.8.090",
        "Community Update #31 - Patch 8 Stress Test - Register Now!",
        "Community Update #33 - Patch 8 Stress Test Now Live",
        "House Flipper Remastered Playtests are LIVE!",
        "Friday Facts #444 - 2.1 Experimental release",
    ],
)
def test_test_builds_are_not_public_patches(title: str) -> None:
    result = classify(news(title))

    assert result.patch_type is PatchType.OTHER
    assert result.rule == "test_build"
    assert not result.is_treatment_candidate
    assert not result.ambiguous


@pytest.mark.parametrize(
    "title",
    [
        "Announcing The Witcher 3: Wild Hunt \u2013 Remastered",
        "Hades II v1.0 Coming September 25!",
        "Post-Launch Patch 1 Preview & Notes",  # an opt-in beta; the patch shipped a week later
        "The Showcase for the Free Title Update 3 (Ver. 1.030) is here!",
        "Update Ver. 1.021 \u2014 Maintenance Notice",
        "Introducing Update Ver. 1.041 \u2014 Available Wednesday, February 18",
        "Optimizing Liberty - Incoming Patch: 27th May",
        "Patch Delay Update",
        "Bike Patch - Announcement Post",
        "Dev Diary: Decorations Patch",
        "Detailer\u2019s Patch #2 - Developer Diary",
        "An update regarding the Econonomy 2.0 Patch;",  # Steam's spelling
        "Free Patch and Updated Plans",
        "REDstreams \u2013 Update 2.2 is coming!",
    ],
)
def test_announcements_are_not_releases(title: str) -> None:
    result = classify(news(title))

    assert result.patch_type is PatchType.OTHER
    assert result.rule == "announcement"
    assert not result.ambiguous


def test_feature_names_are_not_versions() -> None:
    diary = classify(news("Economy 2.0 - Part 1"))  # a development diary
    challenge = classify(news("Take Your Shot in Photo Mode Challenge 3.0"))

    assert (diary.patch_type, diary.rule, diary.ambiguous) == (PatchType.OTHER, "no_rule", True)
    assert challenge.patch_type is PatchType.OTHER


@pytest.mark.parametrize(
    ("title", "expected"),
    [  # synthetic: sales milestones, multipliers and counts are not builds
        ("1.2M copies sold - thank you!", PatchType.MARKETING),
        ("1.5 million copies sold", PatchType.MARKETING),
        ("2.5M players!", PatchType.OTHER),
        ("3.0 Million Players!", PatchType.OTHER),
        ("Double XP weekend: 1.5x rewards", PatchType.OTHER),
        ("4.5k Overwhelmingly Positive reviews", PatchType.OTHER),
    ],
)
def test_milestones_are_not_builds(title: str, expected: PatchType) -> None:
    assert classify(news(title)).patch_type is expected


@pytest.mark.parametrize(
    "title",  # synthetic: Steam titles often set the dash before a version as an en or em dash
    ["The Witcher 3 \u2013 4.04 now live", "Hotline \u2014 2.1 notes", "Hotline - 2.1 notes"],
)
def test_any_dash_marks_a_version(title: str) -> None:
    assert classify(news(title)).patch_type is PatchType.PATCH


def test_studio_release_titles_are_per_game() -> None:
    relics = "Introducing...No Man's Sky: RELICS"
    nms = classify(news(relics, appid=NO_MANS_SKY))
    expedition = classify(news("Introducing..The Cursed Expedition!", appid=NO_MANS_SKY))
    elsewhere = classify(news(relics))
    unjust = classify(news("Into the Unjust - Out Now!", appid=HELLDIVERS_2))
    # Synthetic: a paid Warbond in Arrowhead's release words is not an update.
    warbond = classify(news("Truth Enforcers Warbond - Out Now!", appid=HELLDIVERS_2))
    season = classify(news("Season 06: 'Relics of Hoxxes' is NOW LIVE!", appid=DEEP_ROCK_GALACTIC))
    trailer = classify(news("The Season 06 Narrated Trailer is here!", appid=DEEP_ROCK_GALACTIC))
    other_game = classify(news("Resident Evil Requiem out now!"))

    assert (nms.patch_type, nms.rule) == (PatchType.MAJOR_UPDATE, "studio_release_title")
    assert expedition.patch_type is PatchType.EVENT
    assert (elsewhere.patch_type, elsewhere.ambiguous) == (PatchType.OTHER, True)
    assert (unjust.patch_type, unjust.rule) == (PatchType.MAJOR_UPDATE, "studio_release_title")
    assert not warbond.is_treatment_candidate
    assert (season.patch_type, season.rule) == (PatchType.MAJOR_UPDATE, "studio_release_title")
    assert trailer.patch_type is PatchType.MARKETING
    assert other_game.patch_type is PatchType.OTHER


def test_rule_order_where_words_collide() -> None:
    beta_hotfix = classify(news("Small Beta Hotfix 0.8.090"))
    cross_play = classify(news("Community Update #34 - Connecting With Cross-Play & Hotfix #32"))
    hotfix_out = classify(news("Hotfix 5.00c is out now!", ["patchnotes"]))

    assert (beta_hotfix.patch_type, beta_hotfix.rule) == (PatchType.OTHER, "test_build")
    assert cross_play.patch_type is PatchType.HOTFIX
    assert hotfix_out.patch_type is PatchType.HOTFIX


@pytest.mark.parametrize(
    ("title", "feedname"),
    [
        (
            "The Blood of Dawnwalker получила хотфикс 1.0.5 - "
            "разработчики устранили часть статтеров",
            "PlayGround.ru - официальная группа",
        ),
        ("CD Projekt lays out why you should play The Witcher 3 Remastered", "PC Gamer"),
    ],
)
def test_press_articles_are_never_patches(title: str, feedname: str) -> None:
    result = classify(news(title, feedname=feedname))

    assert result.patch_type is PatchType.OTHER
    assert result.rule == "not_official"
    assert not result.ambiguous


def test_hotfix_beats_patch() -> None:
    assert classify(news("Hotfix - Patch 1.5.10f1", ["patchnotes"])).patch_type is PatchType.HOTFIX


def test_patchnotes_tag_alone_is_a_patch() -> None:
    assert classify(news("Tuesday notes", ["patchnotes"])).patch_type is PatchType.PATCH


@pytest.mark.parametrize(
    "title", ["City Corner #12 - Collected Questions", "Introducing..No Man's Sky: The Swarm"]
)
def test_unmatched_items_are_other_and_ambiguous(title: str) -> None:
    result = classify(news(title))

    assert result.patch_type is PatchType.OTHER
    assert result.ambiguous
    assert result.rule == "no_rule"


def test_only_patches_and_major_updates_are_treatment_candidates() -> None:
    candidates = {
        patch_type
        for title, tags in [
            ("Devoid of Liberty: 7.1.1", ["patchnotes"]),
            ("Major Update #8", []),
            ("Hotfix 1.0.5", []),
            ("Save 35% on Manor Lords", []),
            ("10th Anniversary Expedition", []),
            ("City Corner #12", []),
        ]
        for patch_type in [classify(news(title, tags)).patch_type]
        if classify(news(title, tags)).is_treatment_candidate
    }

    assert candidates == {PatchType.PATCH, PatchType.MAJOR_UPDATE}


def test_classifier_version_is_recorded() -> None:
    assert CLASSIFIER_VERSION == "rules-v2"


def test_dbt_reads_the_classifier_version_python_writes() -> None:
    # core.fact_patch inner-joins the labels of dbt's var; another version would empty it.
    project = Path(__file__).parents[2] / "dbt" / "dbt_project.yml"
    config = yaml.safe_load(project.read_text(encoding="utf-8"))

    assert config["vars"]["classifier_version"] == CLASSIFIER_VERSION
