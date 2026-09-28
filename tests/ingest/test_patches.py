"""Rules-first patch classifier (spec §6.1; plan Decision 2: the LLM fallback waits for Phase 2).

The titles are real Steam news items collected on 2026-09-28.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from patchpulse.ingest.models import NewsItem
from patchpulse.ingest.patches import CLASSIFIER_VERSION, PatchType, classify

OFFICIAL = "steam_community_announcements"


def news(title: str, tags: list[str] | None = None, feedname: str = OFFICIAL) -> NewsItem:
    return NewsItem(
        gid="1",
        appid=1,
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
    ["New BETA version is available for testing (0.8.110)", "Small Beta Hotfix 0.8.090"],
)
def test_beta_branch_builds_are_not_public_patches(title: str) -> None:
    result = classify(news(title))

    assert result.patch_type is PatchType.OTHER
    assert not result.is_treatment_candidate
    assert not result.ambiguous


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
    assert CLASSIFIER_VERSION == "rules-v1"
