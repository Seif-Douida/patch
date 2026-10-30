"""Patches close together are estimated as one bundle (Phase 4 design D2)."""

from __future__ import annotations

from datetime import date, timedelta

from patchpulse.causal.bundles import Bundle, Patch, bundle_patches, window_complete

D0 = date(2026, 1, 1)


def patch(patch_id: str, offset: int, appid: int = 1) -> Patch:
    return Patch(patch_id=patch_id, appid=appid, day=D0 + timedelta(days=offset))


def shape(patches: list[Patch]) -> list[tuple[tuple[str, ...], bool]]:
    return [(b.patch_ids, b.allow_14) for b in bundle_patches(patches)]


def test_bundles_cover_every_spacing() -> None:
    # "Within 7 days" is inside the 7-day window after the bundle's first patch: at most 6 apart.
    assert shape([patch("a", 0), patch("b", 0)]) == [(("a", "b"), True)]
    assert shape([patch("a", 0), patch("b", 1)]) == [(("a", "b"), True)]
    assert shape([patch("a", 0), patch("b", 6)]) == [(("a", "b"), True)]
    assert shape([patch("a", 0), patch("b", 7)]) == [(("a",), False), (("b",), True)]
    assert shape([patch("a", 0), patch("b", 8)]) == [(("a",), False), (("b",), True)]
    # A patch every 3 days joins only while within 7 days of its bundle's first patch.
    chain = [patch(str(n), n) for n in (0, 3, 6, 9, 12)]
    assert shape(chain) == [(("0", "3", "6"), False), (("9", "12"), True)]
    # The next bundle inside the 14 days drops the 14-day estimate; from day 14 on it doesn't.
    assert shape([patch("a", 0), patch("b", 10)])[0] == (("a",), False)
    assert shape([patch("a", 0), patch("b", 14)])[0] == (("a",), True)
    assert shape([patch("a", 0), patch("b", 15)])[0] == (("a",), True)


def test_bundles_are_per_game_and_ordered() -> None:
    bundles = bundle_patches([patch("x", 3, appid=2), patch("b", 2), patch("a", 0)])

    assert [(b.appid, b.bundle_id, b.start) for b in bundles] == [
        (1, "a", D0),
        (2, "x", D0 + timedelta(days=3)),
    ]
    assert bundles[0].patch_ids == ("a", "b")


def test_window_complete_counts_the_patch_day() -> None:
    bundle = Bundle(appid=1, start=D0, patch_ids=("a",), allow_14=True)

    assert window_complete(bundle, 7, D0 + timedelta(days=6))
    assert not window_complete(bundle, 7, D0 + timedelta(days=5))
    assert window_complete(bundle, 14, D0 + timedelta(days=13))
    assert not window_complete(bundle, 14, D0 + timedelta(days=12))
