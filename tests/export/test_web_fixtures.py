"""The dashboard's sample data is exactly what the exporter produces (the site's data contract)."""

from __future__ import annotations

from pathlib import Path

from site_sample import sample_files

FIXTURES = Path(__file__).resolve().parents[2] / "web" / "fixtures"


def test_web_fixtures_match_the_exporter() -> None:
    # If this fails after an exporter change, regenerate:
    #   uv run python tests/support/site_sample.py web/fixtures
    expected = sample_files()
    committed = {
        path.relative_to(FIXTURES).as_posix(): path.read_bytes()
        for path in FIXTURES.rglob("*.json")
    }

    assert committed == expected


def test_sample_data_is_invented_not_steam_data() -> None:
    # The fixtures are committed to the public repo; they must never carry real reviews or titles.
    for name, content in sample_files().items():
        assert b"(sample data)" in content or name.startswith("games/"), name
        assert b"steamcommunity" not in content
        assert b"store.steampowered" not in content
