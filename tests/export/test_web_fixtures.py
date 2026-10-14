"""The dashboard's sample data is exactly what the exporter produces (the site's data contract)."""

from __future__ import annotations

import json
from pathlib import Path

from site_sample import sample_files

from patchpulse.labeling.taxonomy import ASPECTS

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
    # Schema 2: the sample carries aspect counts and the model that made them.
    index = json.loads(expected["index.json"])
    assert index["schema_version"] == 2
    assert index["aspect_model"] is not None
    assert set(index["aspect_model"]["aspect_f1"]) == set(ASPECTS)
    for game in index["games"]:
        assert game["top_complaint_30d"] is not None
        aspects = json.loads(expected[f"games/{game['appid']}.json"])["aspects"]
        assert aspects["codes"] == list(ASPECTS)
        assert aspects["days"]


def test_sample_data_is_invented_not_steam_data() -> None:
    # The fixtures are committed to the public repo; they must never carry real reviews or titles.
    for name, content in sample_files().items():
        assert b"(sample data)" in content or name.startswith("games/"), name
        assert b"steamcommunity" not in content
        assert b"store.steampowered" not in content
