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
    assert index["schema_version"] == 4
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


def test_sample_shows_each_kind_of_estimate() -> None:
    # Schema 3: the site's patch line has a case for each of these (Phase 4 Task 15).
    files = sample_files()
    assert json.loads(files["index.json"])["causal"]["default_method"] == "synth"
    horizons = [
        h
        for name, content in files.items()
        if name.startswith("games/")
        for p in json.loads(content)["patches"]
        if p["effect"] is not None
        for h in p["effect"]["horizons"]
    ]
    assert any(h["effect"] is not None and not h["flags"] for h in horizons)
    assert any(h["effect"] is not None and "confounded" in h["flags"] for h in horizons)
    assert any(
        h["effect"] is None and (h["reason"] or "").startswith("too few comparison games")
        for h in horizons
    )
    assert any(h["reason"] == "window_not_complete" for h in horizons)


def test_sample_shows_each_kind_of_report() -> None:
    # Schema 4: the report page has a case for each of these (Phase 5, design D11).
    files = sample_files()
    reports = {
        name: json.loads(content)["report"]
        for name, content in files.items()
        if name.startswith("reports/")
    }
    assert json.loads(files["index.json"])["reports"] == len(reports) == 3
    verified = [
        r
        for r in reports.values()
        if not r["basic"] and any(c["claim_type"] == "causal" and not c["reworded"]
                                  for c in r["claims"])
    ]  # fmt: skip
    assert verified
    assert any(any(c["reworded"] for c in r["claims"]) and r["withheld"] for r in reports.values())
    assert any(r["basic"] and r["model_ids"] == {} for r in reports.values())
    linked = {
        p["report"]
        for name, content in files.items()
        if name.startswith("games/")
        for p in json.loads(content)["patches"]
        if p["report"]
    }
    assert {f"reports/{bundle}.json" for bundle in linked} == set(reports)
    # Invented quote spans and review ids stay out, as the real ones do.
    for content in (files[name] for name in reports):
        assert b"progression grind" not in content
        assert b"910006" not in content
