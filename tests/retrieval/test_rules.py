"""The index rules (design D1) live in dbt's vars and in Python; the two must agree."""

from __future__ import annotations

from pathlib import Path

import yaml

from patchpulse.retrieval import rules

DBT_PROJECT = Path(__file__).resolve().parents[2] / "dbt" / "dbt_project.yml"


def test_rules_constants_match_the_dbt_vars() -> None:
    dbt_vars = yaml.safe_load(DBT_PROJECT.read_text(encoding="utf-8"))["vars"]

    assert dbt_vars["index_min_words"] == rules.MIN_WORDS == 20
    assert tuple(dbt_vars["index_char_languages"]) == rules.CHAR_LANGUAGES
    assert set(rules.CHAR_LANGUAGES) == {"schinese", "tchinese", "japanese", "thai"}
    assert dbt_vars["index_min_chars"] == rules.MIN_CHARS == 40
    assert dbt_vars["index_cap_per_game"] == rules.CAP_PER_GAME == 40_000
    assert dbt_vars["index_patch_window_days"] == rules.PATCH_WINDOW_DAYS == 30
    assert dbt_vars["index_rules_version"] == rules.INDEX_RULES_VERSION == "r1"
