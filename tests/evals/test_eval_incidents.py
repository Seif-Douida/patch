"""The golden incident set (Phase 6 plan Task 3, spec §11.4): every file loads into a strict model,
and a typo, a backwards window, a mismatched id or an untracked game is refused by file name."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from patchpulse.evals.incidents import INCIDENTS_DIR, load_incidents

SAMPLE = "manor-lords-2025-10-control"


def copy_sample(tmp_path: Path, old: str = "", new: str = "", *, name: str = SAMPLE) -> Path:
    text = (INCIDENTS_DIR / f"{SAMPLE}.yaml").read_text(encoding="utf-8")
    if old:
        assert old in text
        text = text.replace(old, new, 1)
    (tmp_path / f"{name}.yaml").write_text(text, encoding="utf-8")
    return tmp_path


def test_the_golden_set_loads() -> None:
    incidents = load_incidents()

    assert len(incidents) == 32
    assert Counter(i.primary_cause for i in incidents) == {
        "no_change": 12,
        "patch": 8,
        "launch_performance": 5,
        "pricing_sale": 4,
        "off_topic": 2,
        "platform_policy": 1,
    }
    assert [i.id for i in incidents] == sorted(i.id for i in incidents)
    assert all(i.window.start <= i.window.end for i in incidents)


def test_a_typo_in_an_incident_is_refused(tmp_path: Path) -> None:
    copy_sample(tmp_path, "secondary_causes: []", "secondary_causes: []\nsecondery_causes: []")

    with pytest.raises(ValueError, match=SAMPLE):
        load_incidents(tmp_path)


def test_a_window_that_ends_before_it_starts_is_refused(tmp_path: Path) -> None:
    copy_sample(tmp_path, "end: 2025-11-05", "end: 2025-10-01")

    with pytest.raises(ValueError, match="ends before it starts"):
        load_incidents(tmp_path)


def test_an_id_that_differs_from_its_file_name_is_refused(tmp_path: Path) -> None:
    copy_sample(tmp_path, name="manor-lords-renamed")

    with pytest.raises(ValueError, match="manor-lords-renamed"):
        load_incidents(tmp_path)


def test_an_untracked_game_is_refused(tmp_path: Path) -> None:
    copy_sample(tmp_path, "appid: 1363080", "appid: 4")

    with pytest.raises(ValueError, match="not a tracked game"):
        load_incidents(tmp_path)
