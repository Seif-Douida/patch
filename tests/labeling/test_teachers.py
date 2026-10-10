"""The teacher candidates in config/models.yaml (plan Task 7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from patchpulse.labeling.teachers import MODELS_FILE, ModelConfigError, load_model_config

REPO = Path(__file__).resolve().parents[2]

VALID = """
candidates:
  gemma-big:
    id: gemma-4-31b-it
    rpm: 30
    tpm: 16000
    rpd: 14400
    json_mode: false
    system_instruction: false
    use_fraction: 0.5
teacher: null
"""


def test_model_config_loads(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(VALID, encoding="utf-8")

    config = load_model_config(path)

    model = config.candidates["gemma-big"]
    assert (model.id, model.rpm, model.tpm, model.rpd) == ("gemma-4-31b-it", 30, 16000, 14400)
    assert model.use_fraction == 0.5
    assert config.teacher is None


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (("rpm: 30", "rpm: 0"), "rpm"),
        (("use_fraction: 0.5", "use_fraction: 1.5"), "use_fraction"),
        (("json_mode: false", "json_mode: false\n    temperature: 1"), "temperature"),
        (("teacher: null", "teacher: gemma-small"), "teacher"),
    ],
)
def test_model_config_rejects_bad_values(
    tmp_path: Path, change: tuple[str, str], problem: str
) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(VALID.replace(*change), encoding="utf-8")

    with pytest.raises(ModelConfigError, match=problem):
        load_model_config(path)


def test_repo_models_config_is_valid() -> None:
    config = load_model_config(REPO / MODELS_FILE)

    assert {"gemma-4-31b", "gemma-4-26b-a4b", "gemini-3.5-flash-lite"} <= set(config.candidates)
    assert all(model.use_fraction == 0.5 for model in config.candidates.values())
