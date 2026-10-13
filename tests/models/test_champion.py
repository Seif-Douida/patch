"""The champion (plan Task 14, design D7-D8): registered as `aspect-classifier` with `champion`
and `challenger` aliases, pinned in models/champion.yaml, and downloaded only if it matches."""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pytest
from pydantic import ValidationError
from tiny_encoder import write_tiny_encoder
from toy_reviews import toy_reviews

from patchpulse.labeling.taxonomy import TAXONOMY_VERSION
from patchpulse.models.artifact import MODEL_FILE, load_scorer, sha256_file
from patchpulse.models.champion import (
    REGISTERED_MODEL,
    ChampionError,
    ChampionPin,
    download_champion,
    pin_for,
    read_pin,
    register_models,
    write_pin,
)
from patchpulse.models.encoder import OnnxEncoder
from patchpulse.models.s0 import save_s0, train_s0
from patchpulse.models.s1a import save_s1a, train_s1a
from patchpulse.models.text import clean_texts


@pytest.fixture
def models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """An S0 and an S1a under build/models, and a local MLflow store with a registry."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    texts, labels = toy_reviews(200, seed=21)
    s0 = Path("build/models/s0-20261013T000000Z")
    save_s0(train_s0(texts, labels), s0, thresholds=np.full(10, 0.5), version="20261013T000000Z")
    write_tiny_encoder(tmp_path / "encoder")
    encoder = OnnxEncoder(tmp_path / "encoder")
    head = train_s1a(encoder.embed(clean_texts(texts)), labels)
    s1a = Path("build/models/s1a-20261013T000000Z")
    save_s1a(
        head, tmp_path / "encoder", s1a, thresholds=np.full(10, 0.5), version="20261013T000000Z"
    )
    return {"s0": s0, "s1a": s1a}


def test_models_are_registered_with_champion_and_challenger_aliases(
    models: dict[str, Path],
) -> None:
    registered = register_models(models, champion="s0", challenger="s1a")

    client = mlflow.MlflowClient()
    champion = client.get_model_version_by_alias(REGISTERED_MODEL, "champion")
    challenger = client.get_model_version_by_alias(REGISTERED_MODEL, "challenger")
    assert int(champion.version) == registered["s0"].version
    assert int(challenger.version) == registered["s1a"].version
    assert champion.tags["family"] == "s0"


def test_champion_pin_is_validated_and_the_checksum_checked_on_download(
    models: dict[str, Path], tmp_path: Path
) -> None:
    registered = register_models(models, champion="s0", challenger=None)
    pin = pin_for(registered["s0"], models["s0"])
    write_pin(pin, Path("models/champion.yaml"))

    assert read_pin(Path("models/champion.yaml")) == pin
    assert pin.model_json_sha256 == sha256_file(models["s0"] / MODEL_FILE)
    assert pin.taxonomy_version == TAXONOMY_VERSION

    artifact = download_champion(pin, tmp_path / "champion")
    assert artifact.family == "s0"
    assert load_scorer(tmp_path / "champion").predict_proba(["fps drops"]).shape == (1, 10)

    wrong = pin.model_copy(update={"model_json_sha256": "0" * 64})
    with pytest.raises(ChampionError, match=r"model\.json"):
        download_champion(wrong, tmp_path / "elsewhere")
    assert not (tmp_path / "elsewhere").exists()

    for bad in (
        {**pin.model_dump(), "family": "s9"},
        {**pin.model_dump(), "model_json_sha256": "not-a-hash"},
        {**pin.model_dump(), "version": 0},
        {**pin.model_dump(), "extra": "field"},
    ):
        with pytest.raises(ValidationError):
            ChampionPin.model_validate(bad)
