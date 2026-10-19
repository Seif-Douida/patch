"""The review encoder in the registry and its pin (plan Task 13): registered as `review-encoder`,
pinned in config/retrieval.yaml, and downloaded only if every checksum matches."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlflow
import pytest
from pydantic import ValidationError
from tiny_encoder import write_tiny_embedder

from patchpulse.models.artifact import sha256_file
from patchpulse.models.text import CLEANING_VERSION
from patchpulse.retrieval.embedders import CANDIDATES, MANIFEST_FILE
from patchpulse.retrieval.pin import (
    REGISTERED_ENCODER,
    EncoderError,
    RetrievalPin,
    download_encoder,
    pin_for,
    read_pin,
    register_encoder,
    write_pin,
)
from patchpulse.retrieval.rules import INDEX_RULES_VERSION


def a_pin(**changes: object) -> RetrievalPin:
    fields: dict[str, object] = {
        "embedder": "e5-small",
        "registry_version": 3,
        "embedder_json_sha256": "a" * 64,
    }
    return RetrievalPin.model_validate(fields | changes)


def test_pin_round_trip_and_refuses_unknown_fields(tmp_path: Path) -> None:
    pin = a_pin()
    path = tmp_path / "config" / "retrieval.yaml"

    write_pin(pin, path)

    assert read_pin(path) == pin
    assert (pin.collection, pin.alias) == ("reviews_v1", "reviews")
    assert (pin.rules_version, pin.cleaning_version) == (INDEX_RULES_VERSION, CLEANING_VERSION)
    assert path.read_text(encoding="utf-8").startswith("# ")  # says what it is
    with pytest.raises(ValidationError):
        a_pin(surprise=1)
    with pytest.raises(ValidationError):
        a_pin(embedder_json_sha256="not a hash")


@pytest.fixture
def encoder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tiny int8-like embedder under build/encoders, and a local MLflow store with a registry."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    model_dir = Path("build/encoders/e5-small-int8")
    write_tiny_embedder(model_dir, replace(CANDIDATES["e5-small"], dims=4, max_tokens=16))
    return model_dir


def test_the_encoder_is_registered_pinned_and_downloaded(encoder: Path, tmp_path: Path) -> None:
    version = register_encoder(encoder)
    pin = pin_for(encoder, version)

    assert mlflow.MlflowClient().get_model_version(REGISTERED_ENCODER, str(version))
    assert pin.embedder == "e5-small"
    assert pin.embedder_json_sha256 == sha256_file(encoder / MANIFEST_FILE)

    embedder = download_encoder(pin, tmp_path / "out" / "encoder")
    assert embedder.embed_queries(["the game stutters"]).shape == (1, 4)


def test_download_refuses_a_checksum_mismatch(encoder: Path, tmp_path: Path) -> None:
    pin = pin_for(encoder, register_encoder(encoder))
    wrong = pin.model_copy(update={"embedder_json_sha256": "0" * 64})
    target = tmp_path / "out" / "encoder"

    with pytest.raises(EncoderError, match=r"embedder\.json doesn't match the pin"):
        download_encoder(wrong, target)
    assert not target.exists()  # nothing written

    other = pin.model_copy(update={"embedder": "minilm-l12"})
    with pytest.raises(EncoderError, match="isn't the pinned encoder"):
        download_encoder(other, target)
    assert not target.exists()
