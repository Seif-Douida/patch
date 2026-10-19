"""The review encoder: registered in MLflow on DagsHub and pinned in git (plan Task 13).

- **`register_encoder`** uploads the chosen embedder's int8 build (`build/encoders/<name>-int8`,
  checked against its `embedder.json` first) to its own MLflow run and registers it as a version
  of `review-encoder`. It's a public model quantized here: no review text is in it.
- **`config/retrieval.yaml`** (`RetrievalPin`) names that version, the SHA-256 of its
  `embedder.json` (which lists every other file's checksum), the collection and alias the index
  lives under, and the index rules and text cleaning the vectors were made with. The nightly job
  and the API search with exactly this encoder: one encoder per index.
- **`download_encoder`** fetches the pinned version and keeps it only if `embedder.json` matches the
  pin and every file matches `embedder.json`; anything else is refused and nothing is written.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import mlflow
import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from patchpulse.models.artifact import sha256_file
from patchpulse.models.champion import REGISTRY_EXPERIMENT, ensure_registered_model
from patchpulse.models.text import CLEANING_VERSION
from patchpulse.models.tracking import log_model_dir, tracking_run, use_tracking
from patchpulse.retrieval.embedders import (
    MANIFEST_FILE,
    EmbedderIntegrityError,
    OnnxEmbedder,
    read_manifest,
)
from patchpulse.retrieval.rules import INDEX_RULES_VERSION

REGISTERED_ENCODER = "review-encoder"
PIN_PATH = Path("config/retrieval.yaml")
COLLECTION = "reviews_v1"
ALIAS = "reviews"


class EncoderError(RuntimeError):
    """The downloaded encoder isn't the pinned one."""


class RetrievalPin(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    collection: str = COLLECTION  # the index's versioned collection
    alias: str = ALIAS  # what the job and the API search through
    embedder: str  # the candidate's name, as in retrieval.embedders.CANDIDATES
    registry_version: PositiveInt  # the version of `review-encoder`
    embedder_json_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rules_version: str = INDEX_RULES_VERSION
    cleaning_version: str = CLEANING_VERSION


def write_pin(pin: RetrievalPin, path: Path = PIN_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# The review index's encoder (plan Task 13). The deploy downloads this registered\n"
        "# version and refuses it unless embedder.json matches the hash; the nightly job and the\n"
        "# API embed with it, so changing it means rebuilding the index.\n"
    )
    body = yaml.safe_dump(pin.model_dump(), sort_keys=False)
    path.write_text(header + body, encoding="utf-8", newline="\n")


def read_pin(path: Path = PIN_PATH) -> RetrievalPin:
    return RetrievalPin.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def register_encoder(model_dir: Path, *, name: str = REGISTERED_ENCODER) -> int:
    """Upload a checked embedder build and register it; returns the registry version."""
    manifest = read_manifest(model_dir)  # every file must match embedder.json
    spec = manifest["spec"]
    use_tracking()
    client = mlflow.MlflowClient()
    with tracking_run(
        REGISTRY_EXPERIMENT,
        params={
            "embedder": spec["name"],
            "model_id": spec["model_id"],
            "revision": spec["revision"],
            "quantization": manifest["quantization"],
            "embedder_json_sha256": sha256_file(model_dir / MANIFEST_FILE),
        },
        run_name=f"encoder-{spec['name']}",
    ) as run:
        log_model_dir(model_dir)
    ensure_registered_model(client, name)
    version = client.create_model_version(
        name,
        source=f"{run.info.artifact_uri}/model",
        run_id=run.info.run_id,
        tags={"embedder": spec["name"], "revision": spec["revision"]},
    )
    return int(version.version)


def pin_for(model_dir: Path, registry_version: int) -> RetrievalPin:
    manifest = read_manifest(model_dir)
    return RetrievalPin(
        embedder=manifest["spec"]["name"],
        registry_version=registry_version,
        embedder_json_sha256=sha256_file(model_dir / MANIFEST_FILE),
        cleaning_version=manifest["cleaning_version"],
    )


def download_encoder(pin: RetrievalPin, to: Path) -> OnnxEmbedder:
    """The pinned version in `to`, checked against the pin and its own checksums."""
    use_tracking()
    label = f"{REGISTERED_ENCODER} version {pin.registry_version}"
    to.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="encoder-", dir=to.parent) as scratch:
        local = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{REGISTERED_ENCODER}/{pin.registry_version}",
                dst_path=scratch,
            )
        )
        record = local / MANIFEST_FILE
        if not record.is_file() or sha256_file(record) != pin.embedder_json_sha256:
            raise EncoderError(f"{label}: embedder.json doesn't match the pin's SHA-256")
        try:
            manifest = read_manifest(local)
        except EmbedderIntegrityError as error:
            raise EncoderError(f"{label}: {error}") from error
        built = (manifest["spec"]["name"], manifest["cleaning_version"])
        if built != (pin.embedder, pin.cleaning_version):
            raise EncoderError(f"{label} isn't the pinned encoder")
        shutil.rmtree(to, ignore_errors=True)
        shutil.copytree(local, to)
    return OnnxEmbedder(to)
