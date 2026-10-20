"""The champion: registered in MLflow on DagsHub and pinned in git (plan Task 14, design D7-D8).

- **`register_models`** uploads each evaluated model directory to its own MLflow run (artifact path
  `model`), registers it as a version of `aspect-classifier`, and points the `champion` alias at
  the chosen family's version and `challenger` at the runner-up.
- **`models/champion.yaml`** (`ChampionPin`) names the champion's registered version and the
  SHA-256 of its `model.json`, which in turn lists every other file's checksum. Changing it is a
  pull request Seif merges (spec §13).
- **`download_champion`** fetches that version and keeps it only if `model.json` matches the pin
  and every file matches `model.json`; a model that fails stays out of the image.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlflow
import yaml
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from patchpulse.models.artifact import (
    MODEL_FILE,
    ArtifactError,
    ModelArtifact,
    read_artifact,
    sha256_file,
)
from patchpulse.models.tracking import log_model_dir, tracking_run, use_tracking

REGISTERED_MODEL = "aspect-classifier"
REGISTRY_EXPERIMENT = "registry"
PIN_FILE = Path("models/champion.yaml")


class ChampionError(RuntimeError):
    """The downloaded champion isn't the pinned one."""


class ChampionPin(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = REGISTERED_MODEL
    version: PositiveInt  # the registry's version number
    family: Literal["s0", "s1a", "s1b"]
    model_version: str  # the artifact's own version (its training time)
    model_json_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    taxonomy_version: str


@dataclass(frozen=True)
class Registered:
    family: str
    version: int  # in the registry
    run_id: str
    artifact: ModelArtifact


def ensure_registered_model(client: mlflow.MlflowClient, name: str) -> None:
    try:
        client.create_registered_model(name)
    except MlflowException as error:
        if error.error_code != "RESOURCE_ALREADY_EXISTS":
            raise


def register_models(
    model_dirs: Mapping[str, Path],
    *,
    champion: str,
    challenger: str | None,
    name: str = REGISTERED_MODEL,
) -> dict[str, Registered]:
    """Register every model (each checked first) and set the aliases; returns them by family."""
    use_tracking()
    client = mlflow.MlflowClient()
    registered: dict[str, Registered] = {}
    for family, model_dir in model_dirs.items():
        artifact = read_artifact(model_dir)
        with tracking_run(
            REGISTRY_EXPERIMENT,
            params={
                "family": family,
                "version": artifact.version,
                "model_json_sha256": sha256_file(model_dir / MODEL_FILE),
            },
            run_name=f"{family}-{artifact.version}",
        ) as run:
            log_model_dir(model_dir)
        ensure_registered_model(client, name)
        # MLflow 3's `register_model("runs:/...")` wants an MLflow-format logged model; ours is a
        # plain directory, so the version points at the run's artifacts directly.
        version = client.create_model_version(
            name,
            source=f"{run.info.artifact_uri}/model",
            run_id=run.info.run_id,
            tags={"family": family, "model_version": artifact.version},
        )
        registered[family] = Registered(family, int(version.version), run.info.run_id, artifact)
    client.set_registered_model_alias(name, "champion", str(registered[champion].version))
    if challenger is not None:
        client.set_registered_model_alias(name, "challenger", str(registered[challenger].version))
    return registered


def pin_for(registered: Registered, model_dir: Path) -> ChampionPin:
    return ChampionPin(
        version=registered.version,
        family=registered.artifact.family,
        model_version=registered.artifact.version,
        model_json_sha256=sha256_file(model_dir / MODEL_FILE),
        taxonomy_version=registered.artifact.taxonomy_version,
    )


def write_pin(pin: ChampionPin, path: Path = PIN_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# The aspect model the nightly job scores with (plan Task 14, spec §13). The deploy\n"
        "# downloads this registered version and refuses it unless model.json matches the hash.\n"
    )
    body = yaml.safe_dump(pin.model_dump(), sort_keys=False)
    path.write_text(header + body, encoding="utf-8", newline="\n")


def read_pin(path: Path = PIN_FILE) -> ChampionPin:
    return ChampionPin.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def download_champion(pin: ChampionPin, to: Path) -> ModelArtifact:
    """The pinned version in `to`, checked against the pin and its own checksums."""
    use_tracking()
    to.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="champion-", dir=to.parent) as scratch:
        local = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{pin.name}/{pin.version}", dst_path=scratch
            )
        )
        record = local / MODEL_FILE
        if not record.is_file() or sha256_file(record) != pin.model_json_sha256:
            raise ChampionError(
                f"{pin.name} version {pin.version}: model.json doesn't match the pin's SHA-256"
            )
        try:
            artifact = read_artifact(local)
        except ArtifactError as error:
            raise ChampionError(f"{pin.name} version {pin.version}: {error}") from error
        if (artifact.family, artifact.version) != (pin.family, pin.model_version):
            raise ChampionError(f"{pin.name} version {pin.version} isn't the pinned model")
        shutil.rmtree(to, ignore_errors=True)
        shutil.copytree(local, to)
    return artifact
