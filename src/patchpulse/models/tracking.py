"""MLflow tracking for Phase 2 (plan Task 9, design D7): DagsHub's hosted MLflow, free tier.

`tracking_run` won't start without `MLFLOW_TRACKING_URI`, from the environment or `.env`, so a
missing setting can't quietly log to a local folder instead. Only reports (`reports/`) may be
logged as artifacts:
- never anything under `data/`, where review text and the annotators' notes live (spec C4);
- model directories only through `log_model_dir`, from `build/models/`, for the registry
  (`champion.register_models`; ADR-021 as amended on 13 Oct: S0 hashes its n-grams, so no model
  file holds review words). Training runs record each model file's SHA-256 instead.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import mlflow
from mlflow import ActiveRun

from patchpulse.config import get_lab_settings

ALLOWED_ROOTS = (Path("reports"),)
_NEVER = Path("data")


class TrackingError(RuntimeError):
    """MLflow isn't configured, or an artifact may not leave this machine."""


def require_tracking() -> str:
    """The tracking URI. `.env`'s MLflow settings are exported if the environment lacks them."""
    # MLflow's "View run" line starts with an emoji that a redirected Windows console (cp1252)
    # can't encode; it's printed before the run is closed, so the crash left runs open. Our
    # commands print the run id themselves.
    os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        settings = get_lab_settings()
        if not settings.mlflow_tracking_uri:
            raise TrackingError(
                "MLflow isn't configured: set MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME and "
                "MLFLOW_TRACKING_PASSWORD in .env (DagsHub: the repo's Remote → Experiments)"
            )
        os.environ["MLFLOW_TRACKING_URI"] = settings.mlflow_tracking_uri
        if settings.mlflow_tracking_username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow_tracking_username
        if settings.mlflow_tracking_password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = (
                settings.mlflow_tracking_password.get_secret_value()
            )
    return os.environ["MLFLOW_TRACKING_URI"]


def use_tracking() -> str:
    """Point MLflow at the configured store before any client is made; returns the URI.

    MLflow's own global URI wins over the environment, so a process that switches stores (tests,
    or a later command) must set it before creating a client.
    """
    uri = require_tracking()
    mlflow.set_tracking_uri(uri)
    return uri


@contextmanager
def tracking_run(
    experiment: str,
    *,
    params: Mapping[str, object],
    tags: Mapping[str, str] | None = None,
    run_name: str | None = None,
) -> Iterator[ActiveRun]:
    use_tracking()
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags=dict(tags or {})) as run:
        if params:
            mlflow.log_params({name: str(value) for name, value in params.items()})
        yield run


def log_ids_hash(name: str, ids: Iterable[int]) -> None:
    """Which reviews a run used, without the reviews: a SHA-256 of the sorted unique ids."""
    unique = sorted({int(i) for i in ids})
    digest = hashlib.sha256(",".join(str(i) for i in unique).encode("ascii")).hexdigest()
    mlflow.log_params({f"{name}_ids_sha256": digest, f"{name}_n": len(unique)})


def _may_leave(path: Path) -> Path:
    resolved = path.resolve()
    here = Path.cwd().resolve()
    allowed = any(resolved.is_relative_to((here / root).resolve()) for root in ALLOWED_ROOTS)
    if not allowed or resolved.is_relative_to((here / _NEVER).resolve()):
        raise TrackingError(
            f"refused to log {path}: only files under reports/ may leave this machine"
        )
    return resolved


def log_report(path: Path) -> None:
    mlflow.log_artifact(str(_may_leave(path)))


MODELS_ROOT = Path("build/models")


def log_model_dir(path: Path) -> None:
    """A model directory under build/models, logged whole under `model/` (for the registry)."""
    resolved = path.resolve()
    if not resolved.is_relative_to((Path.cwd() / MODELS_ROOT).resolve()) or not resolved.is_dir():
        raise TrackingError(f"refused to log {path}: only model directories under build/models/")
    mlflow.log_artifacts(str(resolved), artifact_path="model")
