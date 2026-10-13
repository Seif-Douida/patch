"""The champion baked into the jobs image at deploy (plan Task 16, design D8).

deploy.yml downloads the version `models/champion.yaml` pins, refusing it on a checksum mismatch,
and builds it into the jobs image; the image pins code and model together, so a rollback restores
both. The download needs only the tracking URI, which reaches that one step through its
environment.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD = "Download the champion model"


def deploy() -> dict[Any, Any]:
    workflow: dict[Any, Any] = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
    )
    return workflow


def build_steps() -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = deploy()["jobs"]["build"]["steps"]
    return steps


def index_of(steps: list[dict[str, Any]], *, name: str = "", uses: str = "") -> int:
    (index,) = [
        i
        for i, step in enumerate(steps)
        if (name and step.get("name") == name) or (uses and step.get("uses", "").startswith(uses))
    ]
    return index


def test_deploy_downloads_the_champion_before_building_the_jobs_image() -> None:
    workflow = deploy()
    assert workflow["jobs"]["build"]["environment"] == "production"
    # PyYAML reads the `on:` key as True.
    assert "models/champion.yaml" in workflow[True]["push"]["paths"]
    steps = build_steps()
    download = steps[index_of(steps, name=DOWNLOAD)]
    assert download["if"] == "matrix.image == 'patchpulse-jobs'"
    assert "patchpulse-ml champion download --to build/champion" in download["run"]
    assert (
        index_of(steps, uses="astral-sh/setup-uv")
        < index_of(steps, name=DOWNLOAD)
        < index_of(steps, uses="docker/build-push-action")
    )


def test_champion_credentials_reach_the_step_only_through_env() -> None:
    # DagsHub serves a public repo's registered models without a token (checked 2026-10-13), so
    # no DagsHub token reaches a job that installs a hundred packages from PyPI.
    steps = build_steps()
    download = steps[index_of(steps, name=DOWNLOAD)]
    assert download["env"] == {"MLFLOW_TRACKING_URI": "${{ vars.MLFLOW_TRACKING_URI }}"}
    assert "${{" not in download["run"]
    text = (ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
    assert "MLFLOW_TRACKING_PASSWORD" not in text
    assert "MLFLOW_TRACKING_USERNAME" not in text


def test_jobs_image_copies_the_model_and_sets_pp_model_dir() -> None:
    dockerfile = (ROOT / "Dockerfile.jobs").read_text(encoding="utf-8")
    syncs = re.findall(r"uv sync [^\n]*", dockerfile)
    assert len(syncs) == 2
    assert all("--group pipeline --group scoring" in sync for sync in syncs), syncs
    runtime = dockerfile.split("\nFROM ", 2)[2]  # the final stage
    assert re.search(r"^COPY build/champion /app/model$", runtime, re.M)
    assert re.search(r"PP_MODEL_DIR=/app/model\b", runtime)
    # The image refuses to build with a model its own packages can't load.
    assert runtime.index("COPY build/champion") < runtime.index("load_scorer")
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    # The rest of build/ (checkpoints, encoders, every trained model) stays out of the context.
    assert ignored.index("build/*") < ignored.index("!build/champion")
