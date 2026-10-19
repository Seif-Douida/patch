"""The champion and the review encoder baked into the jobs image at deploy (Phase 2 plan Task 16,
design D8; Phase 3 plan Task 13).

deploy.yml downloads the versions `models/champion.yaml` and `config/retrieval.yaml` pin, refusing
either on a checksum mismatch, and builds them into the jobs image; the image pins code and models
together, so a rollback restores all of them. Each download needs only the tracking URI, which
reaches its step through its environment.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD = "Download the champion model"
ENCODER = "Download the review encoder"


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


@pytest.mark.parametrize(
    ("step", "command"),
    [
        (DOWNLOAD, ["champion", "download", "--help"]),
        (ENCODER, ["retrieval", "encoder", "download", "--help"]),
    ],
)
def test_downloads_run_with_only_the_steps_groups(step: str, command: list[str]) -> None:
    # 16 Oct: `patchpulse-ml` loaded the retrieval commands at startup, which import qdrant_client
    # (the `retrieval` group), so the download step, which installs fewer groups, failed. The
    # command must start with only what the step installs: other groups' packages are hidden.
    download = build_steps()[index_of(build_steps(), name=step)]
    installed = set(re.findall(r"--group (\w+)", download["run"]))
    groups = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "dependency-groups"
    ]

    def packages(group: str) -> set[str]:
        names: set[str] = set()
        for entry in groups[group]:
            if isinstance(entry, dict):
                names |= packages(entry["include-group"])
            else:
                names.add(re.split(r"[<>=~\[ ;]", entry, maxsplit=1)[0].replace("-", "_"))
        return names

    available = set().union(*(packages(group) for group in installed))
    hidden = sorted(set().union(*(packages(g) for g in groups if g != "dev")) - available)
    assert "qdrant_client" in hidden
    # An import hook, so a hidden package looks absent (not merely None in sys.modules).
    script = (
        "import importlib.abc, sys\n"
        f"HIDDEN = {set(hidden)!r}\n"
        "class Absent(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name.split('.')[0] in HIDDEN:\n"
        "            raise ModuleNotFoundError(f'No module named {name!r}', name=name)\n"
        "sys.meta_path.insert(0, Absent())\n"
        "from patchpulse.models.cli import main\n"
        f"sys.exit(main({command!r}))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr[-1500:]
    assert "--to" in result.stdout


def test_deploy_downloads_the_pinned_encoder() -> None:
    workflow = deploy()
    assert "config/**" in workflow[True]["push"]["paths"]  # config/retrieval.yaml, the pin
    steps = build_steps()
    download = steps[index_of(steps, name=ENCODER)]
    assert download["if"] == "matrix.image == 'patchpulse-jobs'"
    assert "patchpulse-ml retrieval encoder download --to build/encoder" in download["run"]
    # Token-less, like the champion: DagsHub serves the public repo's registered models.
    assert download["env"] == {"MLFLOW_TRACKING_URI": "${{ vars.MLFLOW_TRACKING_URI }}"}
    assert "${{" not in download["run"]
    assert (
        index_of(steps, uses="astral-sh/setup-uv")
        < index_of(steps, name=ENCODER)
        < index_of(steps, uses="docker/build-push-action")
    )


def test_jobs_image_syncs_the_retrieval_group_and_checks_the_encoder() -> None:
    dockerfile = (ROOT / "Dockerfile.jobs").read_text(encoding="utf-8")
    syncs = re.findall(r"uv sync [^\n]*", dockerfile)
    assert all("--group retrieval" in sync for sync in syncs), syncs
    runtime = dockerfile.split("\nFROM ", 2)[2]  # the final stage
    assert re.search(r"^COPY build/encoder /app/encoder$", runtime, re.M)
    assert re.search(r"PP_ENCODER_DIR=/app/encoder\b", runtime)
    # The image refuses to build with an encoder its own packages can't load and run.
    assert runtime.index("COPY build/encoder ") < runtime.index("embed_queries")
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert ignored.index("build/*") < ignored.index("!build/encoder")
