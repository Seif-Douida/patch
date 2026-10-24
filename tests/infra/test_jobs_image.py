"""The champion and the review encoder baked into the jobs image at deploy (Phase 2 plan Task 16,
design D8; Phase 3 plan Task 13).

deploy.yml downloads the versions `models/champion.yaml` and `config/retrieval.yaml` pin, refusing
either on a checksum mismatch, and builds them into the jobs image; the image pins code and models
together, so a rollback restores all of them. Each download needs only the tracking URI, which
reaches its step through its environment.
"""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
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


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def hidden_modules(installed: set[str]) -> set[str]:
    """The top-level modules a `uv sync --no-dev` of these groups wouldn't install: `uv export`
    lists every distribution they lock, transitive ones included, and each installed module is
    traced to its distributions (an import name isn't a distribution name: `mlflow` comes from
    `mlflow-skinny`, `yaml` from `pyyaml`)."""
    exported = subprocess.run(
        ["uv", "export", "--frozen", "--no-dev", "--no-hashes", "--no-emit-project"]
        + [f"--group={group}" for group in sorted(installed)],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout
    image = {
        _canonical(line.split("==", 1)[0])
        for line in exported.splitlines()
        if "==" in line and not line.startswith(("#", " "))
    }
    return {
        module
        for module, distributions in importlib.metadata.packages_distributions().items()
        if module != "patchpulse" and not any(_canonical(d) in image for d in distributions)
    }


def run_with_only(installed: set[str], code: str) -> subprocess.CompletedProcess[str]:
    """Run `code` in a Python where only what these groups install can be imported: an import hook
    makes every other module look absent, not merely None in sys.modules."""
    hidden = hidden_modules(installed)
    script = (
        "import importlib.abc, sys\n"
        f"HIDDEN = {set(hidden)!r}\n"
        "class Absent(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name.split('.')[0] in HIDDEN:\n"
        "            raise ModuleNotFoundError(f'No module named {name!r}', name=name)\n"
        "sys.meta_path.insert(0, Absent())\n" + code
    )
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )


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
    assert "qdrant_client" in hidden_modules(installed)

    result = run_with_only(
        installed,
        f"from patchpulse.models.cli import main\nsys.exit(main({command!r}))\n",
    )

    assert result.returncode == 0, result.stderr[-1500:]
    assert "--to" in result.stdout


def jobs_image_groups() -> set[str]:
    """The groups Dockerfile.jobs syncs (both of its `uv sync` commands name the same ones)."""
    dockerfile = (ROOT / "Dockerfile.jobs").read_text(encoding="utf-8")
    (installed,) = {
        frozenset(re.findall(r"--group (\w+)", sync))
        for sync in re.findall(r"uv sync [^\n]*", dockerfile)
    }
    return set(installed)


def test_the_index_stage_imports_with_only_the_jobs_image_groups() -> None:
    # 20 Oct: retrieval/sync.py imported pandas and retrieval/pin.py MLflow, neither in the jobs
    # image, so the nightly index stage would have failed on import. Its modules must load with
    # only the groups Dockerfile.jobs syncs.
    installed = jobs_image_groups()
    assert {"pandas", "mlflow", "pyarrow"} <= hidden_modules(installed)

    result = run_with_only(
        installed,
        "import patchpulse.pipeline.cli, patchpulse.pipeline.indexing\n"
        "from patchpulse.pipeline.indexing import run_index\n"
        "print('imported')\n",
    )

    assert result.returncode == 0, result.stderr[-1500:]
    assert result.stdout.strip() == "imported"


def test_the_effects_stage_imports_with_only_the_jobs_image_groups() -> None:
    # Phase 4: the effects stage reads its series through causal/series.py, never the benchmark's
    # causal/cli.py, which imports pandas (not in the jobs image).
    installed = jobs_image_groups()
    assert "causal" in installed

    result = run_with_only(
        installed, "from patchpulse.pipeline.effects import run_effects\nprint('imported')\n"
    )

    assert result.returncode == 0, result.stderr[-1500:]
    assert result.stdout.strip() == "imported"


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
