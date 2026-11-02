"""Which packages an image's `uv sync` groups install, and running code that can import only
those (Phase 3's jobs image check, shared with Phase 5's API image check)."""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def canonical(name: str) -> str:
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
        canonical(line.split("==", 1)[0])
        for line in exported.splitlines()
        if "==" in line and not line.startswith(("#", " "))
    }
    return {
        module
        for module, distributions in importlib.metadata.packages_distributions().items()
        if module != "patchpulse" and not any(canonical(d) in image for d in distributions)
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


def image_groups(dockerfile: str) -> set[str]:
    """The groups a Dockerfile syncs (both of its `uv sync` commands name the same ones)."""
    text = (ROOT / dockerfile).read_text(encoding="utf-8")
    syncs = re.findall(r"uv sync [^\n]*", text)
    (installed,) = {frozenset(re.findall(r"--group (\w+)", sync)) for sync in syncs}
    return set(installed)
