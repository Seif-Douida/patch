"""Freezing the eval datasets before the first graded run (Phase 6 design D8), as Phase 3 froze
its queries: the SHA-256 of every incident, the injection set and the judge's rubric, with the
scoring rule's version. A graded run refuses changed files; changing one afterwards is a ledgered
ruling, a new freeze and a re-run."""

from __future__ import annotations

import hashlib
from pathlib import Path

from patchpulse.evals.incidents import INCIDENTS_DIR
from patchpulse.evals.scoring import SCORING_VERSION

FROZEN_FILE = Path("evals") / "frozen.sha256"
INJECTION_FILE = Path("evals") / "injection" / "cases.yaml"
RUBRIC_FILE = Path("evals") / "judge" / "rubric.md"
_SCORING = "scoring_version"


def frozen_paths() -> list[Path]:
    return [*sorted(INCIDENTS_DIR.glob("*.yaml")), INJECTION_FILE, RUBRIC_FILE]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current() -> dict[str, str]:
    paths = frozen_paths()
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise ValueError(f"can't freeze: missing {', '.join(missing)}")
    entries = {p.as_posix(): _digest(p) for p in paths}
    entries[_SCORING] = SCORING_VERSION
    return entries


def freeze() -> str:
    """Writes FROZEN_FILE (one `<digest>  <path>` line per file, then the scoring version) and
    returns its own SHA-256."""
    lines = [f"{digest}  {name}" for name, digest in _current().items()]
    text = "\n".join(lines) + "\n"
    FROZEN_FILE.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_frozen() -> None:
    """Raises naming every file that changed, appeared or disappeared since the freeze."""
    if not FROZEN_FILE.is_file():
        raise ValueError(f"the eval datasets aren't frozen: run `patchpulse-evals freeze` first "
                         f"({FROZEN_FILE} is missing)")  # fmt: skip
    frozen: dict[str, str] = {}
    for line in FROZEN_FILE.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        frozen[name] = digest
    current = _current()
    changed = sorted(
        name for name in set(frozen) | set(current) if frozen.get(name) != current.get(name)
    )
    if changed:
        raise ValueError(
            "the frozen eval datasets changed since `patchpulse-evals freeze`: "
            + ", ".join(changed)
            + "; changing them is a ledgered ruling, a new freeze and a re-run"
        )
