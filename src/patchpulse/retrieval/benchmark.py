"""The speed floor (design D3), measured where each embedder will run (plan Task 5).

Passages are timed in `docker run --cpus=1 --cpuset-cpus=0 --memory=2g`, like `pp-nightly`; single
queries in `--cpus=0.5 --memory=1g`, like `pp-api`. Each container installs the `scoring` group's
pinned packages and runs this module on read-only mounts:

    python -m patchpulse.retrieval.benchmark passages TEXTS_JSON MODEL_DIR [MODEL_DIR ...]
    python -m patchpulse.retrieval.benchmark queries QUERIES_JSON MODEL_DIR [MODEL_DIR ...]

which prints `name<TAB>passages a second`, or `name<TAB>p50 ms<TAB>p95 ms`, per model. The
container side needs only the `scoring` group; the host side imports the rest when called.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from patchpulse.retrieval.embedders import MODEL_FILE, OnnxEmbedder, read_manifest

if TYPE_CHECKING:
    from patchpulse.models.snapshot import Snapshot

BENCH_IMAGE = "python:3.12-slim"
SRC_DIR = Path(__file__).resolve().parents[2]  # the directory holding the `patchpulse` package
NIGHTLY_REVIEWS = 650  # about a night's new index-set reviews
INDEX_BUDGET_S = 300  # PP_INDEX_BUDGET_MINUTES
QUERY_P95_MS = 300.0
WARM_UP = 5
Run = Callable[..., "subprocess.CompletedProcess[str]"]

# Query-like strings for timing only: the eval's queries stay unseen until they're frozen.
_SUBJECTS = (
    ("frame rate drops", "chutes de fps"),
    ("crashes on startup", "plantages au lancement"),
    ("matchmaking queues", "temps d'attente du matchmaking"),
    ("server outages", "pannes de serveurs"),
    ("the price of the new DLC", "le prix du nouveau DLC"),
    ("the story and characters", "l'histoire et les personnages"),
    ("controller support", "le support manette"),
    ("kernel anti-cheat", "l'anti-triche noyau"),
    ("the new content update", "le contenu de la mise a jour"),
    ("weapon balance changes", "l'equilibrage des armes"),
)
_TEMPLATES = (
    ("{}", "{}"),
    ("players complaining about {}", "joueurs qui se plaignent de {}"),
    ("{} after the latest patch", "{} apres le dernier patch"),
    ("reviews about {} this month", "avis sur {} ce mois-ci"),
    ("is {} getting better", "est-ce que {} s'ameliore"),
)
BENCH_QUERIES: list[str] = [
    template[lang].format(subject[lang])
    for subject in _SUBJECTS
    for template in _TEMPLATES
    for lang in (0, 1)
]


@dataclass(frozen=True)
class EmbedderSpeed:
    name: str
    passages_per_s: float  # one vCPU
    query_p50_ms: float  # half a vCPU, one query at a time
    query_p95_ms: float
    size_mb: float


def passes_floor(speed: EmbedderSpeed) -> bool:
    """A night's reviews within the index budget, and a query fast enough for the API."""
    in_budget = speed.passages_per_s * INDEX_BUDGET_S >= NIGHTLY_REVIEWS
    return in_budget and speed.query_p95_ms < QUERY_P95_MS


def newest_index_texts(snapshot: Snapshot, n: int) -> list[str]:
    """The cleaned texts of the `n` newest reviews in the index set. A snapshot pulled before
    `core.review_index` existed has no index set; the rules are then applied to its reviews."""
    from patchpulse.models.text import clean_text
    from patchpulse.retrieval.rules import eligible

    reviews = snapshot.reviews
    if not snapshot.index.empty:
        newest = snapshot.index.sort_values(["created_ts", "review_id"], ascending=False)
        texts = reviews.set_index("review_id")["text"]
        chosen = [str(texts[rid]) for rid in newest["review_id"].head(n)]
    else:
        ordered = reviews.sort_values(["date", "review_id"], ascending=False)
        chosen = []
        for language, text in zip(ordered["language"], ordered["text"], strict=True):
            if eligible(str(language), str(text)):
                chosen.append(str(text))
                if len(chosen) == n:
                    break
    return [clean_text(text) for text in chosen]


def _container(
    mode: str,
    limits: list[str],
    model_dirs: Sequence[Path],
    inputs: Sequence[str],
    *,
    image: str,
    run: Run,
    requirements: str,
) -> str:
    with tempfile.TemporaryDirectory(prefix="pp-bench-") as work_dir:
        work = Path(work_dir)
        (work / f"{mode}.json").write_text(json.dumps(list(inputs)), encoding="utf-8")
        (work / "requirements.txt").write_text(requirements, encoding="utf-8")
        mounts = [
            "-v",
            f"{SRC_DIR.resolve().as_posix()}:/app/src:ro",
            "-v",
            f"{work.resolve().as_posix()}:/work:ro",
        ]
        targets = []
        for index, model_dir in enumerate(model_dirs):
            targets.append(f"/models/{index}")
            mounts += ["-v", f"{model_dir.resolve().as_posix()}:{targets[-1]}:ro"]
        script = (
            "pip install --quiet --no-cache-dir --disable-pip-version-check "
            "--root-user-action=ignore -r /work/requirements.txt && PYTHONPATH=/app/src "
            f"python -m patchpulse.retrieval.benchmark {mode} /work/{mode}.json {' '.join(targets)}"
        )
        command = ["docker", "run", "--rm", *limits, *mounts, image, "sh", "-c", script]
        result = run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"the {mode} benchmark container failed (exit {result.returncode}): "
            f"{result.stderr.strip()[-2000:]}"
        )
    return str(result.stdout)


def _rows(stdout: str, names: set[str]) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {}
    for line in stdout.splitlines():
        name, *numbers = line.split("\t")
        if name in names and numbers:
            rows[name] = [float(number) for number in numbers]
    return rows


def benchmark_embedders(
    model_dirs: Sequence[Path],
    texts: Sequence[str],
    queries: Sequence[str],
    *,
    image: str = BENCH_IMAGE,
    run: Run = subprocess.run,
    requirements: str | None = None,
) -> list[EmbedderSpeed]:
    """Each embedder's speed: passages in a one-CPU container, single queries in a half-CPU one.
    Everything is mounted read-only; the texts stay on this machine."""
    if requirements is None:
        from patchpulse.models.evaluate import scoring_requirements

        requirements = scoring_requirements()
    names = [read_manifest(model_dir)["spec"]["name"] for model_dir in model_dirs]
    passages = _container(
        "passages",
        ["--cpus=1", "--cpuset-cpus=0", "--memory=2g"],
        model_dirs,
        texts,
        image=image,
        run=run,
        requirements=requirements,
    )
    single = _container(
        "queries",
        ["--cpus=0.5", "--cpuset-cpus=0", "--memory=1g"],
        model_dirs,
        queries,
        image=image,
        run=run,
        requirements=requirements,
    )
    rates, latencies = _rows(passages, set(names)), _rows(single, set(names))
    missing = [name for name in names if name not in rates or name not in latencies]
    if missing:
        raise RuntimeError(f"the benchmark printed no result for {', '.join(missing)}")
    return [
        EmbedderSpeed(
            name=name,
            passages_per_s=rates[name][0],
            query_p50_ms=latencies[name][0],
            query_p95_ms=latencies[name][1],
            size_mb=round((model_dir / MODEL_FILE).stat().st_size / 1e6, 1),
        )
        for name, model_dir in zip(names, model_dirs, strict=True)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """The container side: time each model on one thread."""
    mode, inputs_file, *model_dirs = argv if argv is not None else sys.argv[1:]
    inputs = json.loads(Path(inputs_file).read_text(encoding="utf-8"))
    for model_dir in model_dirs:
        embedder = OnnxEmbedder(Path(model_dir), threads=1)
        if mode == "passages":
            embedder.embed_passages(inputs[:WARM_UP])
            started = time.perf_counter()
            embedder.embed_passages(inputs)
            line = f"{len(inputs) / (time.perf_counter() - started):.2f}"
        else:
            embedder.embed_queries(inputs[:WARM_UP])
            times = []
            for query in inputs:
                started = time.perf_counter()
                embedder.embed_queries([query])
                times.append((time.perf_counter() - started) * 1000)
            line = f"{np.percentile(times, 50):.1f}\t{np.percentile(times, 95):.1f}"
        print(f"{embedder.spec.name}\t{line}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
