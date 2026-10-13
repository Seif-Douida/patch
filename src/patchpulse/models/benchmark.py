"""Runs inside the benchmark container (plan Task 14): reviews a second for each model.

    python -m patchpulse.models.benchmark TEXTS_JSON MODEL_DIR [MODEL_DIR ...]

Prints one `family<TAB>reviews per second` line per model. Each model is warmed up on a few texts
first, so one-off loading costs don't count as throughput. Needs only the `scoring` group.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from patchpulse.models.artifact import load_scorer

WARM_UP = 20


def main(argv: Sequence[str] | None = None) -> int:
    texts_file, *model_dirs = argv if argv is not None else sys.argv[1:]
    texts = json.loads(Path(texts_file).read_text(encoding="utf-8"))
    for model_dir in model_dirs:
        scorer = load_scorer(Path(model_dir))
        scorer.predict_proba(texts[:WARM_UP])
        started = time.perf_counter()
        scorer.predict_proba(texts)
        per_second = len(texts) / (time.perf_counter() - started)
        print(f"{scorer.artifact.family}\t{per_second:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
