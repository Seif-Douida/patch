"""`patchpulse-ml reference` and `patchpulse-ml monitor` (Phase 7 Tasks 3 and 4): the drift
reference and the threshold calibration, run on Seif's machine from the local snapshot.

They read review text from `data/local` to score it, and print counts and numbers only.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from patchpulse.config import LabSettings
from patchpulse.labeling.silver import load_silver
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.artifact import load_scorer
from patchpulse.models.metrics import assessable_macro_f1
from patchpulse.models.snapshot import load_snapshot
from patchpulse.monitoring.alerts import ALERTS_FILE
from patchpulse.monitoring.calibrate import (
    calibrate,
    render_calibration,
    weekly_samples,
    write_thresholds,
)
from patchpulse.monitoring.reference import (
    REFERENCE_FILE,
    Reference,
    build_reference,
    read_reference,
    write_reference,
)

if TYPE_CHECKING:
    from patchpulse.monitoring.vectors import QdrantVectors

log = logging.getLogger("patchpulse.models.monitor")
DEFAULT_CHAMPION = Path("build") / "champion"
REPORT = Path("reports") / "monitoring" / "calibration.md"


def register(commands: Any) -> None:
    reference = commands.add_parser("reference", help="the drift reference").add_subparsers(
        dest="reference_command", required=True
    )
    build = reference.add_parser(
        "build", help=f"score silver with the champion; write {REFERENCE_FILE}"
    )
    build.add_argument("--champion", type=Path, default=DEFAULT_CHAMPION)
    monitor = commands.add_parser("monitor", help="drift monitoring").add_subparsers(
        dest="monitor_command", required=True
    )
    calibrate = monitor.add_parser(
        "calibrate", help="drift thresholds from past weeks; writes config/alerts.yaml"
    )
    calibrate.add_argument("--champion", type=Path, default=DEFAULT_CHAMPION)


def run(args: argparse.Namespace, settings: LabSettings) -> int:
    if args.command == "reference":
        return _reference_build(settings, champion=args.champion)
    if args.command == "monitor":
        return _calibrate(settings, champion=args.champion)
    raise ValueError(f"unknown command {args.command!r}")


def reference_from_silver(settings: LabSettings, champion: Path) -> Reference:
    """The champion on the silver training set (its training data) and holdout (its agreement
    with the teacher)."""
    snapshot = load_snapshot(settings.data_dir)
    silver = load_silver(settings.data_dir)
    scorer = load_scorer(champion)
    reviews = snapshot.reviews.set_index("review_id")
    train_ids = [int(i) for i in silver.train["review_id"]]
    holdout_ids = [int(i) for i in silver.holdout["review_id"]]
    texts = [str(reviews.at[i, "text"]) for i in train_ids]
    probs = scorer.predict_proba(texts)
    holdout_probs = scorer.predict_proba([str(reviews.at[i, "text"]) for i in holdout_ids])
    agreement = assessable_macro_f1(
        silver.holdout[list(ASPECTS)].to_numpy(dtype=bool), scorer.labels(holdout_probs)
    )
    return build_reference(
        model_version=scorer.artifact.version,
        languages=[str(reviews.at[i, "language"]) for i in train_ids],
        lengths=[len(t) for t in texts],
        labels=scorer.labels(probs),
        probs=probs,
        thresholds=np.asarray(scorer.artifact.cuts()),
        ids=train_ids,
        holdout_agreement=agreement,
    )


def _reference_build(settings: LabSettings, *, champion: Path) -> int:
    reference = reference_from_silver(settings, champion)
    write_reference(reference)
    print(
        f"{REFERENCE_FILE}: model {reference.model_version}, {reference.n} training reviews, "
        f"{len(reference.sample_ids)} sampled ids, teacher agreement (macro-F1 on the silver "
        f"holdout) {reference.teacher_agreement_macro_f1:.3f}"
    )
    return 0


def _calibrate(settings: LabSettings, *, champion: Path) -> int:
    """Every past week of the snapshot against the committed reference; vectors from the
    cluster's index by id (read only)."""
    if settings.qdrant_url is None or settings.qdrant_api_key is None:
        raise ValueError("calibration needs PP_QDRANT_URL and PP_QDRANT_API_KEY (read access)")
    # Imported here: the deploy's champion download runs this CLI without qdrant-client.
    from patchpulse.monitoring.vectors import QdrantVectors
    from patchpulse.retrieval.pin import read_pin as read_retrieval_pin
    from patchpulse.retrieval.qdrant import make_client

    reference = read_reference()
    scorer = load_scorer(champion)
    if scorer.artifact.version != reference.model_version:
        raise ValueError(
            f"{REFERENCE_FILE} is for model {reference.model_version}, the champion is "
            f"{scorer.artifact.version}: run `patchpulse-ml reference build` first"
        )
    snapshot = load_snapshot(settings.data_dir)
    created = pd.to_datetime(snapshot.reviews["date"]).dt.date
    client = make_client(settings.qdrant_url, settings.qdrant_api_key)
    try:
        vectors = CachedVectors(QdrantVectors(client, read_retrieval_pin().collection))
        result = calibrate(
            reference,
            weekly_samples(snapshot, scorer, start=min(created), end=max(created)),
            vectors,
        )
    finally:
        client.close()
    write_thresholds(result.thresholds)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render_calibration(result), encoding="utf-8", newline="\n")
    print(
        f"{result.weeks} weeks measured, {result.skipped} skipped; thresholds written to "
        f"{ALERTS_FILE}: "
        + ", ".join(f"{rule} {value:.4g}" for rule, value in sorted(result.thresholds.items()))
        + f"; report {REPORT}"
    )
    return 0


class CachedVectors:
    """The reference's vectors are asked for every week: fetch each id once, in batches, and
    remember the ids the index doesn't hold."""

    def __init__(self, inner: QdrantVectors) -> None:
        self.inner = inner
        self.cache: dict[int, NDArray[np.float32]] = {}
        self.missing: set[int] = set()  # ids the index doesn't hold: never asked for again

    def vectors(self, ids: Sequence[int]) -> NDArray[np.float32]:
        wanted = [int(i) for i in ids if int(i) not in self.cache and int(i) not in self.missing]
        if wanted:
            found = self.inner.by_id(wanted)
            self.cache.update(found)
            self.missing.update(i for i in wanted if i not in found)
        from patchpulse.monitoring.vectors import stack  # needs qdrant-client, as `inner` does

        return stack(self.cache, ids)
