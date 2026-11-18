"""`patchpulse-pipeline retrain`: pp-retrain's run (Phase 7 design D5).

1. The data: silver and spot-check training labels, gold dev and test, the recent holdout
   (`retrain_data`), with each review's text from SQL.
2. A new S0, trained as in Phase 2: thresholds tuned on gold dev, its out-of-fold macro-F1 the
   number it is compared on.
3. Both models scored on the same reviews: the recent holdout, gold dev out of fold (the champion
   through the same procedure, which reproduces its recorded value) and gold test (reported only).
   The challenger's reviews a second are timed on one thread.
4. The promotion rule (`promotion.decide_promotion`, `config/retrain.yaml`).
5. One MLflow run in `retraining`: ids hashes and counts, versions, the metrics, tags `decision`
   and `execution`; the table, the challenger's drift reference and the result as reports.
   Then the challenger is registered with the `challenger` alias. retrain.yml opens the promotion
   pull request; nothing here touches the champion.

No review text leaves: params are ids hashes and counts, the reports are numbers, and S0 hashes
its n-grams (ADR-021).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path

import mlflow
import numpy as np
from sqlalchemy import Engine
from threadpoolctl import threadpool_limits

from patchpulse.labeling.taxonomy import TAXONOMY_VERSION
from patchpulse.models.artifact import (
    MODEL_FILE,
    Scorer,
    load_scorer,
    new_version,
    read_artifact,
    sha256_file,
)
from patchpulse.models.champion import register_challenger
from patchpulse.models.metrics import assessable_macro_f1, macro_f1
from patchpulse.models.promotion import (
    Decision,
    PairedSet,
    PromotionRule,
    RetrainResult,
    decide_promotion,
    render_table,
    result_json,
)
from patchpulse.models.retrain_data import RetrainData, load_retrain_data
from patchpulse.models.s0 import S0Params, save_s0, train_s0
from patchpulse.models.split import Split
from patchpulse.models.text import CLEANING_VERSION
from patchpulse.models.thresholds import CV_FOLDS, CV_SEED, score_on_dev
from patchpulse.models.tracking import log_ids_hash, log_report, tracking_run
from patchpulse.monitoring.reference import build_reference, write_reference
from patchpulse.monitoring.store import MetricRow, write_metrics

RETRAIN_EXPERIMENT = "retraining"
REPORTS_DIR = Path("reports") / "retrain"
SPEED_REVIEWS = 2000  # timed on one thread, after a short warm-up
EXECUTION_ENV = "CONTAINER_APP_JOB_EXECUTION_NAME"  # set by Container Apps in each execution
MIN_DEV = CV_FOLDS  # out-of-fold thresholds need a review per fold at least

log = logging.getLogger(__name__)


class RetrainError(RuntimeError):
    """The data can't make a challenger worth comparing."""


def reviews_per_second(scorer: Scorer, texts: Sequence[str]) -> float:
    sample = list(texts[:SPEED_REVIEWS])
    with threadpool_limits(limits=1):
        scorer.predict_proba(sample[:50])  # warm-up
        started = time.perf_counter()
        scorer.predict_proba(sample)
        elapsed = time.perf_counter() - started
    return len(sample) / max(elapsed, 1e-9)


def _labels(scorer: Scorer, split: Split) -> np.ndarray:
    return scorer.labels(scorer.predict_proba(split.texts))


def _test_f1(scorer: Scorer, test: Split) -> float:
    return macro_f1(test.labels, _labels(scorer, test)) if test.ids else float("nan")


def _check(data: RetrainData) -> None:
    if not data.train.ids:
        raise RetrainError("no training labels found")
    if len(data.dev.ids) < MIN_DEV:
        raise RetrainError(f"gold dev has {len(data.dev.ids)} reviews; need {MIN_DEV}")


def run_retrain(
    engine: Engine,
    *,
    champion_dir: Path,
    out_dir: Path,
    today: date,
    rule: PromotionRule,
    clock: Callable[[], datetime],
) -> RetrainResult:
    champion = load_scorer(champion_dir)  # checked first: a bad champion costs no training
    champion_artifact = read_artifact(champion_dir)
    with engine.connect() as connection:
        data = load_retrain_data(connection, today=today)
    _check(data)
    log.info(
        "retrain: %d training labels, %d dev, %d test, %d recent; %d no longer in SQL",
        len(data.train.ids),
        len(data.dev.ids),
        len(data.test.ids),
        len(data.recent.ids),
        data.missing,
    )

    params = S0Params()
    started = time.perf_counter()
    model = train_s0(data.train.texts, data.train.labels, params=params)
    train_seconds = time.perf_counter() - started
    dev = score_on_dev(data.dev, model.predict_proba(data.dev.texts))
    version = new_version(clock())
    model_dir = out_dir / f"s0-{version}"
    artifact = save_s0(model, model_dir, thresholds=dev.thresholds, version=version)
    challenger = load_scorer(model_dir)  # through its checksums, as the jobs image will load it

    champion_dev = score_on_dev(data.dev, champion.predict_proba(data.dev.texts)).cv_macro_f1
    speed = reviews_per_second(challenger, data.train.texts)
    recent = PairedSet(
        y_true=data.recent.labels,
        champion=_labels(champion, data.recent),
        challenger=_labels(challenger, data.recent),
        languages=data.languages["recent"],
    )
    decision = decide_promotion(
        recent,
        dev_champion=champion_dev,
        dev_challenger=dev.cv_macro_f1,
        reviews_per_second=speed,
        rule=rule,
    )
    test_champion, test_challenger = _test_f1(champion, data.test), _test_f1(challenger, data.test)
    counts = {
        "train": len(data.train.ids),
        "dev": len(data.dev.ids),
        "test": len(data.test.ids),
        "recent": len(data.recent.ids),
        "missing": data.missing,
    }
    table = render_table(
        decision,
        champion_version=champion_artifact.version,
        challenger_version=version,
        test_champion=test_champion,
        test_challenger=test_challenger,
        counts=counts,
    )

    train_probs = challenger.predict_proba(data.train.texts)
    agreement = (
        assessable_macro_f1(recent.y_true, recent.challenger)
        if len(recent.y_true)
        else float("nan")
    )
    reference = build_reference(
        model_version=version,
        languages=data.languages["train"],
        lengths=[len(text) for text in data.train.texts],
        labels=challenger.labels(train_probs),
        probs=train_probs,
        thresholds=dev.thresholds,
        ids=data.train.ids,
        # The weekly agreement is the spot-check teacher's on new reviews; the recent holdout is
        # the same teacher on held-out ones. 0 when there is none: such a challenger never passes.
        holdout_agreement=0.0 if np.isnan(agreement) else agreement,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table_path = REPORTS_DIR / f"{version}.md"
    table_path.write_text(table, encoding="utf-8", newline="\n")
    reference_path = REPORTS_DIR / f"{version}-reference.json"
    write_reference(reference, reference_path)

    metrics: dict[str, float] = {
        "champion.recent_macro_f1": _finite(decision.checks[0].champion),
        "challenger.recent_macro_f1": _finite(decision.checks[0].challenger),
        "champion.dev_cv_macro_f1": champion_dev,
        "challenger.dev_cv_macro_f1": dev.cv_macro_f1,
        "champion.test_macro_f1": test_champion,
        "challenger.test_macro_f1": test_challenger,
        "challenger.reviews_per_second": speed,
        "challenger.train_seconds": train_seconds,
        "challenger.teacher_agreement_macro_f1": reference.teacher_agreement_macro_f1,
        **{f"check.{c.name}": float(c.passed) for c in decision.checks},
    }
    record = {
        "family": "s0",
        "challenger_version": version,
        "champion_version": champion_artifact.version,
        "champion_family": champion_artifact.family,
        "taxonomy_version": TAXONOMY_VERSION,
        "cleaning_version": CLEANING_VERSION,
        "cv_folds": CV_FOLDS,
        "cv_seed": CV_SEED,
        "missing": data.missing,
        "today": today.isoformat(),
        **{f"teacher.{name}": n for name, n in data.teachers.items()},
        **{f"rule.{name}": value for name, value in rule.model_dump().items()},
        **{f"file_sha256.{name}": digest for name, digest in artifact.files.items()},
        "word_ngrams": params.word_ngrams,
        "char_ngrams": params.char_ngrams,
        "hash_bits": params.hash_bits,
        "c": params.c,
    }
    tags = {
        "decision": "promote" if decision.promote else "keep",
        "execution": os.environ.get(EXECUTION_ENV, "local"),
    }
    with tracking_run(
        RETRAIN_EXPERIMENT, params=record, tags=tags, run_name=f"s0-{version}"
    ) as run:
        for name in ("train", "dev", "test", "recent"):
            log_ids_hash(name, getattr(data, name).ids)
        mlflow.log_metrics({k: v for k, v in metrics.items() if not np.isnan(v)})
        log_report(table_path)
        log_report(reference_path)
        run_id = str(run.info.run_id)

    registered = register_challenger(mlflow.MlflowClient(), model_dir)
    result = RetrainResult(
        version=version,
        decision=decision,
        mlflow_run_id=run_id,
        registered_version=registered,
        model_json_sha256=sha256_file(model_dir / MODEL_FILE),
        table=table,
    )
    result_path = REPORTS_DIR / f"{version}-result.json"
    result_path.write_text(result_json(result), encoding="utf-8", newline="\n")
    log_report(result_path, run_id=run_id)
    client = mlflow.MlflowClient()
    client.set_tag(run_id, "registered_version", str(registered))
    client.set_tag(run_id, "model_json_sha256", result.model_json_sha256)
    _record_decision(engine, decision, today=today, recent=len(data.recent.ids))
    return result


def _record_decision(engine: Engine, decision: Decision, *, today: date, recent: int) -> None:
    """The Monitoring page's last decision (Phase 7 Task 15), with the recent holdout's gain."""
    first = decision.checks[0]
    gain = (
        None
        if first.champion is None or first.challenger is None
        else first.challenger - first.champion
    )
    word = "promote" if decision.promote else "keep"
    row = MetricRow("retrain", f"decision.{word}", "run", today, today, gain, None, None, recent)
    with engine.begin() as connection:
        write_metrics(connection, [row], run_id=None)


def _finite(value: float | None) -> float:
    return float("nan") if value is None else float(value)
