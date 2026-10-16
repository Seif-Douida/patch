"""`patchpulse-ml`: Phase 2's local commands (snapshot, labeling, training), run on Seif's machine.

    pull         Copy reviews from Azure SQL into data/local (wakes the database; needs --yes).
    gold sample        Draw the gold set and Seif's audit set from the snapshot (data/gold/).
    gold export-batch  Write a labeling batch (pilot, batch-1..4) for a subagent (data/local/).
    gold import        Check a subagent's labels against its batch and add them to the gold set.
    gold agreement     Compare Seif's blind audit with Claude's labels (reports/aspects/).
    teacher bakeoff    Label gold dev with each teacher candidate and pick one (reports/aspects/,
                       MLflow). Needs PP_GEMINI_API_KEY and the MLFLOW_TRACKING_* settings.
    silver sample      Draw the ~30k silver reviews and their holdout (data/local/silver/).
    teacher label      Label the silver set (or `--set gold`, for the eval report) with the chosen
                       teacher; resumable after a quota stop.
    train s0           Train S0 (TF-IDF + logistic regression) on silver train, tune its thresholds
                       on gold dev, save it under build/models/ and log it to MLflow (`students`).
    train s1a          The same for S1a (frozen e5 embeddings + logistic regression). Downloads e5
                       once and embeds the silver set on the CPU (~10 min, then cached).
    train s1b          The same for S1b (e5 fine-tuned on the GTX 1650, exported to int8 ONNX).
                       Needs the gpu group: `uv run --group gpu patchpulse-ml train s1b`.
                       `--checkpoint build/checkpoints/s1b-...` reuses a finished fine-tune.
    evaluate choose    Pick the champion on gold dev by the fixed rules, after timing each model
                       in a 1-CPU, 2 GiB Docker container (reports/aspects/champion-choice.json).
    evaluate report    Score the teacher and the students on gold test and Seif's 150, once, and
                       write reports/aspects/phase-2-eval.md. Needs `evaluate choose` and the
                       teacher's gold labels (`teacher label --set gold`).
    champion register  Register the evaluated models on DagsHub's MLflow registry (aliases
                       `champion` and `challenger`) and pin the champion in models/champion.yaml.
    champion download  Download the pinned champion (`--to build/champion`), refusing it unless
                       it matches the pin; deploy.yml runs it before building the jobs image.

Exit codes: 0 success, 1 the command failed, 2 bad configuration or a refused action.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import httpx2
import mlflow
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from patchpulse.config import LabSettings, SettingsError, get_lab_settings
from patchpulse.db.engine import connect_with_resume_retry, make_entra_engine
from patchpulse.labeling.agreement import agreement, render_agreement
from patchpulse.labeling.audit import SEIF_LABELS_FILE
from patchpulse.labeling.bakeoff import (
    REPORT_PATH,
    choose_teacher,
    log_bakeoff,
    render_report,
    run_bakeoff,
)
from patchpulse.labeling.cache import TEACHER_CACHE_FILE, LabelCache
from patchpulse.labeling.gemini import GeminiClient
from patchpulse.labeling.gold import (
    AUDIT_FILE,
    BATCHES,
    CLAUDE_LABELS_FILE,
    SAMPLE_FILE,
    batch_ids,
    choose_audit,
    export_batch,
    import_labels,
    pilot_ids,
    read_labels,
    sample_gold,
    split_dev_test,
    write_gold,
)
from patchpulse.labeling.limits import USAGE_FILE, RequestLimiter
from patchpulse.labeling.openai_compat import OpenAICompatClient
from patchpulse.labeling.silver import (
    IDS_FILE,
    SILVER_DIR,
    SILVER_SIZE,
    label_silver,
    label_silver_until_done,
    prevalence_report,
    sample_silver,
    split_holdout,
    write_silver_ids,
)
from patchpulse.labeling.teacher import TextModel, label_reviews, prompt_hash, review_inputs
from patchpulse.labeling.teachers import ModelConfig, load_model_config
from patchpulse.models.artifact import MODEL_FILE, Scorer, load_scorer, record_metrics
from patchpulse.models.champion import (
    PIN_FILE,
    ChampionError,
    download_champion,
    pin_for,
    read_pin,
    register_models,
    write_pin,
)
from patchpulse.models.embed_gpu import EMBEDDINGS_DIR, cached_embeddings
from patchpulse.models.encoder import (
    E5_MODEL_ID,
    E5_REVISION,
    MAX_TOKENS,
    QUERY_PREFIX,
    OnnxClassifier,
    OnnxEncoder,
    build_encoder,
)
from patchpulse.models.evaluate import (
    COST_ORDER,
    TEACHER,
    Candidate,
    Choice,
    GoldSet,
    ModelFacts,
    Outputs,
    benchmark_cpu,
    choose_champion,
    evaluate_models,
    render_eval_report,
)
from patchpulse.models.s0 import S0Params, save_s0, train_s0
from patchpulse.models.s1a import HEAD_FILE, S1aParams, S1aScorer, load_head, save_s1a, train_s1a
from patchpulse.models.s1b import (
    S1B_FILE,
    S1bParams,
    S1bScorer,
    S1bTraining,
    read_training,
    save_s1b,
    sigmoid,
)
from patchpulse.models.snapshot import explain_pull_error, load_snapshot, pull
from patchpulse.models.text import clean_texts
from patchpulse.models.thresholds import cv_predictions
from patchpulse.models.tracking import (
    TrackingError,
    log_ids_hash,
    log_report,
    require_tracking,
    tracking_run,
)
from patchpulse.models.training import (
    CV_FOLDS,
    CV_SEED,
    MODELS_DIR,
    DevScores,
    TrainingData,
    label_matrix,
    load_training_data,
    log_student,
    new_version,
    score_on_dev,
)

log = logging.getLogger("patchpulse.ml")

ENCODERS_DIR = Path("build/encoders")
CHECKPOINTS_DIR = Path("build/checkpoints")
CPU_PREVIEW_REVIEWS = 500
CHOICE_PATH = Path("reports/aspects/champion-choice.json")
EVAL_REPORT_PATH = Path("reports/aspects/phase-2-eval.md")
EVAL_EXPERIMENT = "evaluation"
EVAL_SEED = 20261012
BENCHMARK_REVIEWS = 2_000

PULL_COST = (
    "`pull` wakes Azure SQL, which costs about 2,000-2,500 of the month's 100,000 free vCore-s. "
    "Run it again with --yes to go ahead."
)


def configure_logging() -> None:
    """INFO for PatchPulse; azure-identity's credential lines and HTTP dumps only on warnings."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("patchpulse").setLevel(logging.INFO)
    logging.getLogger("azure").setLevel(logging.WARNING)


def _pull(settings: LabSettings, *, yes: bool) -> int:
    if settings.pull_host is None:
        raise SettingsError("missing or invalid settings: PP_PULL_HOST")
    if not yes:
        print(f"patchpulse-ml: {PULL_COST}", file=sys.stderr)
        return 2
    engine = make_entra_engine(settings.pull_host, settings.pull_database)
    try:
        connect_with_resume_retry(engine).close()  # wake a paused database first
        result = pull(engine, settings.data_dir, now=datetime.now(UTC))
    except Exception as error:
        fix = explain_pull_error(error)
        if fix is None:
            raise
        print(f"patchpulse-ml: {fix}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    log.info(
        "pulled %d rows; the snapshot has %d reviews (up to run %d) and an index set of %d in %s",
        result.reviews_new,
        result.reviews_total,
        result.max_run_id,
        result.index_total,
        settings.data_dir,
    )
    return 0


def _gold_sample(settings: LabSettings, *, seed: int, force: bool) -> int:
    if (settings.gold_dir / SAMPLE_FILE).exists() and not force:
        print(
            f"patchpulse-ml: {settings.gold_dir / SAMPLE_FILE} already exists. Re-sampling would "
            "orphan any labels made for it; pass --force only if none exist yet.",
            file=sys.stderr,
        )
        return 2
    sample = split_dev_test(sample_gold(load_snapshot(settings.data_dir), seed=seed), seed=seed)
    audit = choose_audit(sample, seed=seed)
    write_gold(sample, audit, settings.gold_dir)
    strata = sample.groupby(["language", "split"]).size().to_string()
    log.info(
        "gold set written to %s (%d reviews, audit %d)\n%s",
        settings.gold_dir,
        len(sample),
        len(audit),
        strata,
    )
    return 0


BATCH_NAMES = ["pilot", *(f"batch-{k}" for k in range(1, BATCHES + 1))]


def _batch(settings: LabSettings, name: str) -> tuple[pd.DataFrame, list[int]]:
    sample = pd.read_csv(settings.gold_dir / SAMPLE_FILE)
    if name == "pilot":
        return sample, pilot_ids(sample)
    return sample, batch_ids(sample, int(name.removeprefix("batch-")))


def _gold_export(settings: LabSettings, *, name: str) -> int:
    sample, ids = _batch(settings, name)
    path = settings.data_dir / "gold_batches" / f"{name}.jsonl"
    export_batch(sample, load_snapshot(settings.data_dir), ids, path)
    log.info("wrote %d reviews to %s", len(ids), path)
    return 0


def _gold_import(settings: LabSettings, *, name: str, file: Path, guidelines: str) -> int:
    _, ids = _batch(settings, name)
    try:
        imported = import_labels(
            file,
            ids,
            annotator="claude",
            guidelines_version=guidelines,
            gold_dir=settings.gold_dir,
            reasons_path=settings.data_dir / "gold_reasons.csv",
        )
    except ValueError as error:
        print(f"patchpulse-ml: {file}: {error}", file=sys.stderr)
        return 1
    counts = pd.Series([a for labels in imported.labels.values() for a in labels]).value_counts()
    log.info("imported %d labels (%s)\n%s", len(imported.labels), name, counts.to_string())
    return 0


REPORTS_DIR = Path("reports/aspects")


def _gold_agreement(settings: LabSettings) -> int:
    audit = [int(i) for i in pd.read_csv(settings.gold_dir / AUDIT_FILE)["review_id"]]
    seif = read_labels(settings.gold_dir / SEIF_LABELS_FILE)
    unlabeled = [i for i in audit if i not in seif]
    if unlabeled:
        print(
            f"patchpulse-ml: the audit isn't finished: {len(audit) - len(unlabeled)} of "
            f"{len(audit)} labeled",
            file=sys.stderr,
        )
        return 2
    report = agreement(read_labels(settings.gold_dir / CLAUDE_LABELS_FILE), seif, audit)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "gold-agreement.md"
    path.write_text(render_agreement(report), encoding="utf-8", newline="\n")
    log.info(
        "agreement %s (mean kappa %s); report: %s",
        "passed" if report.passed else "FAILED",
        report.macro_kappa,
        path,
    )
    return 0 if report.passed else 1


def _teacher_bakeoff(settings: LabSettings, *, names: list[str] | None) -> int:
    require_tracking()  # before any request, so a missing setting costs nothing
    config = load_model_config()
    unknown = sorted(set(names or []) - set(config.candidates))
    if unknown:
        print(f"patchpulse-ml: unknown candidates {unknown}", file=sys.stderr)
        return 2
    candidates = {name: config.candidates[name] for name in names or config.candidates}
    http = httpx2.Client()
    clients = {name: _client_for(settings, model, http) for name, model in candidates.items()}
    sample = pd.read_csv(settings.gold_dir / SAMPLE_FILE)
    dev = sorted(int(i) for i in sample.loc[sample["split"] == "dev", "review_id"])
    reviews = review_inputs(load_snapshot(settings.data_dir), dev)
    gold = read_labels(settings.gold_dir / CLAUDE_LABELS_FILE)
    result = run_bakeoff(
        candidates,
        reviews,
        gold,
        make_client=lambda name, _: clients[name],
        cache=LabelCache(settings.data_dir / TEACHER_CACHE_FILE),
    )
    quota = [c.name for c in result.candidates if c.stopped_for_quota]
    outage = [c.name for c in result.candidates if c.stopped_for_outage]
    if quota or outage:
        reasons = [f"the daily quota ran out for {', '.join(quota)}"] if quota else []
        reasons += [f"the server stayed unavailable for {', '.join(outage)}"] if outage else []
        print(
            f"patchpulse-ml: no verdict: {'; '.join(reasons)}. The labels so far are cached; "
            "run the same command again later (after midnight Pacific time for the quota).",
            file=sys.stderr,
        )
        return 1
    winner = choose_teacher(result)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(result, winner), encoding="utf-8", newline="\n")
    run_id = log_bakeoff(result, winner, REPORT_PATH)
    log.info("teacher: %s; report: %s; MLflow run %s", winner, REPORT_PATH, run_id)
    return 0


def _silver_sample(settings: LabSettings, *, seed: int, n: int, force: bool) -> int:
    ids_path = settings.data_dir / SILVER_DIR / IDS_FILE
    if ids_path.exists() and not force:
        print(
            f"patchpulse-ml: {ids_path} already exists. Re-sampling would orphan its labels' "
            "split; pass --force only to start the silver set over.",
            file=sys.stderr,
        )
        return 2
    snapshot = load_snapshot(settings.data_dir)
    gold = pd.read_csv(settings.gold_dir / SAMPLE_FILE)["review_id"]
    ids = sample_silver(snapshot, exclude_ids={int(i) for i in gold}, n=n, seed=seed)
    train, holdout = split_holdout(ids, seed=seed)
    write_silver_ids(settings.data_dir, train, holdout)
    chosen = snapshot.reviews.set_index("review_id").loc[ids]
    mix = chosen["language"].value_counts(normalize=True).head(8).round(3).to_string()
    log.info(
        "silver: %d train, %d holdout in %s; language mix:\n%s",
        len(train),
        len(holdout),
        ids_path,
        mix,
    )
    return 0


def _client_for(settings: LabSettings, model: ModelConfig, http: httpx2.Client) -> TextModel:
    """The teacher client for a model's provider; a missing setting is named before any request."""
    if model.provider == "openai_compat":
        if settings.llm_server_url is None or settings.llm_server_key is None:
            raise SettingsError("missing or invalid settings: PP_LLM_SERVER_URL, PP_LLM_SERVER_KEY")
        key = settings.llm_server_key.get_secret_value()
        return OpenAICompatClient(key, settings.llm_server_url, http, model)
    if settings.gemini_api_key is None:
        raise SettingsError("missing or invalid settings: PP_GEMINI_API_KEY")
    limiter = RequestLimiter(model, state_path=settings.data_dir / USAGE_FILE)
    return GeminiClient(settings.gemini_api_key.get_secret_value(), http, limiter, model)


def _teacher_label(
    settings: LabSettings,
    *,
    review_set: str,
    candidate: str | None,
    keep_going: bool,
    workers: int,
) -> int:
    if review_set == "gold" and keep_going:
        print("patchpulse-ml: --keep-going is for the silver set only", file=sys.stderr)
        return 2
    config = load_model_config()
    name = candidate or config.teacher
    if name is None:
        print(
            "patchpulse-ml: no teacher chosen yet: run `patchpulse-ml teacher bakeoff`, then set "
            "`teacher:` in config/models.yaml (or pass --candidate)",
            file=sys.stderr,
        )
        return 2
    if name not in config.candidates:
        print(f"patchpulse-ml: unknown candidate {name!r}", file=sys.stderr)
        return 2
    model = config.candidates[name]
    if model.provider == "gemini" and workers > 1:
        print(
            "patchpulse-ml: Gemini's free tier is paced one request at a time; drop --workers",
            file=sys.stderr,
        )
        return 2
    client = _client_for(settings, model, httpx2.Client())
    snapshot = load_snapshot(settings.data_dir)
    cache = LabelCache(settings.data_dir / TEACHER_CACHE_FILE)
    if review_set == "gold":
        sample = pd.read_csv(settings.gold_dir / SAMPLE_FILE)
        ids = sorted(int(i) for i in sample["review_id"])
        gold_run = label_reviews(
            client, cache, review_inputs(snapshot, ids), model=model.cache_name, workers=workers
        )
        log.info(
            "%s on gold: %d labeled now, %d cached before, %d remaining",
            name,
            gold_run.labeled,
            gold_run.cached,
            gold_run.remaining,
        )
        if gold_run.stopped_for_quota or gold_run.stopped_for_outage:
            log.warning("the teacher stopped early; run the same command again later")
        return 0
    if keep_going:
        run = label_silver_until_done(
            client, cache, snapshot, settings.data_dir, model=model.cache_name, workers=workers
        )
    else:
        run = label_silver(
            client, cache, snapshot, settings.data_dir, model=model.cache_name, workers=workers
        )
    log.info(
        "%s: %d labeled now, %d cached before, %d remaining",
        name,
        run.labeled,
        run.cached,
        run.remaining,
    )
    log.info("silver positives:\n%s", prevalence_report(settings.data_dir, snapshot))
    if run.stopped_for_quota:
        log.warning("the daily quota ran out; run the same command after midnight Pacific time")
    if run.stopped_for_outage:
        log.warning("the teacher's server stayed unavailable; run the same command again later")
    return 0


def _train_s0(settings: LabSettings) -> int:
    require_tracking()  # before training, so a missing setting costs nothing
    data = load_training_data(settings.data_dir, settings.gold_dir)
    params = S0Params()
    log.info("training S0 on %d silver reviews", len(data.train.ids))
    started = time.monotonic()
    model = train_s0(data.train.texts, data.train.labels, params=params)
    train_seconds = time.monotonic() - started
    scores = score_on_dev(data.dev, model.predict_proba(data.dev.texts))
    version = new_version()
    model_dir = MODELS_DIR / f"s0-{version}"
    artifact = save_s0(model, model_dir, thresholds=scores.thresholds, version=version)
    size_mb = sum((model_dir / name).stat().st_size for name in artifact.files) / 1e6
    run_id = log_student(
        family="s0",
        version=version,
        params={
            "word_ngrams": params.word_ngrams,
            "char_ngrams": params.char_ngrams,
            "hash_bits": params.hash_bits,
            "c": params.c,
        },
        data=data,
        scores=scores,
        model_dir=model_dir,
        metrics={"train_seconds": train_seconds, "model_mb": size_mb},
    )
    _print_scores(f"S0 {version}", scores, f"trained in {train_seconds:.0f} s, {size_mb:.1f} MB")
    print(f"model: {model_dir}; MLflow run {run_id}")
    return 0


def _print_scores(name: str, scores: DevScores, detail: str) -> None:
    print(
        f"{name}: CV-tuned dev macro-F1 {scores.cv_macro_f1:.3f} "
        f"(micro {scores.cv_micro_f1:.3f}); {detail}"
    )
    for code, f1 in scores.cv_f1.items():
        print(f"  {code:<17} {f1:.3f}")


def _encoder_dir() -> Path:
    """The int8 e5 encoder, built (and downloaded) once."""
    return build_encoder(ENCODERS_DIR / f"e5-small-{E5_REVISION[:8]}-int8")


def _cpu_preview(one_thread: Scorer, texts: list[str]) -> float:
    """Reviews a second through a saved model limited to one CPU thread of this machine."""
    started = time.perf_counter()
    one_thread.predict_proba(texts)
    return len(texts) / (time.perf_counter() - started)


def _train_s1a(settings: LabSettings) -> int:
    require_tracking()  # before any work, so a missing setting costs nothing
    data = load_training_data(settings.data_dir, settings.gold_dir)
    encoder_dir = _encoder_dir()
    encoder = OnnxEncoder(encoder_dir)
    params = S1aParams()
    # The head learns from the int8 encoder's own embeddings, the ones Azure computes: trained on
    # 16-bit GPU embeddings instead, it lost ~0.075 dev macro-F1 when served through int8.
    log.info(
        "embedding %d silver reviews with e5 (cached after the first run)", len(data.train.ids)
    )
    started = time.monotonic()
    train_x = cached_embeddings(
        settings.data_dir / EMBEDDINGS_DIR,
        data.train.ids,
        clean_texts(data.train.texts),
        model_id=E5_MODEL_ID,
        embed=encoder.embed,
        variant=f"{E5_REVISION}|{QUERY_PREFIX}|{MAX_TOKENS}|int8-onnx",
    )
    embed_seconds = time.monotonic() - started
    started = time.monotonic()
    model = train_s1a(train_x, data.train.labels, params=params)
    train_seconds = time.monotonic() - started
    dev_x = encoder.embed(clean_texts(data.dev.texts))
    scores = score_on_dev(data.dev, model.predict_proba(dev_x))
    version = new_version()
    model_dir = MODELS_DIR / f"s1a-{version}"
    artifact = save_s1a(
        model, encoder_dir, model_dir, thresholds=scores.thresholds, version=version
    )
    one_thread = S1aScorer(
        OnnxEncoder(model_dir, threads=1), load_head(model_dir / HEAD_FILE), artifact
    )
    per_second = _cpu_preview(one_thread, data.train.texts[:CPU_PREVIEW_REVIEWS])
    size_mb = sum((model_dir / name).stat().st_size for name in artifact.files) / 1e6
    run_id = log_student(
        family="s1a",
        version=version,
        params={
            "encoder": E5_MODEL_ID,
            "encoder_revision": E5_REVISION,
            "encoder_quantization": "int8 dynamic (ONNX Runtime)",
            "prefix": QUERY_PREFIX,
            "max_tokens": MAX_TOKENS,
            "train_embeddings": "the int8 ONNX encoder, as served",
            "c": params.c,
        },
        data=data,
        scores=scores,
        model_dir=model_dir,
        metrics={
            "embed_seconds": embed_seconds,
            "train_seconds": train_seconds,
            "model_mb": size_mb,
            "cpu_preview_reviews_per_s": per_second,
        },
    )
    _print_scores(
        f"S1a {version}", scores, f"head trained in {train_seconds:.0f} s, {size_mb:.1f} MB"
    )
    print(
        f"CPU preview (this machine, 1 thread): {per_second:.0f} reviews a second; the gate is 70 "
        "on 1 vCPU, measured in a container in Task 14"
    )
    print(f"model: {model_dir}; MLflow run {run_id}")
    return 0


def _require_torch() -> None:
    """S1b trains in PyTorch: the local `gpu` group."""
    try:
        import torch
    except ImportError as error:
        raise SettingsError(
            "this command needs the gpu group: uv run --group gpu patchpulse-ml train s1b"
        ) from error
    if not torch.cuda.is_available():
        log.warning("CUDA isn't available: fine-tuning on the CPU will take many hours")


def _fine_tune(data: TrainingData, checkpoint_dir: Path, params: S1bParams) -> S1bTraining:
    from patchpulse.models.s1b_train import train_s1b  # the gpu group

    return train_s1b(
        data.train.texts,
        data.train.labels,
        data.holdout.texts,
        data.holdout.labels,
        checkpoint_dir,
        params=params,
    )


def _export(checkpoint: Path, out_dir: Path) -> Path:
    from patchpulse.models.s1b_train import export_s1b  # the gpu group

    return export_s1b(checkpoint, out_dir)


def _torch_dev_probs(checkpoint: Path, texts: list[str]) -> NDArray[np.float64]:
    from patchpulse.models.s1b_train import torch_logits  # the gpu group

    return sigmoid(torch_logits(checkpoint, texts))


def _train_s1b(settings: LabSettings, *, checkpoint: Path | None = None) -> int:
    require_tracking()  # before any work, so a missing setting costs nothing
    _require_torch()
    data = load_training_data(settings.data_dir, settings.gold_dir)
    version = new_version()
    if checkpoint is None:
        log.info(
            "fine-tuning e5 on %d silver reviews, early stopping on %d",
            len(data.train.ids),
            len(data.holdout.ids),
        )
        training = _fine_tune(data, CHECKPOINTS_DIR / f"s1b-{version}", S1bParams())
    else:
        training = read_training(checkpoint)  # a finished fine-tune: export and score it only
        log.info("reusing the fine-tune in %s", checkpoint)
    onnx_dir = _export(training.checkpoint, training.checkpoint / "onnx-int8")
    # Thresholds come from the int8 graph Azure would run; PyTorch's own dev score shows what
    # the quantization costs.
    classifier = OnnxClassifier(onnx_dir, S1B_FILE)
    scores = score_on_dev(data.dev, sigmoid(classifier.logits(clean_texts(data.dev.texts))))
    torch_scores = score_on_dev(data.dev, _torch_dev_probs(training.checkpoint, data.dev.texts))
    model_dir = MODELS_DIR / f"s1b-{version}"
    artifact = save_s1b(onnx_dir, model_dir, thresholds=scores.thresholds, version=version)
    one_thread = S1bScorer(OnnxClassifier(model_dir, S1B_FILE, threads=1), artifact)
    per_second = _cpu_preview(one_thread, data.train.texts[:CPU_PREVIEW_REVIEWS])
    size_mb = sum((model_dir / name).stat().st_size for name in artifact.files) / 1e6
    run_id = log_student(
        family="s1b",
        version=version,
        params={
            "encoder": E5_MODEL_ID,
            "encoder_revision": E5_REVISION,
            "quantization": "int8 dynamic (ONNX Runtime)",
            "prefix": QUERY_PREFIX,
            "holdout_n": len(data.holdout.ids),
            **asdict(training.params),
        },
        data=data,
        scores=scores,
        model_dir=model_dir,
        metrics={
            "train_seconds": training.seconds,
            "epochs_run": training.epochs_run,
            "best_epoch": training.best_epoch,
            **{f"holdout_loss.{i}": loss for i, loss in enumerate(training.holdout_losses, 1)},
            "dev_cv_macro_f1_torch": torch_scores.cv_macro_f1,
            "model_mb": size_mb,
            "cpu_preview_reviews_per_s": per_second,
        },
    )
    _print_scores(
        f"S1b {version}",
        scores,
        f"{training.epochs_run} epochs (best {training.best_epoch}) in "
        f"{training.seconds / 60:.0f} min, {size_mb:.1f} MB",
    )
    print(
        f"PyTorch's own dev macro-F1 (before int8): {torch_scores.cv_macro_f1:.3f}; holdout "
        f"losses {', '.join(f'{loss:.4f}' for loss in training.holdout_losses)}"
    )
    print(
        f"CPU preview (this machine, 1 thread): {per_second:.0f} reviews a second; the gate is 70 "
        "on 1 vCPU, measured in a container in Task 14"
    )
    print(f"model: {model_dir}; MLflow run {run_id}")
    return 0


def _latest_models() -> list[Path]:
    """The newest build of each student family in build/models (versions sort by time)."""
    latest: dict[str, Path] = {}
    for path in sorted(MODELS_DIR.glob("*-*")):
        family = path.name.partition("-")[0]
        if family in COST_ORDER and (path / MODEL_FILE).is_file():
            latest[family] = path
    return [latest[family] for family in COST_ORDER if family in latest]


def _benchmark(model_dirs: list[Path], texts: list[str]) -> dict[str, float]:
    return benchmark_cpu(model_dirs, texts)


def _gold_ids(settings: LabSettings) -> pd.DataFrame:
    return pd.read_csv(settings.gold_dir / SAMPLE_FILE)


def _evaluate_choose(settings: LabSettings, *, model_dirs: list[Path] | None) -> int:
    require_tracking()  # before any work, so a missing setting costs nothing
    dirs = model_dirs or _latest_models()
    if not dirs:
        print("patchpulse-ml: no models in build/models: run `train s0` first", file=sys.stderr)
        return 2
    data = load_training_data(settings.data_dir, settings.gold_dir)
    scorers = {path: load_scorer(path) for path in dirs}
    predicted = {
        path: cv_predictions(
            data.dev.labels, scorer.predict_proba(data.dev.texts), folds=CV_FOLDS, seed=CV_SEED
        )
        for path, scorer in scorers.items()
    }
    reviews = load_snapshot(settings.data_dir).reviews
    pool = reviews[~reviews["review_id"].isin(_gold_ids(settings)["review_id"])]
    sample = pool.sample(n=min(BENCHMARK_REVIEWS, len(pool)), random_state=EVAL_SEED)
    log.info(
        "timing %d models in a 1-CPU container (it installs the scoring group first)", len(dirs)
    )
    rates = _benchmark(dirs, [str(text) for text in sample["text"]])
    candidates = [
        Candidate(
            scorer.artifact.family,
            scorer.artifact.version,
            rates[scorer.artifact.family],
            predicted[path],
        )
        for path, scorer in scorers.items()
    ]
    choice = choose_champion(candidates, data.dev.labels, seed=EVAL_SEED)
    CHOICE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHOICE_PATH.write_text(choice.to_json(), encoding="utf-8", newline="\n")
    with tracking_run(
        EVAL_EXPERIMENT,
        params={f"{family}.version": version for family, version in choice.versions.items()},
        tags={"step": "choose", "champion": choice.champion or "none"},
        run_name="champion-choice",
    ):
        log_ids_hash("gold_dev", data.dev.ids)
        mlflow.log_metrics(
            {
                **{f"{f}.reviews_per_s": r for f, r in choice.reviews_per_s.items()},
                **{f"{f}.dev_cv_macro_f1": m for f, m in choice.dev_macro_f1.items()},
            }
        )
        log_report(CHOICE_PATH)
    for family in choice.dev_macro_f1:
        print(
            f"  {family:<4} {choice.reviews_per_s[family]:>9,.1f} reviews a second (1 CPU), dev "
            f"macro-F1 {choice.dev_macro_f1[family]:.3f}"
            f"{'' if family in choice.eligible else '  (too slow: ineligible)'}"
        )
    print(f"Champion: {choice.champion or 'none'} ({choice.reason}); {CHOICE_PATH}")
    return 0


def _evaluate_report(settings: LabSettings, *, force: bool) -> int:
    require_tracking()
    if not CHOICE_PATH.is_file():
        print(
            "patchpulse-ml: run `patchpulse-ml evaluate choose` first: the champion is chosen on "
            "dev before gold test is read",
            file=sys.stderr,
        )
        return 2
    if EVAL_REPORT_PATH.exists() and not force:
        print(
            f"patchpulse-ml: {EVAL_REPORT_PATH} exists: gold test is read once (design D6). "
            "--force regenerates the report for the same models",
            file=sys.stderr,
        )
        return 2
    choice = Choice.from_json(CHOICE_PATH.read_text(encoding="utf-8"))
    config = load_model_config()
    if config.teacher is None:
        print("patchpulse-ml: no teacher in config/models.yaml", file=sys.stderr)
        return 2
    sample = _gold_ids(settings)
    test_ids = sorted(int(i) for i in sample.loc[sample["split"] == "test", "review_id"])
    audit_ids = sorted(int(i) for i in pd.read_csv(settings.gold_dir / AUDIT_FILE)["review_id"])
    teacher_labels = LabelCache(settings.data_dir / TEACHER_CACHE_FILE).get_many(
        test_ids, model=config.candidates[config.teacher].cache_name, prompt_hash=prompt_hash()
    )
    if len(teacher_labels) < len(test_ids):
        print(
            f"patchpulse-ml: the teacher has labeled {len(teacher_labels)} of {len(test_ids)} "
            "gold test reviews: run `patchpulse-ml teacher label --set gold` first",
            file=sys.stderr,
        )
        return 2
    languages = dict(
        zip(sample["review_id"].astype(int), sample["language"].astype(str), strict=True)
    )
    claude = read_labels(settings.gold_dir / CLAUDE_LABELS_FILE)
    seif = read_labels(settings.gold_dir / SEIF_LABELS_FILE)
    sets = [
        GoldSet("test", test_ids, label_matrix(claude, test_ids), [languages[i] for i in test_ids]),
        GoldSet(
            "audit", audit_ids, label_matrix(seif, audit_ids), [languages[i] for i in audit_ids]
        ),
    ]
    data = load_training_data(settings.data_dir, settings.gold_dir)
    texts = load_snapshot(settings.data_dir).reviews.set_index("review_id")["text"]
    outputs: dict[str, dict[str, Outputs]] = {
        TEACHER: {s.name: Outputs(label_matrix(teacher_labels, s.ids)) for s in sets}
    }
    holdout: dict[str, NDArray[np.bool_]] = {}
    dirs = {
        family: MODELS_DIR / f"{family}-{version}" for family, version in choice.versions.items()
    }
    for family, model_dir in dirs.items():
        scorer = load_scorer(model_dir)
        log.info("scoring gold test, Seif's 150 and the silver holdout with %s", family)
        outputs[family] = {}
        for gold in sets:
            probs = scorer.predict_proba([str(texts[i]) for i in gold.ids])
            outputs[family][gold.name] = Outputs(scorer.labels(probs), probs)
        holdout[family] = scorer.labels(scorer.predict_proba(data.holdout.texts))
    result = evaluate_models(
        sets, outputs, holdout_teacher=data.holdout.labels, holdout_students=holdout, seed=EVAL_SEED
    )
    facts = {}
    for family, model_dir in dirs.items():
        test = result.score("test", family)
        artifact = record_metrics(
            model_dir,
            {
                "test_macro_f1": test.macro_f1,
                "test_macro_f1_low": test.macro_f1_ci[0],
                "test_macro_f1_high": test.macro_f1_ci[1],
                # Per aspect, so the site can mark the aspects detected less reliably.
                **{f"test_f1_{code}": f1 for code, f1 in test.f1.items()},
            },
        )
        size_mb = sum((model_dir / name).stat().st_size for name in artifact.files) / 1e6
        facts[family] = ModelFacts(
            family,
            artifact.version,
            size_mb,
            choice.reviews_per_s[family],
            choice.dev_macro_f1[family],
        )
    EVAL_REPORT_PATH.write_text(
        render_eval_report(result, choice, facts), encoding="utf-8", newline="\n"
    )
    with tracking_run(
        EVAL_EXPERIMENT,
        params={f"{family}.version": version for family, version in choice.versions.items()},
        tags={"step": "report", "champion": choice.champion or "none"},
        run_name="eval-report",
    ):
        log_ids_hash("gold_test", test_ids)
        log_ids_hash("seif_audit", audit_ids)
        mlflow.log_metrics(
            {f"{s.model}.{s.set_name}_macro_f1": s.macro_f1 for s in result.scores}
            | {f"{m}.holdout_agreement": a for m, a in result.agreement.items()}
        )
        log_report(EVAL_REPORT_PATH)
    for model in result.models:
        test, audit = result.score("test", model), result.score("audit", model)
        print(
            f"  {model:<8} gold test macro-F1 {test.macro_f1:.3f} "
            f"({test.macro_f1_ci[0]:.3f} to {test.macro_f1_ci[1]:.3f}); "
            f"Seif's 150 {audit.macro_f1:.3f}"
        )
    print(f"Champion: {choice.champion or 'none'}; report: {EVAL_REPORT_PATH}")
    return 0


def _champion_register(settings: LabSettings) -> int:
    require_tracking()
    if not CHOICE_PATH.is_file() or not EVAL_REPORT_PATH.is_file():
        print(
            "patchpulse-ml: run `evaluate choose` and `evaluate report` first: models are "
            "registered with their gold-test scores in model.json",
            file=sys.stderr,
        )
        return 2
    choice = Choice.from_json(CHOICE_PATH.read_text(encoding="utf-8"))
    if choice.champion is None:
        print(f"patchpulse-ml: no champion ({choice.reason})", file=sys.stderr)
        return 2
    others = [family for family in choice.dev_macro_f1 if family != choice.champion]
    challenger = max(others, key=lambda family: choice.dev_macro_f1[family]) if others else None
    # Only the two the aliases name are uploaded: the champion, and the runner-up on dev.
    dirs = {
        family: MODELS_DIR / f"{family}-{choice.versions[family]}"
        for family in (choice.champion, challenger)
        if family is not None
    }
    registered = register_models(dirs, champion=choice.champion, challenger=challenger)
    pin = pin_for(registered[choice.champion], dirs[choice.champion])
    write_pin(pin)
    for family, entry in registered.items():
        role = (
            "champion"
            if family == choice.champion
            else "challenger"
            if family == challenger
            else ""
        )
        print(
            f"  {family:<4} {entry.artifact.version}: aspect-classifier version "
            f"{entry.version} {role}"
        )
    print(f"pinned {pin.family} (version {pin.version}) in models/champion.yaml")
    return 0


def _champion_download(to: Path) -> int:
    require_tracking()
    if not PIN_FILE.is_file():
        print(f"patchpulse-ml: no {PIN_FILE.as_posix()}: run `champion register`", file=sys.stderr)
        return 2
    pin = read_pin()
    try:
        artifact = download_champion(pin, to)
    except ChampionError as error:
        print(f"patchpulse-ml: {error}", file=sys.stderr)
        return 1
    print(
        f"{pin.name} version {pin.version} ({artifact.family} {artifact.version}) "
        f"checked and written to {to.as_posix()}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="patchpulse-ml", description="PatchPulse Phase 2: local snapshot, labeling, training."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    pull_command = commands.add_parser(
        "pull", help="copy reviews from Azure SQL into the local snapshot (wakes the database)"
    )
    pull_command.add_argument("--yes", action="store_true", help="confirm waking Azure SQL")
    gold = commands.add_parser("gold", help="the gold set").add_subparsers(
        dest="gold_command", required=True
    )
    gold_sample = gold.add_parser("sample", help="draw the gold sample, splits and audit set")
    gold_sample.add_argument("--seed", type=int, required=True)
    gold_sample.add_argument("--force", action="store_true", help="overwrite an existing sample")
    gold_export = gold.add_parser("export-batch", help="write a labeling batch for a subagent")
    gold_export.add_argument("name", choices=BATCH_NAMES)
    gold_import = gold.add_parser("import", help="check and add a subagent's labels")
    gold_import.add_argument("name", choices=BATCH_NAMES)
    gold_import.add_argument("--file", type=Path, required=True)
    gold_import.add_argument("--guidelines", required=True, help="guidelines version, e.g. v2")
    gold.add_parser("agreement", help="compare Seif's blind audit with Claude's labels")
    teacher = commands.add_parser("teacher", help="the LLM teacher").add_subparsers(
        dest="teacher_command", required=True
    )
    bakeoff = teacher.add_parser("bakeoff", help="label gold dev with each candidate; pick one")
    bakeoff.add_argument(
        "--candidates", help="comma-separated names from config/models.yaml (default: all)"
    )
    label = teacher.add_parser("label", help="label a review set with the chosen teacher")
    label.add_argument("--set", dest="review_set", choices=["silver", "gold"], required=True)
    label.add_argument("--candidate", help="a name from config/models.yaml (default: teacher)")
    label.add_argument(
        "--keep-going",
        action="store_true",
        help="after an outage stop, wait 15 minutes and resume (for up to 24 hours)",
    )
    label.add_argument(
        "--workers",
        type=int,
        default=1,
        help="batches in flight at once; a self-hosted server only (default 1)",
    )
    silver = commands.add_parser("silver", help="the silver set").add_subparsers(
        dest="silver_command", required=True
    )
    silver_sample = silver.add_parser("sample", help="draw the silver sample and its holdout")
    silver_sample.add_argument("--seed", type=int, required=True)
    silver_sample.add_argument("--n", type=int, default=SILVER_SIZE)
    silver_sample.add_argument("--force", action="store_true", help="start the silver set over")
    train = commands.add_parser("train", help="train a student").add_subparsers(
        dest="train_command", required=True
    )
    train.add_parser("s0", help="TF-IDF + logistic regression on silver; thresholds on gold dev")
    train.add_parser("s1a", help="frozen e5 embeddings + logistic regression")
    s1b = train.add_parser("s1b", help="e5 fine-tuned on the GPU, int8 ONNX (gpu group)")
    s1b.add_argument(
        "--checkpoint", type=Path, help="reuse a finished fine-tune instead of training again"
    )
    evaluate = commands.add_parser("evaluate", help="champion and eval report").add_subparsers(
        dest="evaluate_command", required=True
    )
    choose = evaluate.add_parser("choose", help="pick the champion on gold dev (needs Docker)")
    choose.add_argument(
        "--models", nargs="+", type=Path, help="model directories (default: newest of each)"
    )
    report = evaluate.add_parser("report", help="score gold test once and write the report")
    report.add_argument("--force", action="store_true", help="regenerate it for the same models")
    champion = commands.add_parser("champion", help="the registered champion").add_subparsers(
        dest="champion_command", required=True
    )
    champion.add_parser("register", help="register the evaluated models and pin the champion")
    download = champion.add_parser("download", help="download the pinned champion and check it")
    download.add_argument("--to", type=Path, required=True, help="the directory to write it to")
    args = parser.parse_args(argv)

    # A character the console can't encode (Windows' cp1252, redirected to a file) prints as "?"
    # instead of killing the command: PyTorch's exporter and MLflow both print emoji.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    configure_logging()
    try:
        settings = get_lab_settings()
        if args.command == "pull":
            return _pull(settings, yes=args.yes)
        if args.command == "champion" and args.champion_command == "download":
            return _champion_download(args.to)
        if args.command == "champion":
            return _champion_register(settings)
        if args.command == "evaluate" and args.evaluate_command == "choose":
            return _evaluate_choose(settings, model_dirs=args.models)
        if args.command == "evaluate":
            return _evaluate_report(settings, force=args.force)
        if args.command == "train":
            if args.train_command == "s1b":
                return _train_s1b(settings, checkpoint=args.checkpoint)
            return _train_s1a(settings) if args.train_command == "s1a" else _train_s0(settings)
        if args.command == "silver":
            return _silver_sample(settings, seed=args.seed, n=args.n, force=args.force)
        if args.command == "teacher" and args.teacher_command == "label":
            return _teacher_label(
                settings,
                review_set=args.review_set,
                candidate=args.candidate,
                keep_going=args.keep_going,
                workers=args.workers,
            )
        if args.command == "teacher":
            names = args.candidates.split(",") if args.candidates else None
            return _teacher_bakeoff(settings, names=names)
        if args.gold_command == "sample":
            return _gold_sample(settings, seed=args.seed, force=args.force)
        if args.gold_command == "export-batch":
            return _gold_export(settings, name=args.name)
        if args.gold_command == "agreement":
            return _gold_agreement(settings)
        return _gold_import(settings, name=args.name, file=args.file, guidelines=args.guidelines)
    except (SettingsError, TrackingError) as error:
        print(f"patchpulse-ml: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
