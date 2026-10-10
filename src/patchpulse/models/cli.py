"""`patchpulse-ml`: Phase 2's local commands (snapshot, labeling, training), run on Seif's machine.

    pull         Copy reviews from Azure SQL into data/local (wakes the database; needs --yes).
    gold sample        Draw the gold set and Seif's audit set from the snapshot (data/gold/).
    gold export-batch  Write a labeling batch (pilot, batch-1..4) for a subagent (data/local/).
    gold import        Check a subagent's labels against its batch and add them to the gold set.
    gold agreement     Compare Seif's blind audit with Claude's labels (reports/aspects/).
    teacher bakeoff    Label gold dev with each teacher candidate and pick one (reports/aspects/,
                       MLflow). Needs PP_GEMINI_API_KEY and the MLFLOW_TRACKING_* settings.
    silver sample      Draw the ~30k silver reviews and their holdout (data/local/silver/).
    teacher label      Label the silver set with the chosen teacher; resumable after a quota stop.

Exit codes: 0 success, 1 the command failed, 2 bad configuration or a refused action.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx2
import pandas as pd

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
from patchpulse.labeling.silver import (
    IDS_FILE,
    SILVER_DIR,
    SILVER_SIZE,
    label_silver,
    prevalence_report,
    sample_silver,
    split_holdout,
    write_silver_ids,
)
from patchpulse.labeling.teacher import review_inputs
from patchpulse.labeling.teachers import ModelConfig, load_model_config
from patchpulse.models.snapshot import explain_pull_error, load_snapshot, pull
from patchpulse.models.tracking import TrackingError, require_tracking

log = logging.getLogger("patchpulse.ml")

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
        "pulled %d rows; the snapshot has %d reviews (up to run %d) in %s",
        result.reviews_new,
        result.reviews_total,
        result.max_run_id,
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
    if settings.gemini_api_key is None:
        raise SettingsError("missing or invalid settings: PP_GEMINI_API_KEY")
    require_tracking()  # before any request, so a missing setting costs nothing
    config = load_model_config()
    unknown = sorted(set(names or []) - set(config.candidates))
    if unknown:
        print(f"patchpulse-ml: unknown candidates {unknown}", file=sys.stderr)
        return 2
    candidates = {name: config.candidates[name] for name in names or config.candidates}
    sample = pd.read_csv(settings.gold_dir / SAMPLE_FILE)
    dev = sorted(int(i) for i in sample.loc[sample["split"] == "dev", "review_id"])
    reviews = review_inputs(load_snapshot(settings.data_dir), dev)
    gold = read_labels(settings.gold_dir / CLAUDE_LABELS_FILE)
    key = settings.gemini_api_key.get_secret_value()
    http = httpx2.Client()

    def make_client(_: str, model: ModelConfig) -> GeminiClient:
        limiter = RequestLimiter(model, state_path=settings.data_dir / USAGE_FILE)
        return GeminiClient(key, http, limiter, model)

    result = run_bakeoff(
        candidates,
        reviews,
        gold,
        make_client=make_client,
        cache=LabelCache(settings.data_dir / TEACHER_CACHE_FILE),
    )
    quota = [c.name for c in result.candidates if c.stopped_for_quota]
    outage = [c.name for c in result.candidates if c.stopped_for_outage]
    if quota or outage:
        reasons = [f"the daily quota ran out for {', '.join(quota)}"] if quota else []
        reasons += [f"Google stayed unavailable for {', '.join(outage)}"] if outage else []
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


def _teacher_label(settings: LabSettings, *, candidate: str | None) -> int:
    if settings.gemini_api_key is None:
        raise SettingsError("missing or invalid settings: PP_GEMINI_API_KEY")
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
    limiter = RequestLimiter(model, state_path=settings.data_dir / USAGE_FILE)
    client = GeminiClient(
        settings.gemini_api_key.get_secret_value(), httpx2.Client(), limiter, model
    )
    snapshot = load_snapshot(settings.data_dir)
    cache = LabelCache(settings.data_dir / TEACHER_CACHE_FILE)
    run = label_silver(client, cache, snapshot, settings.data_dir, model=model.id)
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
        log.warning("Google stayed unavailable; run the same command again later")
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
    label.add_argument("--set", dest="review_set", choices=["silver"], required=True)
    label.add_argument("--candidate", help="a name from config/models.yaml (default: teacher)")
    silver = commands.add_parser("silver", help="the silver set").add_subparsers(
        dest="silver_command", required=True
    )
    silver_sample = silver.add_parser("sample", help="draw the silver sample and its holdout")
    silver_sample.add_argument("--seed", type=int, required=True)
    silver_sample.add_argument("--n", type=int, default=SILVER_SIZE)
    silver_sample.add_argument("--force", action="store_true", help="start the silver set over")
    args = parser.parse_args(argv)

    configure_logging()
    try:
        settings = get_lab_settings()
        if args.command == "pull":
            return _pull(settings, yes=args.yes)
        if args.command == "silver":
            return _silver_sample(settings, seed=args.seed, n=args.n, force=args.force)
        if args.command == "teacher" and args.teacher_command == "label":
            return _teacher_label(settings, candidate=args.candidate)
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
