"""The smoke suite (Phase 6 design D11): what `eval-gate` runs from the eval bundle alone.

- The golden incidents and the injection cases, replayed. The recorded answers and tool results
  answer every model call, review search and patch-notes fetch; nothing reaches a model, Steam or
  the cluster. A case whose run asks anything the bundle didn't record is named with its misses: a
  PR that changes what the model is asked must re-record those cases and pin a new bundle.
- The PR's verifier on every recorded draft: what it would ship, reword or withhold. A claim's
  tolerance comes from code, not from the model, so each claim is parsed again by this code.
- The judge's recorded verdicts on the replayed claims, reported and never gated.
- The retrieval suite on the pool, in a Qdrant container, and the champion student on gold test.

The metrics hold numbers, case and claim ids, categories, rule ids and step names: no review,
claim or summary text.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import MISSING, asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from patchpulse.agent.bundle import Bundle, load_bundle
from patchpulse.agent.graph import ReportRequest, run_recorded
from patchpulse.agent.llm_cache import ReplayCache
from patchpulse.agent.report import Claim
from patchpulse.agent.router import ModelRouter
from patchpulse.agent.tools import Tools, Trace
from patchpulse.agent.verifier import verify_claims
from patchpulse.evals.arms import ARMS, Arm
from patchpulse.evals.classification_suite import ClassificationScores, run_classification
from patchpulse.evals.incidents import Incident, load_incidents
from patchpulse.evals.injection import (
    InjectionCase,
    assign_hosts,
    check_injection,
    injection_run_cases,
    load_injection_cases,
)
from patchpulse.evals.judge import judge, judge_chain, judge_items, judge_metrics
from patchpulse.evals.records import CACHE_FILE, StoredCase, load_arm
from patchpulse.evals.retrieval_suite import (
    POOL_COLLECTION,
    QUERIES_PATH,
    RecordedQueryEmbedder,
    RetrievalScores,
    grades_of,
    load_pool,
    run_retrieval,
)
from patchpulse.evals.runner import Case, case_deps
from patchpulse.evals.scoring import score_case
from patchpulse.labeling.teachers import load_model_config
from patchpulse.retrieval.judging import JUDGMENTS_PATH
from patchpulse.retrieval.qdrant import make_client
from patchpulse.retrieval.queries import load_queries

SMOKE_CHAIN = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
LOCAL_QDRANT = "http://localhost:6333"
CHAMPION_DIR = Path("build") / "champion"
MANIFEST = "manifest.json"
DATA_FILE = "data.json.gz"
SMOKE_SCHEMA = 1  # the metrics file's layout; a baseline of another one still compares


@dataclass(frozen=True)
class SmokeMetrics:
    bundle_version: int
    bundle_commit: str  # the commit the bundle was built from
    commit: str  # the commit smoked
    # A count is None only in a baseline written before it existed: it isn't compared then.
    root_cause_correct: int | None
    cause_alarms: int | None
    change_alarms: int | None
    refusals: int | None
    predicted: dict[str, str]  # golden case -> predicted category
    unreplayed: dict[str, list[str]]  # case -> its misses ("model:<step>", "tool:<name>")
    injection_before: int | None  # successes over the replayed, exposed cases
    injection_after: int | None
    injection_exposed: int | None
    judge: dict[str, float | None]  # replayed verdicts, reported only
    verifier: dict[str, dict[str, str]]  # case -> claim -> shipped, reworded or withheld:Vn
    # case -> the SHA-256 of its recorded run: the verifier is compared only on runs both
    # bundles recorded alike (claim ids are per draft).
    recorded: dict[str, str]
    ndcg_at_10: float | None
    recall_at_20: float | None
    macro_f1: float | None
    schema: int = SMOKE_SCHEMA

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, document: dict[str, Any]) -> SmokeMetrics:
        """Metrics of any schema: unknown keys are ignored; a missing one is empty (a mapping)
        or None, and a missing metric isn't compared."""
        values: dict[str, Any] = {}
        for field in fields(cls):
            if field.name in document:
                values[field.name] = document[field.name]
            elif field.default is not MISSING:
                values[field.name] = field.default
            else:
                values[field.name] = {} if str(field.type).startswith("dict") else None
        return cls(**values)


def verifier_decisions(stored: StoredCase, bundle: Bundle) -> dict[str, str]:
    """This code's verifier on a recorded draft: per claim id, `shipped`, `reworded` or
    `withheld:Vn`. None for a refusal or a basic report (no model draft)."""
    if stored.draft is None:
        return {}
    record = stored.record()
    # The model never writes a tolerance (the draft schema refuses one): it's derived from the
    # unit by this code, so a recorded claim is parsed again.
    claims = [
        Claim.model_validate(c.model_dump(exclude={"tolerance"})) for c in stored.draft.claims
    ]
    tools = Tools(bundle, Trace(run=record.trace.run))  # V2's re-runs, never recorded
    bomb = any(h.code == "H3" and h.status == "supported" for h in record.hypotheses)
    verified = verify_claims(claims, record.trace, tools, bundle.causal, review_bomb=bomb)
    decisions = {c.claim_id: "reworded" if c.reworded else "shipped" for c in verified.claims}
    decisions |= {w.claim.claim_id: f"withheld:{w.rule}" for w in verified.withheld}
    return dict(sorted(decisions.items()))


def _replay(
    cases: Sequence[tuple[Case, Arm]],
    *,
    bundle: Bundle,
    cache: ReplayCache,
    clock: Callable[[], datetime],
) -> dict[str, StoredCase]:
    """Each case run again from the cache only, with its misses."""
    router = ModelRouter(list(SMOKE_CHAIN), {}, cache, clock=clock)
    replayed = {}
    for case, arm in cases:
        deps = case_deps(
            case, arm, bundle=bundle, router=router, cache=cache, search=None, steam=None
        )
        cache.misses.clear()
        record = run_recorded(ReportRequest.for_question(case.question, appid=case.appid), deps)
        replayed[case.case_id] = StoredCase.from_record(
            record,
            case=case.case_id,
            arm=arm.name,
            data_version=bundle.data_version,
            misses=list(dict.fromkeys(cache.misses)),
            indexed_reviews=None,
        )
    return replayed


def pool_suite(bundle_dir: Path, qdrant_url: str) -> RetrievalScores:
    """The pool loaded into a fresh `eval_pool` collection, then Phase 3's queries on it."""
    client = make_client(qdrant_url, None)
    try:
        if client.collection_exists(POOL_COLLECTION):
            client.delete_collection(POOL_COLLECTION)
        load_pool(client, pd.read_parquet(bundle_dir / "pool.parquet"))
        vectors = json.loads((bundle_dir / "query_vectors.json").read_text(encoding="utf-8"))
        judgments = pd.read_csv(JUDGMENTS_PATH, dtype={"query_id": str})
        return run_retrieval(
            client, load_queries(QUERIES_PATH), grades_of(judgments), RecordedQueryEmbedder(vectors)
        )
    finally:
        client.close()


def gold_suite(bundle_dir: Path, champion_dir: Path) -> ClassificationScores:
    return run_classification(champion_dir, pd.read_parquet(bundle_dir / "gold_test.parquet"))


def run_smoke(
    bundle_dir: Path,
    *,
    commit: str,
    qdrant_url: str = LOCAL_QDRANT,
    champion_dir: Path = CHAMPION_DIR,
    incidents: Sequence[Incident] | None = None,
    injection: Sequence[InjectionCase] | None = None,
    retrieval: Callable[[Path, str], RetrievalScores] = pool_suite,
    classification: Callable[[Path, Path], ClassificationScores] = gold_suite,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> SmokeMetrics:
    """Every check of the opened bundle in `bundle_dir`."""
    manifest = json.loads((bundle_dir / MANIFEST).read_text(encoding="utf-8"))
    bundle = load_bundle((bundle_dir / DATA_FILE).read_bytes())
    golden = list(incidents) if incidents is not None else load_incidents()
    planted = list(injection) if injection is not None else load_injection_cases()
    runs = {arm: load_arm(arm, root=bundle_dir) for arm in ("main", "injection")}
    hosts = assign_hosts(planted, runs["main"]) if planted else {}
    cases = [(Case(i.id, i.appid, i.question), ARMS["main"]) for i in golden]
    cases += [(c, ARMS["injection"]) for c in injection_run_cases(planted, hosts, golden)]
    cache = ReplayCache(bundle_dir / CACHE_FILE, "replay")
    replayed = _replay(cases, bundle=bundle, cache=cache, clock=clock)

    scores = [score_case(i, replayed[i.id].record()) for i in golden]
    outcomes = [check_injection(c, replayed[c.id]) for c in planted if not replayed[c.id].misses]
    exposed = [o for o in outcomes if o.exposed]
    clean = {i.id: replayed[i.id] for i in golden if not replayed[i.id].misses}
    items = [item for case in clean.values() for item in judge_items(case)]
    judge_router = ModelRouter(judge_chain(load_model_config()), {}, cache, clock=clock)
    verdicts = judge(items, judge_router).verdicts if items else {}
    judged = {key: estimate.value for key, estimate in judge_metrics(clean, verdicts).items()}
    verifier = {
        case_id: verifier_decisions(stored, bundle)
        for stored_runs in runs.values()
        for case_id, stored in sorted(stored_runs.items())
    }
    files = manifest["files"]
    recorded = {
        case_id: str(files[f"runs/{arm}/{case_id}.json"])
        for arm, stored_runs in runs.items()
        for case_id in sorted(stored_runs)
    }
    pool = retrieval(bundle_dir, qdrant_url)
    gold = classification(bundle_dir, champion_dir)
    return SmokeMetrics(
        bundle_version=int(manifest["bundle_version"]),
        bundle_commit=str(manifest["commit"]),
        commit=commit,
        root_cause_correct=sum(s.correct for s in scores),
        cause_alarms=sum(s.cause_alarm for s in scores if s.control),
        change_alarms=sum(s.change_alarm for s in scores if s.control),
        refusals=sum(s.predicted == "refused" for s in scores),
        predicted={s.incident_id: s.predicted for s in scores},
        unreplayed={case_id: s.misses for case_id, s in replayed.items() if s.misses},
        injection_before=sum(o.before for o in exposed),
        injection_after=sum(o.after for o in exposed),
        injection_exposed=len(exposed),
        judge=judged,
        verifier=verifier,
        recorded=recorded,
        ndcg_at_10=pool.ndcg_at_10,
        recall_at_20=pool.recall_at_20,
        macro_f1=gold.macro_f1,
    )
