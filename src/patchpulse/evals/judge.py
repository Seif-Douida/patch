"""The judge (Phase 6 design D5, spec §11.4): Gemma 4 31B checks the claims of the stored runs.

Two questions, written in evals/judge/rubric.md (the first 12 hex digits of its SHA-256 are its
version):

1. Citation support: does a review the claim cites, as the agent saw it, support the claim? One
   item per claim and cited review.
2. Figure faithfulness: does the claim describe the tool result it cites correctly: what it counts,
   its period, share or count, its direction? One item per claim citing a figure.

A claim is supported when every item of it is. Shipped and withheld claims are both judged, a
reworded claim on the text it shipped with. A cited review the run never retrieved, or a figure the
run doesn't hold, gives an item with no evidence: unsupported, and nothing is asked.

Answers go through the eval cache: a rerun asks only what's new, and a changed rubric asks again.
The cache keeps the judge's reasons, which can quote reviews; nothing else does. The live run is
Seif's command: Claude never sends review text to the self-hosted server (ADR-020).

Calibration (spec §11.4): Seif labels a blind sample of the main arm's items, 40 citation pairs
and 20 figure claims, and Cohen's kappa against the judge's verdicts must reach 0.6 on each
question before the judge counts anywhere. The rule is fixed before any label exists.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx2
import numpy as np

from patchpulse.agent.llm_cache import ReplayCache
from patchpulse.agent.report import Claim, EvidenceRef
from patchpulse.agent.router import (
    AGENT_ATTEMPTS,
    AGENT_TIMEOUT_S,
    AllModelsDownError,
    ModelClient,
    ModelRouter,
)
from patchpulse.agent.untrusted import DATA_NOT_INSTRUCTIONS, review_snippet
from patchpulse.agent.verifier import extract
from patchpulse.config import LabSettings
from patchpulse.evals.metrics import EVAL_SEED, SAMPLES, Estimate, ratio
from patchpulse.evals.records import StoredCall, StoredCase
from patchpulse.labeling.gemini import GeminiClient
from patchpulse.labeling.limits import USAGE_FILE, RequestLimiter
from patchpulse.labeling.openai_compat import OpenAICompatClient
from patchpulse.labeling.teachers import TeacherModels
from patchpulse.models.metrics import cohen_kappa

log = logging.getLogger("patchpulse.evals")

JUDGE_STEP = "judge"
RUBRIC_PATH = Path("evals") / "judge" / "rubric.md"
JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["supported", "reason"],
}
JUDGE_OUTPUT_TOKENS = 256  # a verdict and a sentence; Google's per-minute token budget reads it
PROGRESS_EVERY = 25
ItemKind = Literal["citation", "figure"]
KINDS: tuple[ItemKind, ...] = ("citation", "figure")
METRIC_KEYS: dict[ItemKind, str] = {"citation": "citation_support", "figure": "figure_faithfulness"}
QUESTIONS: dict[ItemKind, str] = {
    "citation": "Question 1: citation support.",
    "figure": "Question 2: figure faithfulness.",
}
# Calibration (design D5): Seif's blind sample per question, its seed, and the pass rule.
CALIBRATION: dict[ItemKind, int] = {"citation": 40, "figure": 20}
CALIBRATION_SEED = 20261108
KAPPA_PASS = 0.6
CALIBRATED_ARM = "main"
AGREEMENT_PATH = Path("reports") / "evals" / "judge-agreement.md"
QUESTION_NAMES = {
    "citation": "Citation support",
    "figure": "Figure faithfulness",
    "overall": "Both questions",
}


def rubric_version(path: Path = RUBRIC_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


@dataclass(frozen=True)
class JudgeItem:
    item_id: str  # <arm>/<case>/<claim_id>/r<review_id>, or .../figure
    kind: ItemKind
    claim_text: str
    evidence: str  # the review's snippet or the figure as compact JSON; "" if the run never saw it
    shipped: bool


@dataclass(frozen=True)
class JudgeRun:
    verdicts: dict[str, bool]  # item id -> supported
    asked: int = 0
    cached: int = 0
    blank: int = 0  # no evidence: unsupported without asking
    unanswered: int = 0  # every model failed, or replay had no answer


def _parent(path: str) -> str:
    """The path of the object holding the value at `path` ("" for the whole result)."""
    cut = max(path.rfind("."), path.rfind("["))
    return path[:cut] if cut > 0 else ""


def _figure(ref: EvidenceRef, calls: Mapping[str, StoredCall]) -> str:
    """The cited value with what names it: the tool, its arguments (the periods), the path, and
    the object holding the value, whose other fields say what it measures."""
    call = calls.get(ref.call_id)
    if call is None:
        return ""
    try:
        value = extract(call.result, ref.json_path)
        holder = extract(call.result, _parent(ref.json_path))
    except (KeyError, IndexError, ValueError):
        return ""
    shown = {
        "tool": call.tool,
        "args": {name: arg for name, arg in call.args.items() if name != "appid"},
        "path": ref.json_path,
        "value": value,
        "holder": holder,
    }
    return json.dumps(shown, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _claims(stored: StoredCase) -> list[tuple[Claim, bool]]:
    """The model's claims with whether they shipped; none for a refusal or a basic report."""
    report = stored.report
    if report is None or report.basic:
        return []
    return [(c, True) for c in report.claims] + [(w.claim, False) for w in report.withheld_claims]


def _items(stored: StoredCase, claim: Claim, shipped: bool) -> list[JudgeItem]:
    prefix = f"{stored.arm}/{stored.case}/{claim.claim_id}"
    items = []
    for review_id in sorted(claim.supporting_reviews):
        text = stored.trace.retrieved.get(review_id)
        evidence = "" if text is None else review_snippet(text)
        items.append(JudgeItem(f"{prefix}/r{review_id}", "citation", claim.text, evidence, shipped))
    if claim.evidence is not None:
        calls = {c.call_id: c for c in stored.trace.calls}
        evidence = _figure(claim.evidence, calls)
        items.append(JudgeItem(f"{prefix}/figure", "figure", claim.text, evidence, shipped))
    return items


def judge_items(stored: StoredCase) -> list[JudgeItem]:
    """Every item of a stored run, shipped claims first, in a stable order."""
    return [item for claim, shipped in _claims(stored) for item in _items(stored, claim, shipped)]


def _escape(text: str) -> str:
    """Whitespace collapsed and `<` escaped: the text can't open or close a tag."""
    return " ".join(text.split()).replace("<", "&lt;")


def judge_prompt(item: JudgeItem) -> str:
    """The user prompt: the question, the claim and its evidence, as data."""
    tag = "review" if item.kind == "citation" else "figure"
    return (
        f"{QUESTIONS[item.kind]}\n{DATA_NOT_INSTRUCTIONS}\n"
        f"<claim>{_escape(item.claim_text)}</claim>\n<{tag}>{_escape(item.evidence)}</{tag}>"
    )


def judge(
    items: Sequence[JudgeItem], router: ModelRouter, *, rubric_path: Path = RUBRIC_PATH
) -> JudgeRun:
    """Each item's verdict, with the rubric as the system prompt. An item no model answers stays
    unjudged; the run goes on (the router rests a failing model)."""
    system = rubric_path.read_text(encoding="utf-8")
    verdicts: dict[str, bool] = {}
    asked = cached = blank = unanswered = 0
    for n, item in enumerate(items, start=1):
        if n % PROGRESS_EVERY == 0:
            log.info("judge: item %d of %d", n, len(items))
        if not item.evidence:
            verdicts[item.item_id] = False
            blank += 1
            continue
        try:
            routed = router.generate(
                JUDGE_STEP,
                system,
                judge_prompt(item),
                JUDGE_SCHEMA,
                expected_output_tokens=JUDGE_OUTPUT_TOKENS,
            )
        except AllModelsDownError:
            unanswered += 1
            continue
        verdicts[item.item_id] = bool(routed.data["supported"])
        if routed.cached:
            cached += 1
        else:
            asked += 1
    return JudgeRun(verdicts, asked, cached, blank, unanswered)


def _support(verdicts: Sequence[bool | None]) -> bool | None:
    if any(v is False for v in verdicts):
        return False
    if not verdicts or any(v is None for v in verdicts):
        return None
    return True


def claim_support(stored: StoredCase, verdicts: Mapping[str, bool]) -> dict[str, bool | None]:
    """Per claim id: False when any of its items was judged unsupported, None when an item is
    unjudged (or it has none), else True."""
    return {
        claim.claim_id: _support([verdicts.get(i.item_id) for i in _items(stored, claim, shipped)])
        for claim, shipped in _claims(stored)
    }


def _tally(stored: StoredCase, verdicts: Mapping[str, bool], kind: ItemKind) -> tuple[int, int]:
    """The shipped items of a kind judged supported, and judged at all."""
    judged = [
        verdicts[i.item_id]
        for i in judge_items(stored)
        if i.kind == kind and i.shipped and i.item_id in verdicts
    ]
    return sum(judged), len(judged)


def citation_share(stored: StoredCase, verdicts: Mapping[str, bool]) -> float | None:
    """The run's judged citations of shipped claims that support them (an ablation's measure)."""
    supported, judged = _tally(stored, verdicts, "citation")
    return supported / judged if judged else None


def judge_metrics(
    stored: Mapping[str, StoredCase], verdicts: Mapping[str, bool]
) -> dict[str, Estimate]:
    """Citation support (cited review pairs) and figure faithfulness (cited figures) of the
    shipped claims, judged items only, pooled per incident."""
    out = {}
    for kind, key in METRIC_KEYS.items():
        tallies = [_tally(case, verdicts, kind) for _, case in sorted(stored.items())]
        out[key] = ratio([t[0] for t in tallies], [t[1] for t in tallies])
    return out


def judge_chain(models: TeacherModels) -> list[str]:
    if models.judge is None:
        raise ValueError("config/models.yaml has no judge section")
    return list(models.judge.chain)


def make_judge_router(
    models: TeacherModels,
    settings: LabSettings,
    http: httpx2.Client,
    cache: ReplayCache,
    *,
    clock: Callable[[], datetime],
    sleep: Callable[[float], None] = time.sleep,
) -> ModelRouter:
    """The judge's chain: the self-hosted server (PP_LLM_SERVER_URL, PP_LLM_SERVER_KEY), then
    Google's free Gemma 4 31B with its limiter. A model whose settings are missing is skipped
    with a warning; with none left, ValueError names what's missing."""
    clients: dict[str, ModelClient] = {}
    missing: list[str] = []
    for name in judge_chain(models):
        config = models.candidates[name]
        if config.provider == "openai_compat":
            if settings.llm_server_url is None or settings.llm_server_key is None:
                missing.append("PP_LLM_SERVER_URL and PP_LLM_SERVER_KEY")
                log.warning("judge: skipping %s (no PP_LLM_SERVER_URL or PP_LLM_SERVER_KEY)", name)
                continue
            key = settings.llm_server_key.get_secret_value()
            clients[name] = OpenAICompatClient(key, settings.llm_server_url, http, config,
                                               sleep=sleep)  # fmt: skip
            continue
        if settings.gemini_api_key is None:
            missing.append("PP_GEMINI_API_KEY")
            log.warning("judge: skipping %s (no PP_GEMINI_API_KEY)", name)
            continue
        limiter = RequestLimiter(
            config, state_path=settings.data_dir / USAGE_FILE, clock=clock, sleep=sleep
        )
        clients[name] = GeminiClient(
            settings.gemini_api_key.get_secret_value(),
            http,
            limiter,
            config,
            sleep=sleep,
            max_attempts=AGENT_ATTEMPTS,
            retry_errors=False,
            timeout_s=AGENT_TIMEOUT_S,
        )
    if not clients:
        raise ValueError(f"the judge needs {' or '.join(missing)}")
    chain = [name for name in judge_chain(models) if name in clients]
    return ModelRouter(chain, clients, cache, clock=clock)


@dataclass(frozen=True)
class CalibrationSample:
    items: tuple[JudgeItem, ...]
    short: dict[str, int]  # kind -> how many items the arm lacked


def calibration_sample(items: Sequence[JudgeItem]) -> CalibrationSample:
    """The main arm's items with evidence, CALIBRATION of each kind drawn with the seed (whatever
    the input order), then shuffled together. A kind with too few gives all of them and records
    the shortfall."""
    rng = np.random.default_rng(CALIBRATION_SEED)
    chosen: list[JudgeItem] = []
    short: dict[str, int] = {}
    for kind, size in CALIBRATION.items():
        pool = sorted(
            (
                i
                for i in items
                if i.kind == kind and i.evidence and i.item_id.startswith(f"{CALIBRATED_ARM}/")
            ),
            key=lambda i: i.item_id,
        )
        if len(pool) <= size:
            chosen += pool
            if len(pool) < size:
                short[kind] = size - len(pool)
            continue
        chosen += [pool[int(k)] for k in rng.choice(len(pool), size=size, replace=False)]
    order = rng.permutation(len(chosen))
    return CalibrationSample(tuple(chosen[int(k)] for k in order), short)


@dataclass(frozen=True)
class JudgeLabel:
    """Seif's label for one item: ids and flags only (the labels file is committed)."""

    kind: ItemKind
    label: bool  # supported, or described correctly
    unsure: bool
    rubric_version: str  # the rubric he read while labeling


@dataclass(frozen=True)
class KappaRow:
    kappa: float | None
    lo: float | None
    hi: float | None
    agreement: float | None  # the share of items where Seif and the judge agree
    n: int  # items compared: labeled, sure and judged
    unsure: int  # left out


@dataclass(frozen=True)
class JudgeAgreement:
    rows: dict[str, KappaRow]  # citation, figure, overall
    passed: bool  # kappa of at least KAPPA_PASS on each question
    unjudged: int  # sure labels with no verdict, left out
    rubric_versions: tuple[str, ...]


def _kappa_row(pairs: Sequence[tuple[bool, bool]], unsure: int) -> KappaRow:
    """Kappa with a percentile bootstrap interval over items; a resample where either side gave
    one answer throughout has no kappa and is left out of the percentiles."""
    if not pairs:
        return KappaRow(None, None, None, None, 0, unsure)
    seif = np.asarray([a for a, _ in pairs], dtype=bool)
    judged = np.asarray([b for _, b in pairs], dtype=bool)
    draws = np.random.default_rng(EVAL_SEED).integers(0, len(pairs), size=(SAMPLES, len(pairs)))
    values = [cohen_kappa(seif[ix], judged[ix]) for ix in draws]
    finite = np.array([np.nan if v is None else v for v in values], dtype=float)
    lo: float | None = None
    hi: float | None = None
    if not np.isnan(finite).all():
        low, high = np.nanpercentile(finite, [2.5, 97.5])
        lo, hi = float(low), float(high)
    return KappaRow(
        kappa=cohen_kappa(seif, judged),
        lo=lo,
        hi=hi,
        agreement=float(np.mean(seif == judged)),
        n=len(pairs),
        unsure=unsure,
    )


def judge_agreement(
    labels: Mapping[str, JudgeLabel], verdicts: Mapping[str, bool]
) -> JudgeAgreement:
    """Cohen's kappa between Seif and the judge per question and overall, with its bootstrap
    interval. Unsure labels and items the judge hasn't answered are left out and counted."""
    rows = {}
    for name in (*KINDS, "overall"):
        chosen = [(i, lab) for i, lab in sorted(labels.items()) if name in ("overall", lab.kind)]
        pairs = [(lab.label, verdicts[i]) for i, lab in chosen if not lab.unsure and i in verdicts]
        rows[name] = _kappa_row(pairs, sum(lab.unsure for _, lab in chosen))
    kappas = [rows[kind].kappa for kind in KINDS]
    passed = all(k is not None and k >= KAPPA_PASS for k in kappas)
    unjudged = sum(1 for i, lab in labels.items() if not lab.unsure and i not in verdicts)
    versions = tuple(sorted({lab.rubric_version for lab in labels.values()}))
    return JudgeAgreement(rows, passed, unjudged, versions)


def agreement_estimates(agreement: JudgeAgreement) -> dict[str, Estimate]:
    """The kappas as the eval report shows them."""
    return {
        f"kappa.{name}": Estimate(row.kappa, row.lo, row.hi, row.n)
        for name, row in agreement.rows.items()
    }


def _two(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def render_judge_agreement(agreement: JudgeAgreement) -> str:
    """reports/evals/judge-agreement.md: numbers only, with the pass rule."""
    lines = [
        "# The judge's agreement with Seif",
        "",
        "Seif labeled a blind random sample of the main arm's judge items (design D5) without "
        "seeing the judge's verdicts, the arm or whether the claim shipped. Unsure labels are "
        "left out.",
        "",
        "| Question | Kappa | 95% interval | Agreement | Items | Unsure (left out) |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in agreement.rows.items():
        interval = "-" if row.lo is None or row.hi is None else f"{row.lo:.2f} to {row.hi:.2f}"
        share = "-" if row.agreement is None else f"{100 * row.agreement:.1f}%"
        lines.append(
            f"| {QUESTION_NAMES[name]} | {_two(row.kappa)} | {interval} | {share} | {row.n} | "
            f"{row.unsure} |"
        )
    verdict = (
        "Passed."
        if agreement.passed
        else "Failed: the rubric is revised and the items judged again before the judge counts "
        "anywhere, and the eval report says so."
    )
    lines += [
        "",
        f"Pass rule (fixed before labeling): Cohen's kappa of at least {KAPPA_PASS} on each "
        f"question. {verdict}",
        "",
        f"Labeled under rubric {', '.join(agreement.rubric_versions) or '-'}. Labeled items "
        f"with no verdict yet (left out): {agreement.unjudged}.",
    ]
    return "\n".join(lines) + "\n"
