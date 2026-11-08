"""The judge (Phase 6 plan Task 6, design D5): items from the stored runs, verdicts through the eval
cache, and claim-level support for A1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
import pytest
from eval_fixtures import aspects_call, claim, hypotheses
from eval_fixtures import report as make_report
from pydantic import SecretStr

from patchpulse.agent.graph import RunRecord
from patchpulse.agent.llm_cache import ReplayCache
from patchpulse.agent.report import Claim, EvidenceRef, WithheldClaim
from patchpulse.agent.router import ModelRouter
from patchpulse.agent.search import SNIPPET_CHARS
from patchpulse.agent.tools import Trace
from patchpulse.config import LabSettings
from patchpulse.evals.judge import (
    JUDGE_SCHEMA,
    RUBRIC_PATH,
    claim_support,
    judge,
    judge_items,
    judge_metrics,
    make_judge_router,
    rubric_version,
)
from patchpulse.evals.records import StoredCase
from patchpulse.labeling.gemini import GeminiClient
from patchpulse.labeling.llm import LLMResponse
from patchpulse.labeling.openai_compat import OpenAICompatClient
from patchpulse.labeling.teachers import load_model_config

REPO = Path(__file__).resolve().parents[2]
LONG = "the servers keep dropping me mid match " * 40  # over SNIPPET_CHARS
FIGURE_PATH = "aspects.platform_policy.negative.change"


def now() -> datetime:
    return datetime(2026, 11, 15, 12, tzinfo=UTC)


def quoting(claim_id: str, *review_ids: int) -> Claim:
    return claim(claim_id, "Players report dropped matches.").model_copy(
        update={"review_ids": list(review_ids)}
    )


def figure(claim_id: str, call_id: str = "r.1") -> Claim:
    return claim(claim_id, "Policy complaints rose 2 points.").model_copy(
        update={"value": 2.0, "unit": "pp", "evidence": EvidenceRef(call_id=call_id,
                                                                      json_path=FIGURE_PATH)}
    )  # fmt: skip


def stored(
    claims: tuple[Claim, ...], withheld: tuple[WithheldClaim, ...] = (), case: str = "p0"
) -> StoredCase:
    result = make_report(hypotheses(("H1",)), claims=claims, withheld=withheld)
    trace = Trace(calls=[aspects_call(2.0, 1.0)], retrieved={101: LONG, 102: LONG + "late"})
    record = RunRecord(result, None, result.hypotheses, None, None, trace, False)
    return StoredCase.from_record(
        record, case=case, arm="main", data_version=60, misses=[], indexed_reviews=None
    )


class FakeJudge:
    """Answers every request with `supported`; records the system prompts it saw."""

    def __init__(self, supported: bool = True) -> None:
        self.supported = supported
        self.systems: list[str] = []

    def generate(
        self,
        system: str,
        user: str,
        *,
        json_schema: Mapping[str, Any] | None,
        expected_output_tokens: int = 512,
        max_output_tokens: int = 4096,
    ) -> LLMResponse:
        self.systems.append(system)
        answer = {"supported": self.supported, "reason": "a short reason"}
        return LLMResponse(json.dumps(answer), 900, 20, "STOP")


def router(cache: ReplayCache, client: FakeJudge | None = None) -> ModelRouter:
    clients = {} if client is None else {"judge-model": client}
    return ModelRouter(["judge-model"], clients, cache, clock=now)


def test_items_cover_cited_reviews_and_cited_figures() -> None:
    case = stored((quoting("c1", 102, 101), figure("c2")))

    items = judge_items(case)

    assert [i.item_id for i in items] == ["main/p0/c1/r101", "main/p0/c1/r102", "main/p0/c2/figure"]
    assert [i.kind for i in items] == ["citation", "citation", "figure"]
    assert judge_items(case) == items
    for item in items[:2]:
        assert 0 < len(item.evidence) <= SNIPPET_CHARS
        assert item.claim_text == "Players report dropped matches."
    shown = json.loads(items[2].evidence)
    assert shown["tool"] == "aspect_breakdown"
    assert shown["path"] == FIGURE_PATH
    assert shown["value"] == 2.0
    assert shown["holder"] == {"change": 2.0}
    assert all(i.shipped for i in items)


def test_withheld_claims_are_judged_and_reworded_ones_on_their_shipped_text() -> None:
    reworded = quoting("c1", 101).model_copy(update={"text": "Hedged text.", "reworded": True})
    withheld = WithheldClaim(claim=quoting("c2", 102), rule="V7")

    items = judge_items(stored((reworded,), (withheld,)))

    assert [(i.item_id, i.shipped) for i in items] == [
        ("main/p0/c1/r101", True),
        ("main/p0/c2/r102", False),
    ]
    assert items[0].claim_text == "Hedged text."


def test_a_cited_review_or_figure_the_run_never_saw_has_no_evidence() -> None:
    items = judge_items(stored((quoting("c1", 999), figure("c2", call_id="r.9"))))

    assert [(i.item_id, i.evidence) for i in items] == [
        ("main/p0/c1/r999", ""),
        ("main/p0/c2/figure", ""),
    ]


def test_a_claim_is_unsupported_when_any_item_fails() -> None:
    case = stored((quoting("c1", 101, 102), figure("c2"), quoting("c3", 101)))
    verdicts = {
        "main/p0/c1/r101": True,
        "main/p0/c1/r102": False,
        "main/p0/c2/figure": True,
    }

    support = claim_support(case, verdicts)

    assert support == {"c1": False, "c2": True, "c3": None}


def test_judge_answers_are_cached_and_replayed(tmp_path: Path) -> None:
    items = judge_items(stored((quoting("c1", 101),)))
    client = FakeJudge(supported=False)
    path = tmp_path / "data" / "local" / "evals" / "answers.sqlite"

    first = judge(items, router(ReplayCache(path, "fill"), client))
    second = judge(items, router(ReplayCache(path, "fill"), client))
    replayed = judge(items, router(ReplayCache(path, "replay")))

    assert len(client.systems) == 1
    assert first.verdicts == second.verdicts == replayed.verdicts == {"main/p0/c1/r101": False}
    assert (first.asked, first.cached) == (1, 0)
    assert (second.asked, second.cached) == (0, 1)


def test_an_item_without_evidence_is_unsupported_without_asking(tmp_path: Path) -> None:
    items = judge_items(stored((quoting("c1", 999),)))
    client = FakeJudge(supported=True)

    run = judge(items, router(ReplayCache(tmp_path / "a.sqlite", "fill"), client))

    assert run.verdicts == {"main/p0/c1/r999": False}
    assert (run.blank, run.asked) == (1, 0)
    assert client.systems == []


def test_an_unanswered_item_stays_unjudged(tmp_path: Path) -> None:
    items = judge_items(stored((quoting("c1", 101),)))

    run = judge(items, router(ReplayCache(tmp_path / "a.sqlite", "replay")))

    assert run.verdicts == {}
    assert run.unanswered == 1


def test_the_rubric_is_the_system_prompt(tmp_path: Path) -> None:
    items = judge_items(stored((quoting("c1", 101), figure("c2"))))
    client = FakeJudge()
    rubric = REPO / RUBRIC_PATH

    judge(items, router(ReplayCache(tmp_path / "a.sqlite", "fill"), client), rubric_path=rubric)

    text = rubric.read_text(encoding="utf-8")
    assert client.systems == [text, text]
    assert "Question 1" in text
    assert "Question 2" in text
    assert rubric_version(rubric) == hashlib.sha256(rubric.read_bytes()).hexdigest()[:12]
    assert JUDGE_SCHEMA["required"] == ["supported", "reason"]


def test_judge_metrics_count_shipped_items_per_kind() -> None:
    withheld = WithheldClaim(claim=quoting("c3", 102), rule="V7")
    case = stored((quoting("c1", 101, 102), figure("c2")), (withheld,))
    verdicts = {
        "main/p0/c1/r101": True,
        "main/p0/c1/r102": False,
        "main/p0/c2/figure": True,
        "main/p0/c3/r102": False,
    }

    metrics = judge_metrics({"p0": case}, verdicts)

    assert metrics["citation_support"].value == 0.5
    assert metrics["citation_support"].n == 1
    assert metrics["figure_faithfulness"].value == 1.0


def test_the_judge_router_tries_the_server_then_google(tmp_path: Path) -> None:
    settings = LabSettings(
        data_dir=tmp_path,
        llm_server_url="http://judge.invalid/v1",
        llm_server_key=SecretStr("server-key"),
        gemini_api_key=SecretStr("gemini-key"),
    )
    cache = ReplayCache(tmp_path / "a.sqlite", "fill")
    models = load_model_config(REPO / "config" / "models.yaml")

    with httpx2.Client() as http:
        built = make_judge_router(models, settings, http, cache, clock=now)

    assert built.chain == ["selfhosted-gemma-4-31b", "gemma-4-31b"]
    assert isinstance(built.clients["selfhosted-gemma-4-31b"], OpenAICompatClient)
    assert isinstance(built.clients["gemma-4-31b"], GeminiClient)


def test_the_judge_router_skips_a_model_without_its_settings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = LabSettings(data_dir=tmp_path, gemini_api_key=SecretStr("gemini-key"),
                           llm_server_url=None, llm_server_key=None)  # fmt: skip
    cache = ReplayCache(tmp_path / "a.sqlite", "fill")
    models = load_model_config(REPO / "config" / "models.yaml")
    nothing = LabSettings(data_dir=tmp_path, gemini_api_key=None, llm_server_url=None,
                          llm_server_key=None)  # fmt: skip

    with httpx2.Client() as http:
        built = make_judge_router(models, settings, http, cache, clock=now)
        with pytest.raises(ValueError, match="PP_GEMINI_API_KEY"):
            make_judge_router(models, nothing, http, cache, clock=now)

    assert built.chain == ["gemma-4-31b"]
    assert "PP_LLM_SERVER_URL" in caplog.text
