"""Seif's audit against Claude's gold labels, judged by the rule fixed before any comparison
(design D2): an aspect is assessable with >= 8 positives for either annotator; the gold set
passes if macro kappa over assessable aspects >= 0.70 and every assessable aspect >= 0.60. And
Seif's blind re-label of 30 of his 150, reported next to it with no pass rule (ADR-019)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from patchpulse.labeling.agreement import agreement, render_agreement, self_agreement
from patchpulse.labeling.gold import format_aspects
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.cli import main

IDS = list(range(1, 151))
# Ids far from every count the report prints, so the test can tell an id leaked.
RELABELED = [7_000_001 + i for i in range(30)]


def labels(positive: dict[str, set[int]]) -> dict[int, frozenset[str]]:
    return {i: frozenset(code for code, ids in positive.items() if i in ids) for i in IDS}


def common() -> dict[str, set[int]]:
    """Every aspect assessable: 30 positives each, on overlapping blocks of reviews."""
    return {code: set(range(1 + 10 * n, 31 + 10 * n)) for n, code in enumerate(ASPECTS)}


def disagree(base: dict[str, set[int]], code: str, flips: int) -> dict[str, set[int]]:
    """Seif drops `flips` of Claude's positives for one aspect and adds as many elsewhere."""
    changed = {k: set(v) for k, v in base.items()}
    positives = sorted(changed[code])
    negatives = [i for i in IDS if i not in changed[code]]
    changed[code] = (changed[code] - set(positives[:flips])) | set(negatives[:flips])
    return changed


def test_identical_labels_give_kappa_one() -> None:
    report = agreement(labels(common()), labels(common()), IDS)

    assert all(row.kappa == 1.0 for row in report.aspects)
    assert report.macro_kappa == 1.0
    assert report.passed


def test_all_aspects_pass() -> None:
    report = agreement(labels(common()), labels(disagree(common(), "content", 3)), IDS)

    assert report.passed
    assert report.failing == []


def test_one_assessable_aspect_below_0_6_fails_and_is_named() -> None:
    report = agreement(labels(common()), labels(disagree(common(), "monetization", 12)), IDS)

    row = next(r for r in report.aspects if r.code == "monetization")
    assert row.kappa is not None
    assert row.kappa < 0.6
    assert not report.passed
    assert report.failing == ["monetization"]


def test_low_macro_kappa_fails() -> None:
    seif = common()
    for code in ASPECTS:
        seif = disagree(seif, code, 8)  # every aspect around 0.62: each passes, the mean doesn't

    report = agreement(labels(common()), labels(seif), IDS)

    assert all(r.kappa is not None and r.kappa >= 0.6 for r in report.aspects)
    assert report.macro_kappa is not None
    assert report.macro_kappa < 0.70
    assert not report.passed


def test_rare_aspect_is_reported_too_rare_and_left_out_of_the_macro() -> None:
    claude = common()
    claude["platform_policy"] = {1, 2, 3}
    seif = {k: set(v) for k, v in claude.items()}
    seif["platform_policy"] = {4, 5}  # total disagreement, but on 3 and 2 positives

    report = agreement(labels(claude), labels(seif), IDS)

    row = next(r for r in report.aspects if r.code == "platform_policy")
    assert not row.assessable
    assert report.passed
    assert report.macro_kappa == 1.0  # the rare aspect isn't averaged in


def test_report_contains_no_ids_or_text() -> None:
    report = agreement(labels(common()), labels(disagree(common(), "content", 3)), IDS)

    rendered = render_agreement(report)

    assert "performance" in rendered
    assert "0.70" in rendered  # the rule is stated
    for review_id in (137, 149):
        assert str(review_id) not in rendered


def audit_answers() -> dict[int, frozenset[str]]:
    """Seif's first answers on the 30: performance on 10, content on 15, story on 2."""
    answers: dict[int, frozenset[str]] = {}
    for n, review_id in enumerate(RELABELED, start=1):
        codes = {"performance"} if n <= 10 else set()
        if 6 <= n <= 20:
            codes.add("content")
        if n in (25, 26):
            codes.add("story_world")
        answers[review_id] = frozenset(codes)
    return answers


def relabel_answers() -> dict[int, frozenset[str]]:
    """Two weeks later: content moves from review 20 to 21, and story is dropped on 26."""
    answers = dict(audit_answers())
    answers[RELABELED[19]] = answers[RELABELED[19]] - {"content"}
    answers[RELABELED[20]] = answers[RELABELED[20]] | {"content"}
    answers[RELABELED[25]] = answers[RELABELED[25]] - {"story_world"}
    return answers


def test_seifs_relabel_is_compared_with_his_audit_aspect_by_aspect() -> None:
    reliability = self_agreement(audit_answers(), relabel_answers(), RELABELED)

    rows = {row.code: row for row in reliability.aspects}
    assert list(rows) == list(ASPECTS)
    assert (rows["performance"].positives_first, rows["performance"].kappa) == (10, 1.0)
    content = rows["content"]
    assert (content.positives_first, content.positives_second) == (15, 15)
    assert content.kappa == pytest.approx(0.8667, abs=1e-4)  # 28 of 30 agree, chance 0.5
    assert rows["story_world"].kappa == pytest.approx(0.6512, abs=1e-4)
    assert rows["monetization"].kappa is None  # no positives either time: undefined
    # 297 of 300 review-aspect pairs agree; 27 positives against 26.
    assert reliability.pooled_kappa == pytest.approx(0.9379, abs=1e-4)
    assert (reliability.n_reviews, reliability.identical) == (30, 27)


def test_self_agreement_refuses_a_review_answered_only_once() -> None:
    second = relabel_answers()
    del second[RELABELED[0]]

    with pytest.raises(ValueError, match="1 re-labeled review"):
        self_agreement(audit_answers(), second, RELABELED)


def test_the_report_gives_the_relabel_next_to_the_audit() -> None:
    report = agreement(labels(common()), labels(common()), IDS)
    reliability = self_agreement(audit_answers(), relabel_answers(), RELABELED)

    rendered = render_agreement(report, reliability=reliability)

    assert "## Seif's re-label" in rendered
    assert "30 of the 150" in rendered
    assert "pairs: kappa 0.94" in rendered
    assert "27 of 30 reviews" in rendered
    assert "| performance | 10 | 10 | 1.00 |" in rendered
    assert "| content | 15 | 15 | 0.87 |" in rendered
    assert "| monetization | 0 | 0 | n/a |" in rendered
    assert not any(str(review_id) in rendered for review_id in RELABELED)


def test_the_report_marks_the_relabel_pending_until_it_is_done() -> None:
    report = agreement(labels(common()), labels(common()), IDS)

    rendered = render_agreement(report, relabeled=12)

    assert "## Seif's re-label" in rendered
    assert "Pending" in rendered
    assert "12 of 30 so far" in rendered


def write_labels(path: Path, answers: dict[int, frozenset[str]]) -> None:
    pd.DataFrame(
        {"review_id": list(answers), "aspects": [format_aspects(a) for a in answers.values()]}
    ).to_csv(path, index=False)


@pytest.mark.parametrize(("answered", "expected"), [(30, "27 of 30 reviews"), (12, "12 of 30")])
def test_gold_agreement_reads_the_relabel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, answered: int, expected: str
) -> None:
    for name in list(os.environ):
        if name.startswith("PP_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    gold = tmp_path / "data" / "gold"
    gold.mkdir(parents=True)
    audited = {**labels(common()), **audit_answers()}
    pd.DataFrame({"review_id": list(audited)}).to_csv(gold / "audit.csv", index=False)
    write_labels(gold / "labels_claude.csv", audited)
    write_labels(gold / "labels_seif.csv", audited)
    second = relabel_answers()
    write_labels(gold / "relabel_seif.csv", {i: second[i] for i in RELABELED[:answered]})

    assert main(["gold", "agreement"]) == 0

    rendered = (tmp_path / "reports" / "aspects" / "gold-agreement.md").read_text(encoding="utf-8")
    assert "## Seif's re-label" in rendered
    assert expected in rendered
