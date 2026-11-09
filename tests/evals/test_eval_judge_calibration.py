"""The judge's calibration (Phase 6 plan Task 7, design D5): Seif's blind sample of 40 citation
and 20 figure items, his labels (no text), and Cohen's kappa per question with the pass rule."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from patchpulse.evals.judge import (
    CALIBRATION,
    KAPPA_PASS,
    JudgeItem,
    JudgeLabel,
    calibration_sample,
    judge_agreement,
    render_judge_agreement,
)
from patchpulse.evals.judge_audit import (
    LABEL_COLUMNS,
    JudgeAuditSession,
    export_audit,
    read_audit,
    read_labels,
)

SENTINEL = "SENTINEL-TEXT"


def items(citations: int, figures: int, arm: str = "main") -> list[JudgeItem]:
    made = [
        JudgeItem(f"{arm}/case{n}/c1/r{n}", "citation", f"claim {n}", f"review {n}", n % 2 == 0)
        for n in range(citations)
    ]
    made += [
        JudgeItem(f"{arm}/case{n}/c2/figure", "figure", f"claim {n}", '{"value":1}', True)
        for n in range(figures)
    ]
    return made


def test_the_sample_is_fixed_by_its_seed_and_split_40_20() -> None:
    pool = items(100, 50) + items(30, 30, arm="a2")
    pool.append(JudgeItem("main/blank/c1/r9", "citation", "claim", "", False))

    sample = calibration_sample(pool)

    assert CALIBRATION == {"citation": 40, "figure": 20}
    assert Counter(i.kind for i in sample.items) == {"citation": 40, "figure": 20}
    assert all(i.item_id.startswith("main/") and i.evidence for i in sample.items)
    assert sample.short == {}
    assert calibration_sample(list(reversed(pool))) == sample
    kinds = [i.kind for i in sample.items]
    assert kinds != sorted(kinds)  # shuffled together, not one kind after the other


def test_a_short_kind_takes_every_item_and_says_so() -> None:
    sample = calibration_sample(items(50, 7))

    assert Counter(i.kind for i in sample.items) == {"citation": 40, "figure": 7}
    assert sample.short == {"figure": 13}


def labels_for(pairs: list[tuple[bool, bool]], kind: str, *, start: int = 0) -> tuple[
    dict[str, JudgeLabel], dict[str, bool]
]:  # fmt: skip
    """Seif's labels and the judge's verdicts for `pairs` of (Seif, judge)."""
    labels, verdicts = {}, {}
    for n, (seif, judged) in enumerate(pairs, start=start):
        item_id = f"main/case{n}/c1/{kind}"
        labels[item_id] = JudgeLabel(kind, seif, False, "abc123")  # type: ignore[arg-type]
        verdicts[item_id] = judged
    return labels, verdicts


def test_kappa_report_per_question_with_the_pass_rule() -> None:
    same = [(True, True)] * 10 + [(False, False)] * 10
    half = [(True, True)] * 10 + [(True, False)] * 10 + [(False, False)] * 20  # kappa 0.5
    perfect_c, perfect_cv = labels_for(same, "citation")
    perfect_f, perfect_fv = labels_for(same, "figure", start=100)
    weak_c, weak_cv = labels_for(half, "citation", start=200)

    good = judge_agreement(perfect_c | perfect_f, perfect_cv | perfect_fv)
    bad = judge_agreement(weak_c | perfect_f, weak_cv | perfect_fv)

    assert KAPPA_PASS == 0.6
    assert good.rows["citation"].kappa == 1.0
    assert good.rows["figure"].kappa == 1.0
    assert good.rows["overall"].n == 40
    assert good.passed
    assert bad.rows["citation"].kappa == pytest.approx(0.5)
    assert bad.rows["citation"].lo is not None
    assert not bad.passed
    shown = render_judge_agreement(good), render_judge_agreement(bad)
    assert "at least 0.6 on each question" in shown[0]
    assert "| Citation support | 1.00 |" in shown[0]
    assert "Passed" in shown[0]
    assert "| Citation support | 0.50 |" in shown[1]
    assert "Failed" in shown[1]


def test_unsure_labels_are_left_out() -> None:
    labels, verdicts = labels_for([(True, True)] * 10 + [(False, False)] * 10, "citation")
    labels["main/x/c1/citation"] = JudgeLabel("citation", True, True, "abc123")
    verdicts["main/x/c1/citation"] = False
    labels["main/y/c1/citation"] = JudgeLabel("citation", True, False, "abc123")  # never judged

    agreement = judge_agreement(labels, verdicts)

    assert agreement.rows["citation"].kappa == 1.0
    assert agreement.rows["citation"].n == 20
    assert agreement.rows["citation"].unsure == 1
    assert agreement.unjudged == 1


def views_from(pool: list[JudgeItem], tmp_path: Path) -> Path:
    path = tmp_path / "data" / "local" / "evals" / "judge_audit.jsonl"
    export_audit(calibration_sample(pool), path)
    return path


def test_the_audit_session_saves_each_answer(tmp_path: Path) -> None:
    audit = views_from(items(3, 2), tmp_path)
    labels_path = tmp_path / "labels_seif.csv"
    views = read_audit(audit)
    session = JudgeAuditSession(views, labels_path, rubric_version="abc123")
    first = session.next_item()
    assert first is not None

    session.save(first.item_id, 1, unsure=False)
    session.save(first.item_id, 0, unsure=True)  # a correction: the latest row counts
    reopened = JudgeAuditSession(read_audit(audit), labels_path, rubric_version="abc123")

    assert reopened.progress() == (1, 5)
    assert reopened.next_item() == views[1]
    assert read_labels(labels_path)[first.item_id] == JudgeLabel(first.kind, False, True, "abc123")
    with pytest.raises(ValueError, match="not in this audit"):
        session.save("main/other/c1/r1", 1, unsure=False)
    with pytest.raises(ValueError, match="0 or 1"):
        session.save(first.item_id, 2, unsure=False)


def test_the_export_is_drawn_once(tmp_path: Path) -> None:
    path = views_from(items(3, 2), tmp_path)

    export_audit(calibration_sample(items(3, 2)), path)  # the same draw: a no-op
    with pytest.raises(ValueError, match="already drawn"):
        export_audit(calibration_sample(items(4, 2)), path)
    with pytest.raises(ValueError, match="data/local"):
        export_audit(calibration_sample(items(3, 2)), tmp_path / "judge_audit.jsonl")


def test_the_labels_file_holds_no_text(tmp_path: Path) -> None:
    pool = [
        JudgeItem("main/p0/c1/r101", "citation", f"{SENTINEL} claim", f"{SENTINEL} review", True),
        JudgeItem("main/p0/c2/figure", "figure", f"{SENTINEL} claim", '{"v":1}', False),
    ]
    audit = views_from(pool, tmp_path)
    labels_path = tmp_path / "labels_seif.csv"
    session = JudgeAuditSession(read_audit(audit), labels_path, rubric_version="abc123")

    while (view := session.next_item()) is not None:
        session.save(view.item_id, 1, unsure=False)

    written = labels_path.read_text(encoding="utf-8")
    assert SENTINEL not in written
    assert written.splitlines()[0] == ",".join(LABEL_COLUMNS)
    assert SENTINEL in audit.read_text(encoding="utf-8")  # the text stays in the local export
