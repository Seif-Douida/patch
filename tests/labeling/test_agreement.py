"""Seif's audit against Claude's gold labels, judged by the rule fixed before any comparison
(design D2): an aspect is assessable with >= 8 positives for either annotator; the gold set
passes if macro kappa over assessable aspects >= 0.70 and every assessable aspect >= 0.60."""

from __future__ import annotations

from patchpulse.labeling.agreement import agreement, render_agreement
from patchpulse.labeling.taxonomy import ASPECTS

IDS = list(range(1, 151))


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
