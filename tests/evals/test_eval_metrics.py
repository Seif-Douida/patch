"""Metrics over the golden set (Phase 6 plan Task 5, design D4 and D8): rates with 95% bootstrap
intervals over incidents, alarms over controls only, claim rates resampling incidents, paired
differences for ablations, and A1's unsupported-claim rates."""

from __future__ import annotations

from collections.abc import Sequence

from eval_fixtures import claim, hypotheses, incident, record, report

from patchpulse.agent.report import Draft, WithheldClaim
from patchpulse.evals.incidents import Incident
from patchpulse.evals.metrics import a1, golden_metrics, paired_difference
from patchpulse.evals.records import StoredCase
from patchpulse.evals.scoring import CaseScore, score_case


def stored(case: str, result: object, *, draft: Draft | None = None) -> StoredCase:
    return StoredCase.from_record(
        record(result, draft=draft),  # type: ignore[arg-type]
        case=case,
        arm="main",
        data_version=60,
        misses=[],
        indexed_reviews=None,
    )


def patches(n: int) -> list[Incident]:
    return [incident("patch", incident_id=f"p{i}") for i in range(n)]


def test_accuracy_and_its_interval_are_deterministic() -> None:
    incidents = patches(4)
    runs = {
        "p0": stored("p0", report(hypotheses(("H1",)))),
        "p1": stored("p1", report(hypotheses(("H1",)))),
        "p2": stored("p2", report(hypotheses(("H1",)))),
        "p3": stored("p3", report(hypotheses(("H2",)))),
    }

    first = golden_metrics(incidents, runs)["root_cause_accuracy"]
    again = golden_metrics(incidents, runs)["root_cause_accuracy"]

    assert (first.value, first.n) == (0.75, 4)
    assert first == again
    assert first.lo is not None
    assert first.hi is not None
    assert first.lo <= 0.75 <= first.hi
    assert golden_metrics(incidents, runs)["root_cause_correct"].value == 3


def test_alarm_rates_count_controls_only() -> None:
    incidents = [
        incident("no_change", direction="none", incident_id="c0"),
        incident("no_change", direction="none", incident_id="c1"),
        *patches(2),
    ]
    runs = {
        "c0": stored("c0", report(hypotheses(("H2",)))),
        "c1": stored("c1", report(hypotheses(h5="supported"))),
        "p0": stored("p0", report(hypotheses(("H1",)))),
        "p1": stored("p1", report(hypotheses(("H1",)))),
    }

    metrics = golden_metrics(incidents, runs)

    assert (metrics["cause_alarm_rate"].value, metrics["cause_alarm_rate"].n) == (0.5, 2)
    assert metrics["cause_alarms"].value == 1
    assert metrics["change_alarm_rate"].value == 0.0


def drafted(shipped: int, withheld: int) -> tuple[object, Draft]:
    """A report with `shipped` claims shipped and `withheld` withheld by V2, and its draft."""
    kept = tuple(claim(f"c{i}", "Crashes rose.") for i in range(shipped))
    dropped = tuple(
        WithheldClaim(claim=claim(f"w{i}", "Reviews fell 77 points."), rule="V2")
        for i in range(withheld)
    )
    draft = Draft(findings=[], claims=[*kept, *(w.claim for w in dropped)])
    return report(hypotheses(("H1",)), claims=kept, withheld=dropped), draft


def test_claim_rates_resample_incidents_not_claims() -> None:
    many, many_draft = drafted(10, 0)
    none, none_draft = drafted(0, 2)
    runs = {
        "p0": stored("p0", many, draft=many_draft),
        "p1": stored("p1", none, draft=none_draft),
    }

    rate = golden_metrics(patches(2), runs)["pre_verifier_pass_rate"]

    assert rate.value is not None
    assert round(rate.value, 4) == round(10 / 12, 4)
    assert rate.n == 2
    # Resampling the two incidents gives 1.0 (p0 twice) and 0.0 (p1 twice) a quarter of the time
    # each; resampling the 12 claims would give a narrow interval around 0.83.
    assert (rate.lo, rate.hi) == (0.0, 1.0)
    withheld = golden_metrics(patches(2), runs)["withheld.V2"].value
    assert withheld is not None
    assert abs(withheld - 2 / 12) < 1e-6  # estimates are rounded to 6 places


def scores(correct: Sequence[bool]) -> dict[str, CaseScore]:
    incidents = patches(len(correct))
    return {
        i.id: score_case(i, record(report(hypotheses(("H1",) if ok else ("H2",)))))
        for i, ok in zip(incidents, correct, strict=True)
    }


def test_paired_difference_of_identical_arms_is_zero() -> None:
    same = scores([True, False, True, True, False])

    diff = paired_difference(same, same, lambda s: float(s.correct))

    assert (diff.value, diff.lo, diff.hi, diff.n) == (0.0, 0.0, 0.0, 5)


def test_paired_difference_follows_the_other_arm() -> None:
    base = scores([True, True, True, True])
    other = scores([True, False, True, False])

    diff = paired_difference(base, other, lambda s: float(s.correct))

    assert diff.value == -0.5


def test_a1_counts_withheld_and_judged_unsupported() -> None:
    shipped = (claim("c1", "Crashes rose."), claim("c2", "Servers failed."))
    withheld = (
        WithheldClaim(claim=claim("c3", "Reviews fell 77 points."), rule="V2"),
        WithheldClaim(claim=claim("c4", "Quoted text."), rule="V7"),
    )
    draft = Draft(findings=[], claims=[*shipped, *(w.claim for w in withheld)])
    runs = {
        "p0": stored("p0", report(hypotheses(), claims=shipped, withheld=withheld), draft=draft)
    }

    rates = a1(runs, {"p0": {"c1": True, "c2": False}})

    assert rates["unsupported_off"].value == 0.75  # 2 withheld and 1 judged unsupported, of 4
    assert rates["unsupported_on"].value == 0.5  # 1 judged unsupported, of 2 shipped
