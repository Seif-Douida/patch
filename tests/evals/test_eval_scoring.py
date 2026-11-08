"""Scoring a report against an incident (Phase 6 plan Task 3, design D3-D4), on hand-made reports
and traces: the category the rules' statuses name, the alarms on controls, the window's overlap,
the direction, and the claim, tool-call and patch-blame counts."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from eval_fixtures import (
    WINDOW,
    aspects_call,
    claim,
    hypotheses,
    incident,
    record,
    report,
)

from patchpulse.agent.graph import Refusal
from patchpulse.agent.report import (
    Draft,
    WithheldClaim,
)
from patchpulse.agent.tools import ToolCall, Trace
from patchpulse.evals.scoring import (
    ClaimStats,
    ToolCallStats,
    claim_stats,
    patch_blame,
    predicted_category,
    score_case,
    tool_call_stats,
    window_iou,
)


def test_refusal_is_refused() -> None:
    assert predicted_category(Refusal("no"), Trace()) == "refused"


@pytest.mark.parametrize(
    ("supported", "expected"),
    [
        (("H1",), "patch"),
        (("H2",), "pricing_sale"),
        (("H4",), "audience_shift"),
        (("H2", "H3"), "off_topic"),
        (("H3", "H4"), "off_topic"),
        (("H2", "H4"), "pricing_sale"),
    ],
)
def test_first_supported_explanation_names_the_cause(
    supported: tuple[str, ...], expected: str
) -> None:
    trace = Trace(calls=[aspects_call(1.0, 4.0)])

    assert predicted_category(report(hypotheses(supported)), trace) == expected


def test_a_review_bomb_with_a_larger_policy_rise_is_platform_policy() -> None:
    bomb = report(hypotheses(("H3",)))

    assert predicted_category(bomb, Trace(calls=[aspects_call(6.0, 4.0)])) == "platform_policy"
    assert predicted_category(bomb, Trace(calls=[aspects_call(4.0, 4.0)])) == "off_topic"
    assert predicted_category(bomb, Trace(calls=[aspects_call(None, 4.0)])) == "off_topic"


def test_h5_supported_is_no_change_and_nothing_supported_is_undetermined() -> None:
    assert predicted_category(report(hypotheses(h5="supported")), Trace()) == "no_change"
    assert predicted_category(report(hypotheses(h5="rejected")), Trace()) == "undetermined"
    assert predicted_category(report(hypotheses()), Trace()) == "undetermined"


def test_lenient_accepts_a_secondary_cause() -> None:
    sale = record(report(hypotheses(("H2",))))

    score = score_case(incident("patch", secondary=("pricing_sale",)), sale)

    assert (score.predicted, score.correct, score.lenient) == ("pricing_sale", False, True)


def test_alarms_on_controls() -> None:
    control = incident("no_change", direction="none")

    sale = score_case(control, record(report(hypotheses(("H2",)))))
    change = score_case(control, record(report(hypotheses(h5="rejected"))))
    quiet = score_case(control, record(report(hypotheses(h5="supported"))))
    patch_case = score_case(incident("patch"), record(report(hypotheses(("H1",)))))

    assert (sale.control, sale.cause_alarm, sale.change_alarm) == (True, True, False)
    assert (change.cause_alarm, change.change_alarm) == (False, True)
    assert (quiet.correct, quiet.cause_alarm, quiet.change_alarm) == (True, False, False)
    assert (patch_case.control, patch_case.cause_alarm) == (False, False)


def test_window_iou() -> None:
    first = date(2025, 9, 1)

    assert (
        window_iou(
            (first, first + timedelta(days=6)),
            (first + timedelta(days=3), first + timedelta(days=9)),
        )
        == 0.4
    )
    assert (
        window_iou(
            (first, first + timedelta(days=6)),
            (first + timedelta(days=7), first + timedelta(days=9)),
        )
        == 0.0
    )
    assert window_iou(WINDOW, WINDOW) == 1.0
    refused = score_case(incident(), record(Refusal("no"), window=None))
    assert (refused.window_iou, refused.decided, refused.predicted) == (0.0, 0, "refused")


def test_direction() -> None:
    falling = score_case(
        incident(direction="negative"), record(report(hypotheses(total_change=-3.1)))
    )
    unknown = score_case(
        incident(direction="negative"), record(report(hypotheses(total_change=None)))
    )
    rising = score_case(
        incident(direction="positive"), record(report(hypotheses(total_change=-3.1)))
    )
    control = score_case(incident("no_change", direction="none"), record(report(hypotheses())))

    assert (falling.direction_ok, unknown.direction_ok, rising.direction_ok) == (True, False, False)
    assert control.direction_ok is None


def test_claim_and_tool_call_stats() -> None:
    shipped = (claim("c1", "Crashes rose."), claim("c2", "The share fell 3 points.", reworded=True))
    withheld = (
        WithheldClaim(claim=claim("c3", "Reviews fell 77 points."), rule="V2"),
        WithheldClaim(claim=claim("c4", "Quoted text."), rule="V7"),
        WithheldClaim(claim=claim("c5", "Wrong number."), rule="V2"),
    )
    drafted = Draft(findings=[], claims=[*shipped, *(w.claim for w in withheld)])
    search = {"appid": 1, "query": "crashes"}
    calls = (
        aspects_call(0.0, 0.0),
        ToolCall("r.2", "search_reviews", search, {"hits": []}),
        ToolCall("r.3", "search_reviews", search, {"hits": []}),
        ToolCall("r.4", "get_patch_notes", {"patch_id": "p9"}, {"error": "no patch p9"}),
        ToolCall("r.5", "search_reviews", {"appid": 1, "query": "x"},
                 {"error": "search_reviews wasn't recorded for this run"}),
    )  # fmt: skip

    stats = claim_stats(
        record(report(hypotheses(), claims=shipped, withheld=withheld), draft=drafted)
    )

    assert stats == ClaimStats(
        drafted=5, shipped=2, reworded=1, withheld_by_rule={"V2": 2, "V7": 1}
    )
    assert claim_stats(record(Refusal("no"))) is None
    assert claim_stats(record(report(hypotheses(), basic=True))) is None
    assert tool_call_stats(Trace(calls=list(calls))) == ToolCallStats(
        planned=4, errors=1, repeats=1
    )


def test_patch_blame_needs_causal_wording_and_a_patch() -> None:
    blamed = claim("c1", "The update caused the drop.", claim_type="causal")
    lexical = claim("c2", "Reviews fell because of the hotfix.")
    plain = claim("c3", "The update shipped on 2 Sep.")
    sale = claim("c4", "The sale caused the rise.", claim_type="causal")
    drafted = Draft(findings=[], claims=[blamed, lexical, plain, sale])
    shipped = report(
        hypotheses(),
        claims=(plain,),
        withheld=(WithheldClaim(claim=blamed, rule="V4"), WithheldClaim(claim=lexical, rule="V5"),
                  WithheldClaim(claim=sale, rule="V4")),
    )  # fmt: skip

    assert patch_blame(record(shipped, draft=drafted)) == (2, 0)
    assert patch_blame(record(Refusal("no"))) is None
