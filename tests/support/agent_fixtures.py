"""A small data bundle for the agent's tests (Phase 5), built in code.

One tracked game with 71 days around a patch bundle (p1 and p2, three days apart) and an event
post, plus a second game with a single day. Reviews turn more negative after the patch; the mix
shifts towards new players; a discount runs over the patch days. Estimates for bundle p1 cover the
positive share (both methods, synthetic control without a 14-day estimate), the established
sample, volume and one aspect. `sample_report` is an invented bundle report with a quote span,
review ids and a withheld claim, as the store keeps it.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta

from patchpulse.agent.bundle import (
    AspectRow,
    BucketRow,
    Bundle,
    DayRow,
    GameData,
    PatchRow,
)
from patchpulse.agent.prompts import PROMPT_VERSION
from patchpulse.agent.report import Report
from patchpulse.causal.config import CausalConfig
from patchpulse.causal.store import StoredEstimate
from patchpulse.labeling.taxonomy import ASPECTS

GAME = 553850
OTHER_GAME = 949230
FIRST = date(2026, 8, 1)
PATCH_DAY = date(2026, 9, 10)
LAST = date(2026, 10, 10)
DATA_VERSION = 52
NO_SYNTH_14 = "the comparison games are checked over the 14 days after the patch"
TOO_FEW = "too few comparison games without their own patch (3 of 5 needed)"


def _days() -> Iterable[date]:
    day = FIRST
    while day <= LAST:
        yield day
        day += timedelta(days=1)


def _counts(day: date) -> tuple[int, int, int]:
    """(positives, reviews, new players) for a day: steadier before the patch."""
    i = (day - FIRST).days
    if day < PATCH_DAY:
        totals = 20 + i % 5
        return round(totals * 0.8), totals, totals // 4
    totals = 30 + i % 7
    return round(totals * 0.6), totals, totals // 2


def estimate(
    horizon: int,
    method: str,
    effect: float | None,
    *,
    outcome: str = "positive_share",
    sample: str = "all",
    reason: str | None = None,
    flags: frozenset[str] = frozenset(),
) -> StoredEstimate:
    synth = method == "synth"
    return StoredEstimate(
        bundle_id="p1",
        patch_ids=("p1", "p2"),
        appid=GAME,
        outcome=outcome,
        sample=sample,
        horizon=horizon,
        method=method,
        run_id=DATA_VERSION - 1,
        effect=effect,
        lo90=None if effect is None else effect - 3.0999999046325684,
        hi90=None if effect is None else effect + 2.5,
        p_value=0.04761904761904762 if synth and effect is not None else None,
        pre_fit_rmse=1.2345678806304932 if synth and effect is not None else None,
        donors=((1086940, 0.6000000238418579), (892970, 0.4000000059604645)) if synth else (),
        flags=flags,
        reason=reason,
        fingerprint=f"{horizon}{method}{outcome}{sample}".ljust(64, "0")[:64],
        version="e2",
        computed_at=datetime(2026, 10, 10, 2, 40, tzinfo=UTC),
    )


ESTIMATES = (
    estimate(7, "its", -11.5),
    estimate(7, "synth", -12.25, flags=frozenset({"volume_spike"})),
    estimate(14, "its", -10.0),
    estimate(14, "synth", None, reason=NO_SYNTH_14),
    estimate(7, "its", -6.5, sample="established"),
    estimate(7, "its", 48.0, outcome="volume"),
    estimate(7, "synth", None, outcome="volume", reason=TOO_FEW),
    estimate(7, "its", 9.75, outcome="aspect:performance"),
)


def game_data() -> GameData:
    days, buckets, aspects = [], [], []
    for day in _days():
        positives, totals, new = _counts(day)
        discount = (
            None if day < date(2026, 9, 1) else 25 if PATCH_DAY <= day < date(2026, 9, 15) else 0
        )
        days.append(DayRow(day, positives, totals, sale=day.day == 20, discount=discount))
        new_positives = min(new, positives) // 2
        buckets += [
            BucketRow(day, "lt_2h", new_positives, new),
            BucketRow(day, "10h_50h", positives - new_positives - 1, totals - new - 2),
            BucketRow(day, "unknown", 1, 2),
        ]
        negatives = totals - positives
        for k, code in enumerate(ASPECTS):
            after = 2 if day >= PATCH_DAY and code == "performance" else 1
            aspects.append(
                AspectRow(
                    day,
                    code,
                    neg_mentions=min(negatives, (k % 3 + 1) * after),
                    neg_scored=negatives,
                    pos_mentions=k % 2,
                    pos_scored=positives,
                )
            )
    patches = (
        PatchRow("p0", "p0", datetime(2026, 8, 5, 17, tzinfo=UTC), date(2026, 8, 5),
                 "Patch 1.0.1", "patch", "p0"),
        PatchRow("p1", "p1", datetime(2026, 9, 10, 16, tzinfo=UTC), PATCH_DAY,
                 "Patch 1.1.0", "patch", "p1"),
        PatchRow("e1", "e1", datetime(2026, 9, 11, 9, tzinfo=UTC), date(2026, 9, 11),
                 "Autumn event", "event", None),
        PatchRow("p2", "p2", datetime(2026, 9, 13, 12, tzinfo=UTC), date(2026, 9, 13),
                 "Hotfix 1.1.1", "hotfix", "p1"),
    )  # fmt: skip
    return GameData(
        appid=GAME,
        name="Helldivers 2",
        genres="Action, Shooter",
        days=tuple(days),
        aspects=tuple(aspects),
        buckets=tuple(buckets),
        patches=patches,
        estimates=ESTIMATES,
    )


def other_game() -> GameData:
    day = date(2026, 10, 1)
    return GameData(
        appid=OTHER_GAME,
        name="Cities: Skylines II",
        genres="Simulation",
        days=(DayRow(day, 3, 4, sale=False, discount=None),),
        aspects=(),
        buckets=(BucketRow(day, "2h_10h", 3, 4),),
        patches=(),
        estimates=(),
    )


def make_bundle(*games: GameData, default_method: str = "synth") -> Bundle:
    return Bundle(
        data_version=DATA_VERSION,
        built_on=LAST + timedelta(days=1),
        causal=CausalConfig(default_method, "e2", "ab" * 32, validated=True),
        games=games or (game_data(), other_game()),
    )


SAMPLE_QUOTE = "the frame rate halves in every city map since the update"  # invented


def sample_report(bundle_id: str = "p1", horizon: int = 7, *, basic: bool = False) -> Report:
    end = PATCH_DAY + timedelta(days=horizon - 1)
    return Report.model_validate(
        {
            "report_id": f"{bundle_id}-{horizon}d",
            "appid": GAME,
            "scope": "bundle",
            "bundle_id": bundle_id,
            "horizon": horizon,
            "question": None,
            "window": {"start": PATCH_DAY.isoformat(), "end": end.isoformat()},
            "data_version": DATA_VERSION,
            "model_ids": {} if basic else {"draft": "gemini-3.5-flash-lite"},
            "prompt_version": PROMPT_VERSION,
            "summary": "Negative reviews mentioning performance rose after the patch.",
            "findings": [{"title": "Performance", "hypothesis": "H3", "claim_ids": ["c1"]}],
            "hypotheses": [
                {"code": "H3", "status": "supported", "rule": "H3.rise", "evidence": {"v": 1.0}}
            ],
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "Negative reviews mentioning performance rose from 12% to 31%.",
                    "claim_type": "descriptive",
                    "value": 19.0,
                    "unit": "pp",
                    "evidence": {"call_id": "r1.6", "json_path": "aspects.performance.change"},
                    "review_ids": [7001, 7002],
                    "quote_spans": [{"review_id": 7003, "text": SAMPLE_QUOTE}],
                }
            ],
            "withheld_claims": [
                {
                    "claim": {
                        "claim_id": "c2",
                        "text": "The patch caused the drop.",
                        "claim_type": "causal",
                    },
                    "rule": "V4",
                }
            ],
            "confidence": "low" if basic else "medium",
            "basic": basic,
            "rule_pass_rates": {"V1": 1.0},
        }
    )
