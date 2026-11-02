"""Synthetic site data for `web/fixtures/`, built by the real exporter (`shape_site_export`).

The dashboard builds and tests against these files, so they must have exactly the shape the
nightly job publishes; `tests/export/test_web_fixtures.py` fails if they drift. The games and
numbers are invented (no Steam data, no review text): two fictional titles over 120 days, one
with a patch that drops sentiment and a Steam sale that lifts volume. Aspect counts are invented
too: the first two weeks aren't scored yet (as during a backlog), and the bad patch brings a wave
of gameplay-balance complaints. So are the patch estimates (schema 3), one of each kind the site
explains: an effect, an effect during a sale, too few comparison games, a window not yet over.
And the reports (schema 4), one of each kind the report page explains: a verified report with a
causal claim, one whose causal claim the verifier reworded and some withheld, and a basic report
written without a model. Their quote spans and review ids are invented too, and the exporter drops
them as it drops the real ones.

Regenerate after changing the exporter: `uv run python tests/support/site_sample.py web/fixtures`
"""

from __future__ import annotations

import math
import random
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from patchpulse.agent.report import Report
from patchpulse.export.site import (
    AspectDayRow,
    AspectModel,
    Aspects,
    DayRow,
    EffectRow,
    Effects,
    GameRow,
    PatchRow,
    shape_site_export,
)
from patchpulse.labeling.taxonomy import ASPECTS

Z90 = 1.6449
START = date(2026, 6, 1)
DAYS = 120
GENERATED_AT = datetime(2026, 9, 29, 3, 32, tzinfo=UTC)
UNSCORED_DAYS = 14
# Invented per-aspect F1, three under the site's 0.5 bar so the sample shows them marked.
MODEL = AspectModel(
    family="s0",
    version="20260901T000000Z",
    test_macro_f1=0.61,
    aspect_f1=dict(
        zip(ASPECTS, (0.85, 0.66, 0.82, 0.72, 0.56, 0.4, 0.77, 0.48, 0.2, 0.7), strict=True)
    ),
)
# How often a negative or a positive review mentions each aspect, in ASPECTS order.
NEG_RATES = (0.25, 0.2, 0.15, 0.12, 0.1, 0.1, 0.03, 0.06, 0.04, 0.08)
POS_RATES = (0.05, 0.03, 0.3, 0.35, 0.05, 0.05, 0.25, 0.05, 0.01, 0.15)
BALANCE = ASPECTS.index("gameplay_balance")


def _wilson(positive: int, n: int) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = positive / n
    centre = p + Z90**2 / (2 * n)
    spread = Z90 * math.sqrt(p * (1 - p) / n + Z90**2 / (4 * n * n))
    denominator = 1 + Z90**2 / n
    return max(0.0, (centre - spread) / denominator), min(1.0, (centre + spread) / denominator)


def _series(
    appid: int, *, volume: int, base: float, dip_day: int | None, sale: range, seed: int
) -> list[DayRow]:
    rng = random.Random(seed)  # noqa: S311 - invents sample numbers; nothing secret
    rows = []
    for offset in range(DAYS):
        day = START + timedelta(days=offset)
        share = base
        if dip_day is not None and offset >= dip_day:
            share = base - 0.3 * math.exp(-(offset - dip_day) / 12)  # a bad patch, then recovery
        n = max(0, int(rng.gauss(volume, volume * 0.25)))
        if offset in sale:
            n = int(n * 2.2)
        if offset == 40:
            n = 0  # a day without reviews: no share, no interval
        positive = sum(rng.random() < share for _ in range(n))
        lo, hi = _wilson(positive, n)
        rows.append(
            DayRow(
                appid=appid,
                date=day,
                n_reviews=n,
                n_positive=positive,
                pos_share=positive / n if n else None,
                pos_lo90=lo,
                pos_hi90=hi,
                discount_pct=40 if offset in sale else 0,
                is_sale=offset in sale,
            )
        )
    return rows


def _aspect_days(rows: list[DayRow], *, spike_day: int | None, seed: int) -> list[AspectDayRow]:
    rng = random.Random(seed)  # noqa: S311 - invents sample numbers; nothing secret
    out = []
    for offset, row in enumerate(rows):
        if offset < UNSCORED_DAYS or row.n_reviews == 0:
            continue
        neg_rates = list(NEG_RATES)
        if spike_day is not None and offset >= spike_day:
            neg_rates[BALANCE] += 0.45 * math.exp(-(offset - spike_day) / 12)
        n_neg = row.n_reviews - row.n_positive
        out.append(
            AspectDayRow(
                appid=row.appid,
                date=row.date,
                n_neg_scored=n_neg,
                n_pos_scored=row.n_positive,
                neg=tuple(sum(rng.random() < rate for _ in range(n_neg)) for rate in neg_rates),
                pos=tuple(
                    sum(rng.random() < rate for _ in range(row.n_positive)) for rate in POS_RATES
                ),
            )
        )
    return out


def sample_files() -> dict[str, bytes]:
    games = [
        GameRow(900001, "Example Arena (sample data)", "Action"),
        GameRow(900002, "Example Colony (sample data)", "Simulation, Strategy"),
    ]
    arena = _series(900001, volume=180, base=0.78, dip_day=70, sale=range(25, 32), seed=1)
    colony = _series(900002, volume=45, base=0.62, dip_day=None, sale=range(25, 32), seed=2)
    days = arena + colony
    aspect_days = _aspect_days(arena, spike_day=70, seed=3)
    aspect_days += _aspect_days(colony, spike_day=None, seed=4)
    patches = [
        PatchRow(
            900001,
            datetime(2026, 8, 10, 17, 0, tzinfo=UTC),
            "Update 2.0: new progression system",
            "https://example.com/arena/2-0",
            "major_update",
            True,
            "5001",
        ),
        PatchRow(
            900001,
            datetime(2026, 8, 13, 9, 30, tzinfo=UTC),
            "Hotfix 2.0.1",
            "https://example.com/arena/2-0-1",
            "hotfix",
            False,
            "5002",
        ),
        PatchRow(
            900001,
            datetime(2026, 9, 24, 16, 0, tzinfo=UTC),
            "Patch 2.1: balance changes",
            "https://example.com/arena/2-1",
            "patch",
            True,
            "5003",
        ),
        PatchRow(
            900002,
            datetime(2026, 7, 2, 12, 0, tzinfo=UTC),
            "Patch 0.9.4",
            "https://example.com/colony/0-9-4",
            "patch",
            True,
            "5011",
        ),
        PatchRow(
            900002,
            datetime(2026, 8, 20, 12, 0, tzinfo=UTC),
            "Patch 0.9.5",
            "https://example.com/colony/0-9-5",
            "patch",
            True,
            "5012",
        ),
        PatchRow(
            900002,
            datetime(2026, 9, 1, 12, 0, tzinfo=UTC),
            "Dev blog: what's next",
            "https://example.com/colony/blog",
            "other",
            False,
            "5013",
        ),
    ]
    return shape_site_export(
        games,
        days,
        patches,
        data_version=42,
        generated_at=GENERATED_AT,
        aspects=Aspects(MODEL, aspect_days),
        effects=EFFECTS,
        reports=REPORTS,
    )


TOO_FEW = "too few comparison games without their own patch (3 of 5 needed)"
# Invented estimates, served as the nightly job serves them (the default method's row per bundle
# and horizon). Update 2.0's drop; Patch 0.9.4 during the sale; Patch 0.9.5 without enough
# comparison games (and too little quiet history for the fallback); Patch 2.1's 7 days end after
# the sample's night, so it has none.
EFFECTS = Effects(
    default_method="synth",
    effects_version="e2",
    benchmark_report="reports/causal/benchmark.md",
    validated=True,
    rows=[
        EffectRow("5001", 7, -12.4, -16.0, -8.9, "synth", frozenset(), None),
        EffectRow("5001", 14, -9.8, -13.1, -6.4, "synth", frozenset(), None),
        EffectRow("5011", 7, 1.1, -2.9, 5.0, "synth", frozenset({"confounded"}), None),
        EffectRow("5011", 14, -0.6, -4.4, 3.1, "synth", frozenset({"confounded"}), None),
        EffectRow("5012", 7, None, None, None, "synth", frozenset(), TOO_FEW),
        EffectRow("5012", 14, None, None, None, "synth", frozenset(), TOO_FEW),
    ],
)


MODELS = {s: "gemini-3.5-flash-lite" for s in ("gather", "draft", "summary")}
ALL_PASSED = {f"V{n}": 1.0 for n in range(1, 8)}


def _hypotheses(**statuses: str) -> list[dict[str, object]]:
    return [
        {"code": code, "status": status, "rule": f"{code}.sample", "evidence": {}}
        for code, status in sorted(statuses.items())
    ]


def _claim(
    claim_id: str, text: str, claim_type: str, value: float, **extra: object
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "text": text,
        "claim_type": claim_type,
        "value": value,
        "unit": "pp",
        "evidence": {"call_id": "s1.3", "json_path": "total.share"},
        **extra,
    }


def _report(bundle_id: str, start: date, appid: int, **fields: object) -> Report:
    return Report.model_validate(
        {
            "report_id": f"{bundle_id}-14d",
            "appid": appid,
            "scope": "bundle",
            "bundle_id": bundle_id,
            "horizon": 14,
            "question": None,
            "window": {
                "start": start.isoformat(),
                "end": (start + timedelta(days=13)).isoformat(),
            },
            "data_version": 42,
            "prompt_version": "5a3c9e1f0b72",
            "withheld_claims": [],
            **fields,
        }
    )


# Invented reports for three bundles; Patch 2.1's window isn't over, so it has none.
REPORTS = {
    "5001": _report(
        "5001",
        date(2026, 8, 10),
        900001,
        model_ids=MODELS,
        summary=(
            "Positive reviews fell by about 14 points in the two weeks after Update 2.0, and the "
            "drop holds against comparison games. Complaints about gameplay balance rose the most. "
            "(sample data)"
        ),
        findings=[
            {"title": "The patch", "hypothesis": "H1", "claim_ids": ["c1", "c2"]},
            {"title": "Gameplay balance complaints", "hypothesis": "H1", "claim_ids": ["c3"]},
        ],
        hypotheses=_hypotheses(
            H1="supported", H2="rejected", H3="rejected", H4="rejected", H5="rejected"
        ),
        claims=[
            _claim(
                "c1",
                "The positive share fell from 78.2% to 64.4% in the 14 days after the update.",
                "comparative",
                -13.8,
                review_ids=[910001, 910002, 910003],
            ),
            _claim(
                "c2",
                "Update 2.0 lowered the positive share by about 9.8 points (90% range 6.4 to "
                "13.1).",
                "causal",
                -9.8,
                review_ids=[910004, 910005],
                quote_spans=[{"review_id": 910006, "text": "the new progression grind is endless"}],
            ),
            _claim(
                "c3",
                "Negative reviews mentioning gameplay balance rose from 18% to 41%.",
                "comparative",
                23.0,
                quote_spans=[{"review_id": 910007, "text": "every match ends the same way now"}],
            ),
        ],
        confidence="high",
        basic=False,
        rule_pass_rates=ALL_PASSED,
    ),
    "5011": _report(
        "5011",
        date(2026, 7, 2),
        900002,
        model_ids=MODELS,
        summary=(
            "Reviews stayed about as positive after Patch 0.9.4. A Steam sale ran over the same "
            "days, so the patch's own effect can't be told apart from the sale's. (sample data)"
        ),
        findings=[
            {"title": "The patch", "hypothesis": "H1", "claim_ids": ["c2"]},
            {"title": "A price change or sale", "hypothesis": "H2", "claim_ids": ["c1"]},
        ],
        hypotheses=_hypotheses(
            H1="inconclusive", H2="supported", H3="rejected", H4="rejected", H5="inconclusive"
        ),
        claims=[
            _claim(
                "c1",
                "Reviews per day doubled during the sale, from 45 to 92.",
                "comparative",
                47.0,
                unit="reviews",
                review_ids=[920001],
            ),
            _claim(
                "c2",
                "The data are consistent with Patch 0.9.4 moving the positive share by about -0.6 "
                "points, but a sale ran over the same days, so this is not established.",
                "causal",
                -0.6,
                reworded=True,
            ),
        ],
        withheld_claims=[
            {
                "claim": {
                    "claim_id": "c3",
                    "text": "Players loved the new colony map.",
                    "claim_type": "descriptive",
                },
                "rule": "V1",
            },
            {
                "claim": {
                    "claim_id": "c4",
                    "text": "The share rose by 6 points.",
                    "claim_type": "comparative",
                },
                "rule": "V2",
            },
            {
                "claim": {
                    "claim_id": "c5",
                    "text": "The sale caused the rise.",
                    "claim_type": "causal",
                },
                "rule": "V4",
            },
        ],
        confidence="medium",
        basic=False,
        rule_pass_rates={**ALL_PASSED, "V1": 0.75, "V2": 0.8},
    ),
    "5012": _report(
        "5012",
        date(2026, 8, 20),
        900002,
        model_ids={},
        summary=(
            "Inconclusive: the patch, noise or seasonality. Ruled out: a price change or sale, "
            "off-topic reviews or a review bomb, a change in who reviewed. The claims below passed "
            "the verifier. (sample data)"
        ),
        findings=[{"title": "The patch", "hypothesis": "H1", "claim_ids": ["c1"]}],
        hypotheses=_hypotheses(
            H1="inconclusive", H2="rejected", H3="rejected", H4="rejected", H5="inconclusive"
        ),
        claims=[
            _claim(
                "c1",
                "The positive share went from 61.8% in the baseline to 60.9% in the window.",
                "comparative",
                -0.9,
            )
        ],
        confidence="medium",
        basic=True,
        rule_pass_rates={"V1": 1.0, "V2": 1.0, "V3": 1.0},
    ),
}


def write(directory: Path) -> None:
    for relative, content in sample_files().items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
