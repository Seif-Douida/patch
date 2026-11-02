"""The agent's data bundle (plan Task 2, design D2): aggregates only, gzip JSON, an allowlist of
field names, and a schema version a reader checks."""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime

import pytest

from patchpulse.agent.bundle import (
    BUNDLE_SCHEMA,
    AspectRow,
    BucketRow,
    Bundle,
    DayRow,
    GameData,
    PatchRow,
    dump_bundle,
    load_bundle,
)
from patchpulse.causal.config import CausalConfig
from patchpulse.causal.store import StoredEstimate

DAY = date(2026, 10, 1)
STORED_AT = datetime(2026, 10, 9, 2, 30, tzinfo=UTC)


def stored(horizon: int, method: str, effect: float | None) -> StoredEstimate:
    synth = method == "synth"
    return StoredEstimate(
        bundle_id="5001",
        patch_ids=("5001", "5002"),
        appid=553850,
        outcome="positive_share",
        sample="all",
        horizon=horizon,
        method=method,
        run_id=41,
        effect=effect,
        lo90=None if effect is None else effect - 1.2345678806304932,
        hi90=None if effect is None else effect + 2.0999999046325684,
        p_value=0.04545454680919647 if synth else None,
        pre_fit_rmse=0.9817 if synth else None,
        donors=((1086940, 0.625), (892970, 0.375)) if synth else (),
        flags=frozenset({"volume_spike", "discounts_unknown"}),
        reason=None if effect is not None else "interval empty",
        fingerprint=f"{horizon}{method}".ljust(64, "f"),
        version="e2",
        computed_at=STORED_AT,
    )


def sample_bundle() -> Bundle:
    game = GameData(
        appid=553850,
        name="Helldivers 2",
        genres="Action, Shooter",
        days=(
            DayRow(date=DAY, positives=120, totals=150, sale=False, discount=None),
            DayRow(date=date(2026, 10, 2), positives=0, totals=0, sale=True, discount=40),
        ),
        aspects=(
            AspectRow(
                date=DAY,
                aspect="performance",
                neg_mentions=9,
                neg_scored=30,
                pos_mentions=4,
                pos_scored=120,
            ),
        ),
        buckets=(
            BucketRow(date=DAY, bucket="lt_2h", positives=20, totals=31),
            BucketRow(date=DAY, bucket="unknown", positives=1, totals=1),
        ),
        patches=(
            PatchRow(
                patch_id="5001",
                gid="5001",
                published_at=datetime(2026, 10, 1, 16, 5, tzinfo=UTC),
                day=DAY,
                title="Patch 1.004.100",
                patch_type="patch",
                bundle_id="5001",
            ),
            PatchRow(
                patch_id="5003",
                gid="5003",
                published_at=datetime(2026, 10, 2, 9, 0, tzinfo=UTC),
                day=date(2026, 10, 2),
                title="Halloween event",
                patch_type="event",
                bundle_id=None,
            ),
        ),
        estimates=(
            stored(7, "its", -3.4000000953674316),
            stored(7, "synth", -2.5),
            stored(14, "synth", None),
        ),
    )
    return Bundle(
        data_version=41,
        built_on=date(2026, 10, 9),
        causal=CausalConfig("synth", "e2", "ab" * 32, validated=True, spike_causal_ok=False),
        games=(game,),
    )


def test_bundle_round_trips() -> None:
    bundle = sample_bundle()

    data = dump_bundle(bundle)

    assert load_bundle(data) == bundle
    assert dump_bundle(load_bundle(data)) == data  # the same bundle, the same bytes
    assert json.loads(gzip.decompress(data))["schema"] == BUNDLE_SCHEMA


def test_bundle_refuses_fields_outside_the_allowlist() -> None:
    @dataclass(frozen=True)
    class LeakyDay(DayRow):
        text: str = "the frame pacing has been awful since the patch"

    bundle = sample_bundle()
    (game,) = bundle.games
    leaky = replace(bundle, games=(replace(game, days=(LeakyDay(**asdict(game.days[0])),)),))

    with pytest.raises(ValueError, match="'text'"):
        dump_bundle(leaky)


def test_unknown_bundle_schema_is_refused() -> None:
    document = json.loads(gzip.decompress(dump_bundle(sample_bundle())))
    document["schema"] = BUNDLE_SCHEMA + 1

    with pytest.raises(ValueError, match="schema"):
        load_bundle(gzip.compress(json.dumps(document).encode()))
