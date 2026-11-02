"""The causal gate (plan Task 5, design D7 V4, Seif's option C): when an estimate may be stated as
the patch's effect, and the first condition that blocks it otherwise."""

from __future__ import annotations

from dataclasses import replace

import pytest
from agent_fixtures import estimate

from patchpulse.agent.gate import (
    CONFOUNDED,
    DISCOUNTS_NOTE,
    NO_ESTIMATE,
    NOT_VALIDATED,
    PLACEBO_FAILED,
    RANGE_INCLUDES_ZERO,
    SPIKE_CAVEAT,
    VOLUME_SPIKE,
    Gate,
    causal_gate,
)
from patchpulse.causal.config import CausalConfig
from patchpulse.causal.store import StoredEstimate


def config(*, validated: bool = True, spike_ok: bool = False) -> CausalConfig:
    return CausalConfig("its", "e2", "ab" * 32, validated=validated, spike_causal_ok=spike_ok)


def row(
    lo: float | None = -14.0,
    hi: float | None = -9.0,
    *,
    method: str = "its",
    p: float | None = None,
    flags: tuple[str, ...] = (),
    outcome: str = "positive_share",
    sample: str = "all",
) -> StoredEstimate:
    effect = None if lo is None or hi is None else (lo + hi) / 2
    return replace(
        estimate(7, method, effect, outcome=outcome, sample=sample),
        lo90=lo,
        hi90=hi,
        p_value=p,
        flags=frozenset(flags),
    )


ALLOWED = Gate(allowed=True, caveat=None, blocked_by=None)


def blocked(reason: str) -> Gate:
    return Gate(allowed=False, caveat=None, blocked_by=reason)


@pytest.mark.parametrize(
    ("case", "settings", "expected"),
    [
        (row(), config(), ALLOWED),
        (row(9.0, 14.0), config(), ALLOWED),  # a rise excludes 0 too
        (row(None, None), config(), blocked(NO_ESTIMATE)),
        (row(-4.0, 1.0), config(), blocked(RANGE_INCLUDES_ZERO)),
        (row(-4.0, 0.0), config(), blocked(RANGE_INCLUDES_ZERO)),  # touching 0 includes it
        (row(method="synth", p=0.2), config(), blocked(PLACEBO_FAILED)),
        (row(method="synth", p=None), config(), blocked(PLACEBO_FAILED)),
        (row(method="synth", p=0.10), config(), ALLOWED),  # spec §9.2's 10% level
        (row(method="synth", p=0.11), config(), blocked(PLACEBO_FAILED)),
        (row(), config(validated=False), blocked(NOT_VALIDATED)),
        (row(outcome="aspect:performance"), config(), blocked(NOT_VALIDATED)),
        (row(sample="established"), config(), blocked(NOT_VALIDATED)),
        (row(outcome="volume"), config(), blocked(NOT_VALIDATED)),
        (row(flags=("confounded",)), config(), blocked(CONFOUNDED)),
        (row(flags=("volume_spike",)), config(), blocked(VOLUME_SPIKE)),
        # the first failing condition, in the order above
        (row(-4.0, 1.0, flags=("confounded",)), config(), blocked(RANGE_INCLUDES_ZERO)),
        (row(method="synth", p=0.5), config(validated=False), blocked(PLACEBO_FAILED)),
        (row(flags=("confounded",)), config(validated=False), blocked(NOT_VALIDATED)),
        (row(flags=("confounded", "volume_spike")), config(), blocked(CONFOUNDED)),
    ],
)
def test_each_gate_condition(case: StoredEstimate, settings: CausalConfig, expected: Gate) -> None:
    assert causal_gate(case, settings) == expected


def test_spike_follows_the_benchmark_verdict() -> None:
    spike = row(flags=("volume_spike",))

    assert causal_gate(spike, config(spike_ok=True)) == Gate(True, SPIKE_CAVEAT, None)
    assert causal_gate(spike, config(spike_ok=False)) == blocked(VOLUME_SPIKE)


def test_unknown_discounts_get_the_note() -> None:
    unknown = row(flags=("discounts_unknown",))
    both = row(flags=("discounts_unknown", "volume_spike"))

    assert causal_gate(unknown, config()) == Gate(True, DISCOUNTS_NOTE, None)
    assert causal_gate(both, config(spike_ok=True)) == Gate(
        True, f"{SPIKE_CAVEAT} {DISCOUNTS_NOTE}", None
    )
    assert causal_gate(both, config(spike_ok=False)) == blocked(VOLUME_SPIKE)
