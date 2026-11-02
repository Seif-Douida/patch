"""The causal gate (design D7, V4; Seif's option C, 25 Oct 2026): when an estimate may be stated
as the patch's effect.

All must hold, checked in this order; `blocked_by` names the first that fails:
1. an estimate with a 90% range (`no_estimate`);
2. the range excludes 0; a range touching 0 includes it (`range_includes_zero`);
3. the method's placebo check: synthetic control's placebo p-value at most PLACEBO_P; interrupted
   time series' range is calibrated on the game's own past, which is its check (`placebo_failed`);
4. the benchmark validated it: `validated` in config/causal.yaml, and the estimate is of what the
   benchmark measured, the positive share of all reviews (`not_validated`);
5. no Steam sale or game discount within 3 days of the bundle's start (`confounded`);
6. a volume spike only when the benchmark's spike verdict allows it (`volume_spike`).
An allowed estimate on a spike window carries SPIKE_CAVEAT; one from before discounts were tracked
carries DISCOUNTS_NOTE.
"""

from __future__ import annotations

from dataclasses import dataclass

from patchpulse.causal.config import CausalConfig
from patchpulse.causal.its import POSITIVE_SHARE
from patchpulse.causal.store import StoredEstimate

PLACEBO_P = 0.10  # spec §9.2: the placebo-in-space test at the 10% level
VALIDATED_SAMPLE = "all"  # the benchmark's trials: the positive share of all reviews

NO_ESTIMATE = "no_estimate"
RANGE_INCLUDES_ZERO = "range_includes_zero"
PLACEBO_FAILED = "placebo_failed"
NOT_VALIDATED = "not_validated"
CONFOUNDED = "confounded"
VOLUME_SPIKE = "volume_spike"
BLOCKERS = (
    NO_ESTIMATE,
    RANGE_INCLUDES_ZERO,
    PLACEBO_FAILED,
    NOT_VALIDATED,
    CONFOUNDED,
    VOLUME_SPIKE,
)

SPIKE_CAVEAT = (
    "Reviews came in a burst during this window; on such windows the method raised false alarms "
    "at most 10% of the time in its benchmark."
)
DISCOUNTS_NOTE = (
    "The game's discounts weren't tracked at the time, so a price change can't be ruled out."
)


@dataclass(frozen=True)
class Gate:
    allowed: bool
    caveat: str | None
    blocked_by: str | None


def _blocker(row: StoredEstimate, config: CausalConfig) -> str | None:
    if row.effect is None or row.lo90 is None or row.hi90 is None:
        return NO_ESTIMATE
    if row.lo90 <= 0 <= row.hi90:
        return RANGE_INCLUDES_ZERO
    if row.method == "synth" and (row.p_value is None or row.p_value > PLACEBO_P):
        return PLACEBO_FAILED
    if not config.validated or (row.outcome, row.sample) != (POSITIVE_SHARE, VALIDATED_SAMPLE):
        return NOT_VALIDATED
    if "confounded" in row.flags:
        return CONFOUNDED
    if "volume_spike" in row.flags and not config.spike_causal_ok:
        return VOLUME_SPIKE
    return None


def causal_gate(row: StoredEstimate, config: CausalConfig) -> Gate:
    blocker = _blocker(row, config)
    if blocker is not None:
        return Gate(allowed=False, caveat=None, blocked_by=blocker)
    caveats = [
        text
        for flag, text in (("volume_spike", SPIKE_CAVEAT), ("discounts_unknown", DISCOUNTS_NOTE))
        if flag in row.flags
    ]
    return Gate(allowed=True, caveat=" ".join(caveats) or None, blocked_by=None)
