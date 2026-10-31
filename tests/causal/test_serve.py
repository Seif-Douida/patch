"""Which stored row a reader is served (store.serve): the default method's, or interrupted time
series' where the default has no estimate with a range, as the site shows it."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from patchpulse.causal.store import StoredEstimate, serve

NO_RANGE = "interval wider than the grid"


def stored(method: str, **changes: object) -> StoredEstimate:
    row = StoredEstimate(
        bundle_id="b1",
        patch_ids=("p1",),
        appid=553850,
        outcome="positive_share",
        sample="all",
        horizon=14,
        method=method,
        run_id=1,
        effect=-2.0,
        lo90=-4.0,
        hi90=-0.5,
        p_value=None,
        pre_fit_rmse=None,
        donors=(),
        flags=frozenset(),
        reason=None,
        fingerprint="f" * 64,
        version="e2",
        computed_at=datetime(2026, 11, 2, 3, tzinfo=UTC),
    )
    return replace(row, **changes)  # type: ignore[arg-type]


def test_a_default_without_a_range_falls_back() -> None:
    no_range = stored("synth", effect=-1.2, lo90=None, hi90=None, reason=NO_RANGE)
    its = stored("its")

    assert serve([no_range, its], "synth", fallback=True) == [replace(its, reason=NO_RANGE)]
    assert serve([no_range, its], "synth", fallback=False) == [no_range]


def test_an_outcome_the_default_isnt_run_for_falls_back() -> None:
    aspect = stored("its", outcome="aspect:performance")
    thin = replace(aspect, horizon=7, effect=None, lo90=None, hi90=None, reason="too few reviews")

    assert serve([aspect, thin], "synth", fallback=True) == [
        thin,  # nothing to serve but its own reason
        replace(aspect, reason="synthetic control isn't run for this outcome or sample"),
    ]
    assert serve([aspect, thin], "synth", fallback=False) == []
