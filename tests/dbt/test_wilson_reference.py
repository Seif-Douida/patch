"""The Wilson 90% interval macro, checked against statsmodels and a hand calculation (spec §0.6).

The macro is compiled by dbt and executed on SQL Server, so this tests the SQL that the marts run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text
from statsmodels.stats.proportion import proportion_confint

from patchpulse.config import Settings
from patchpulse.db.engine import make_engine
from patchpulse.pipeline.dbt import compile_inline

pytestmark = pytest.mark.integration

CASES = [(8, 10), (0, 5), (5, 5), (500, 1000), (1, 3), (0, 0)]


@pytest.fixture(scope="module")
def bounds(
    sql_settings: Settings, tmp_path_factory: pytest.TempPathFactory
) -> dict[tuple[int, int], tuple[float | None, float | None]]:
    values = ", ".join(f"({pos}, {n})" for pos, n in CASES)
    sql = compile_inline(
        "select p.pos, p.n, {{ wilson_lower('p.pos', 'p.n') }} as lo, "
        "{{ wilson_upper('p.pos', 'p.n') }} as hi "
        f"from (values {values}) as p(pos, n)",
        settings=sql_settings,
        target_path=tmp_path_factory.mktemp("dbt-target"),
    )
    with make_engine(sql_settings).connect() as connection:
        return {(r.pos, r.n): (r.lo, r.hi) for r in connection.execute(text(sql))}


def test_hand_calculated_value(
    bounds: dict[tuple[int, int], tuple[float | None, float | None]],
) -> None:
    lo, hi = bounds[(8, 10)]
    assert lo == pytest.approx(0.5408, abs=1e-3)
    assert hi == pytest.approx(0.9314, abs=1e-3)


@pytest.mark.parametrize(("pos", "n"), [case for case in CASES if case[1] > 0])
def test_matches_statsmodels(
    bounds: dict[tuple[int, int], tuple[float | None, float | None]], pos: int, n: int
) -> None:
    expected_lo, expected_hi = proportion_confint(pos, n, alpha=0.10, method="wilson")
    lo, hi = bounds[(pos, n)]
    # statsmodels uses z = 1.644854; the macro pins 1.6449.
    assert lo == pytest.approx(expected_lo, abs=1e-4)
    assert hi == pytest.approx(expected_hi, abs=1e-4)


def test_bounds_stay_within_0_and_1(
    bounds: dict[tuple[int, int], tuple[float | None, float | None]],
) -> None:
    # Unclamped, an all-positive day gives an upper bound of 1.0000000000000002.
    for lo, hi in bounds.values():
        if lo is not None and hi is not None:
            assert 0.0 <= lo <= hi <= 1.0


def test_no_reviews_gives_no_interval(
    bounds: dict[tuple[int, int], tuple[float | None, float | None]],
) -> None:
    assert bounds[(0, 0)] == (None, None)


def test_compiled_sql_is_not_left_on_disk_in_the_repo(tmp_path: Path) -> None:
    # compile_inline writes dbt's target files where it's told, never into the repo's dbt/.
    assert not (
        Path(__file__).resolve().parents[2] / "dbt" / "target" / "run_results.json"
    ).exists()
