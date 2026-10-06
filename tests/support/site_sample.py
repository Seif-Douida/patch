"""Synthetic site data for `web/fixtures/`, built by the real exporter (`shape_site_export`).

The dashboard builds and tests against these files, so they must have exactly the shape the
nightly job publishes; `tests/export/test_web_fixtures.py` fails if they drift. The games and
numbers are invented (no Steam data, no review text): two fictional titles over 120 days, one
with a patch that drops sentiment and a Steam sale that lifts volume.

Regenerate after changing the exporter: `uv run python tests/support/site_sample.py web/fixtures`
"""

from __future__ import annotations

import math
import random
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from patchpulse.export.site import DayRow, GameRow, PatchRow, shape_site_export

Z90 = 1.6449
START = date(2026, 6, 1)
DAYS = 120
GENERATED_AT = datetime(2026, 9, 29, 3, 32, tzinfo=UTC)


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


def sample_files() -> dict[str, bytes]:
    games = [
        GameRow(900001, "Example Arena (sample data)", "Action"),
        GameRow(900002, "Example Colony (sample data)", "Simulation, Strategy"),
    ]
    days = _series(900001, volume=180, base=0.78, dip_day=70, sale=range(25, 32), seed=1)
    days += _series(900002, volume=45, base=0.62, dip_day=None, sale=range(25, 32), seed=2)
    patches = [
        PatchRow(
            900001,
            datetime(2026, 8, 10, 17, 0, tzinfo=UTC),
            "Update 2.0: new progression system",
            "https://example.com/arena/2-0",
            "major_update",
            True,
        ),
        PatchRow(
            900001,
            datetime(2026, 8, 13, 9, 30, tzinfo=UTC),
            "Hotfix 2.0.1",
            "https://example.com/arena/2-0-1",
            "hotfix",
            False,
        ),
        PatchRow(
            900002,
            datetime(2026, 7, 2, 12, 0, tzinfo=UTC),
            "Patch 0.9.4",
            "https://example.com/colony/0-9-4",
            "patch",
            True,
        ),
        PatchRow(
            900002,
            datetime(2026, 9, 1, 12, 0, tzinfo=UTC),
            "Dev blog: what's next",
            "https://example.com/colony/blog",
            "other",
            False,
        ),
    ]
    return shape_site_export(games, days, patches, data_version=42, generated_at=GENERATED_AT)


def write(directory: Path) -> None:
    for relative, content in sample_files().items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
