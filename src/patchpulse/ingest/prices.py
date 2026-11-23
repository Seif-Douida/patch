"""Daily price snapshots from the store's appdetails endpoint (spec §6.1).

`filters=price_overview` lets one request carry many app IDs (verified 2026-09-28). US prices are
used because `cc=gb` answered `success: false` for Helldivers 2 while `cc=us` worked for every
candidate; the analysis only uses `discount_percent`, and the currency is recorded anyway.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from patchpulse.ingest.http import SteamHttp
from patchpulse.ingest.models import PriceSnapshot, RawPage

APPDETAILS_URL = "https://store.steampowered.com/api/appdetails"
PRICE_BATCH_SIZE = 25
PRICE_COUNTRY = "us"


@dataclass(frozen=True)
class PriceFetch:
    raws: list[RawPage]
    snapshots: dict[int, PriceSnapshot | None]  # None: failed, or free with no price
    failed: frozenset[int]  # Steam answered success: false


def fetch_prices(http: SteamHttp, appids: Sequence[int], *, cc: str = PRICE_COUNTRY) -> PriceFetch:
    raws: list[RawPage] = []
    snapshots: dict[int, PriceSnapshot | None] = {}
    failed: set[int] = set()
    for offset in range(0, len(appids), PRICE_BATCH_SIZE):
        batch = appids[offset : offset + PRICE_BATCH_SIZE]
        request: dict[str, str | int] = {
            "appids": ",".join(str(appid) for appid in batch),
            "cc": cc,
            "filters": "price_overview",
        }
        payload, status = http.get_json(APPDETAILS_URL, request)
        raws.append(RawPage("prices", None, request, status, payload, len(batch)))
        for appid in batch:
            entry = payload.get(str(appid))
            if not isinstance(entry, dict) or not entry.get("success"):
                failed.add(appid)
                snapshots[appid] = None
                continue
            data = entry.get("data")
            overview = data.get("price_overview") if isinstance(data, dict) else None
            snapshots[appid] = PriceSnapshot.model_validate(overview) if overview else None
    return PriceFetch(raws, snapshots, frozenset(failed))


def fill_one_day_gaps(discounts: Mapping[date, int | None]) -> dict[date, int | None]:
    """A game's discount by day, with each single day lacking a price given the discount of the
    day before and the day after when both have one and they are equal.

    The nightly runs every other night since 5 Nov 2026, so every second day has no price
    snapshot. Anything else stays None: a gap of two days or more (a failed night), neighbours
    that differ (a discount began or ended there), and a day whose calendar neighbour isn't in
    `discounts`. The cost: a discount that starts and ends between two snapshots is missed.
    Returns a new dict with the same keys in the same order."""
    filled = dict(discounts)
    for day, discount in discounts.items():
        if discount is not None:
            continue
        before = discounts.get(day - timedelta(days=1))
        after = discounts.get(day + timedelta(days=1))
        if before is not None and before == after:
            filled[day] = before
    return filled
