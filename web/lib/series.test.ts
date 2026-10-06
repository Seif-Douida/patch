import { describe, expect, it } from "vitest";

import {
  dateTicks,
  fillDays,
  formatDate,
  formatShare,
  lastDays,
  placePatchMarkers,
  rollingShare,
  saleRanges,
} from "./series";
import type { DayPoint, Patch } from "./types";

function day(date: string, n: number, pos: number, extra: Partial<DayPoint> = {}): DayPoint {
  return {
    date,
    n,
    pos,
    share: n ? pos / n : null,
    lo: null,
    hi: null,
    sale: false,
    discount: null,
    ...extra,
  };
}

function patch(date: string, title: string, treatment = true): Patch {
  return { date, title, url: `https://example.com/${title}`, type: "patch", treatment };
}

describe("fillDays", () => {
  it("fills_missing_days_with_null_share", () => {
    const filled = fillDays([day("2026-09-27", 10, 7), day("2026-09-30", 4, 2)]);

    expect(filled.map((d) => d.date)).toEqual([
      "2026-09-27",
      "2026-09-28",
      "2026-09-29",
      "2026-09-30",
    ]);
    expect(filled[1]).toMatchObject({ n: 0, pos: 0, share: null, lo: null, hi: null, sale: false });
    expect(filled[3]?.share).toBe(0.5);
  });

  it("crosses month and year ends in UTC", () => {
    const filled = fillDays([day("2026-12-31", 1, 1), day("2027-01-02", 1, 0)]);

    expect(filled.map((d) => d.date)).toEqual(["2026-12-31", "2027-01-01", "2027-01-02"]);
  });

  it("returns an empty series unchanged", () => {
    expect(fillDays([])).toEqual([]);
  });
});

describe("placePatchMarkers", () => {
  it("places_patch_markers_on_their_day", () => {
    const dates = ["2026-09-01", "2026-09-02", "2026-09-03"];

    const markers = placePatchMarkers(
      [
        patch("2026-08-31", "before the data"),
        patch("2026-09-02", "1.1"),
        patch("2026-09-02", "dev blog", false),
        patch("2026-09-03", "hotfix", false),
        patch("2026-09-04", "after the data"),
      ],
      dates,
    );

    expect(markers).toEqual([
      { date: "2026-09-02", titles: ["1.1", "dev blog"], treatment: true },
      { date: "2026-09-03", titles: ["hotfix"], treatment: false },
    ]);
  });
});

describe("saleRanges", () => {
  it("joins consecutive sale days into one range", () => {
    const series = fillDays([
      day("2026-09-01", 1, 1),
      day("2026-09-02", 1, 1, { sale: true }),
      day("2026-09-03", 1, 1, { sale: true }),
      day("2026-09-04", 1, 1),
      day("2026-09-05", 1, 1, { sale: true }),
    ]);

    expect(saleRanges(series)).toEqual([
      { from: "2026-09-02", to: "2026-09-03" },
      { from: "2026-09-05", to: "2026-09-05" },
    ]);
  });
});

describe("rollingShare", () => {
  it("pools reviews over the trailing window instead of averaging daily shares", () => {
    const series = [day("2026-09-01", 100, 90), day("2026-09-02", 2, 0), day("2026-09-03", 0, 0)];

    const rolled = rollingShare(series, 2);

    expect(rolled.map((p) => p.share)).toEqual([0.9, 90 / 102, 0]);
  });

  it("is null while the window has no reviews", () => {
    expect(rollingShare([day("2026-09-01", 0, 0)], 7)[0]?.share).toBeNull();
  });
});

describe("lastDays", () => {
  it("keeps the last N calendar days, or everything for null", () => {
    const series = fillDays([day("2026-09-01", 1, 1), day("2026-09-10", 1, 1)]);

    expect(lastDays(series, 3).map((d) => d.date)).toEqual([
      "2026-09-08",
      "2026-09-09",
      "2026-09-10",
    ]);
    expect(lastDays(series, null)).toHaveLength(10);
  });
});

describe("dateTicks", () => {
  const span = (from: string, to: string) =>
    fillDays([day(from, 1, 1), day(to, 1, 1)]).map((d) => d.date);

  it("marks the 1st and 15th over a few months, so both charts share their ticks", () => {
    expect(dateTicks(span("2026-07-01", "2026-09-28"))).toEqual([
      "2026-07-01",
      "2026-07-15",
      "2026-08-01",
      "2026-08-15",
      "2026-09-01",
      "2026-09-15",
    ]);
  });

  it("marks each month over a year, and each quarter beyond", () => {
    expect(dateTicks(span("2025-10-02", "2026-09-28"))).toHaveLength(11);
    expect(dateTicks(span("2024-01-10", "2026-09-28"))).toEqual([
      "2024-04-01",
      "2024-07-01",
      "2024-10-01",
      "2025-01-01",
      "2025-04-01",
      "2025-07-01",
      "2025-10-01",
      "2026-01-01",
      "2026-04-01",
      "2026-07-01",
    ]);
  });
});

describe("formatting", () => {
  it("shows shares as whole percentages and dates in day month year", () => {
    expect(formatShare(0.7632)).toBe("76%");
    expect(formatShare(null)).toBe("–");
    expect(formatDate("2026-09-02")).toBe("2 Sep 2026");
  });
});
