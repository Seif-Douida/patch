import { describe, expect, it } from "vitest";

import { MIN_COMPLAINT_SUPPORT, MIN_WEEK_SCORED, topComplaint, weeklyAspectShares } from "./aspects";
import type { AspectDay, AspectSeries, GameSummary } from "./types";

const CODES = [
  "performance",
  "stability_bugs",
  "gameplay_balance",
  "content",
  "monetization",
  "online_servers",
  "story_world",
  "ux_controls",
  "platform_policy",
  "off_topic",
];

function counts(values: Partial<Record<string, number>>): number[] {
  return CODES.map((code) => values[code] ?? 0);
}

function day(date: string, scored: [number, number], neg = {}, pos = {}): AspectDay {
  return { date, scored, neg: counts(neg), pos: counts(pos) };
}

function series(days: AspectDay[]): AspectSeries {
  return { codes: CODES, days };
}

describe("weeklyAspectShares", () => {
  it("weekly_shares_sum_counts_not_shares", () => {
    // Monday 5 Oct 2026: 2 of 10 negative reviews mention performance; Tuesday: 20 of 20.
    // Averaging the two days' shares would give 60%; the week's share is 22 of 30.
    const weeks = weeklyAspectShares(
      series([
        day("2026-10-05", [10, 0], { performance: 2 }),
        day("2026-10-06", [20, 0], { performance: 20 }),
      ]),
      "neg",
      { from: "2026-10-05", to: "2026-10-11" },
    );

    expect(weeks).toHaveLength(1);
    const [week] = weeks;
    expect(week?.start).toBe("2026-10-05");
    expect(week?.n).toBe(30);
    expect(week?.counts[0]).toBe(22);
    expect(week?.shares[0]).toBeCloseTo(22 / 30);
    expect(week?.shares[1]).toBe(0);
  });

  it("thin_weeks_are_null", () => {
    // Weeks run Monday to Sunday over the whole range, so the columns stay on the calendar:
    // a week with nothing scored, or fewer than MIN_WEEK_SCORED of that polarity, has no share.
    const weeks = weeklyAspectShares(
      series([
        day("2026-09-30", [MIN_WEEK_SCORED - 1, 40], { content: 3 }, { content: 10 }),
        day("2026-10-14", [MIN_WEEK_SCORED, 0], { content: 5 }),
      ]),
      "neg",
      { from: "2026-09-28", to: "2026-10-18" },
    );

    expect(weeks.map((w) => w.start)).toEqual(["2026-09-28", "2026-10-05", "2026-10-12"]);
    expect(weeks.map((w) => w.n)).toEqual([MIN_WEEK_SCORED - 1, 0, MIN_WEEK_SCORED]);
    expect(weeks[0]?.shares.every((share) => share === null)).toBe(true);
    expect(weeks[1]?.shares.every((share) => share === null)).toBe(true);
    expect(weeks[2]?.shares[3]).toBeCloseTo(0.5);
    // The same week has enough positive reviews.
    const positive = weeklyAspectShares(
      series([day("2026-09-30", [MIN_WEEK_SCORED - 1, 40], {}, { content: 10 })]),
      "pos",
      { from: "2026-09-28", to: "2026-10-04" },
    );
    expect(positive[0]?.shares[3]).toBeCloseTo(0.25);
  });

  it("days_outside_the_range_are_left_out", () => {
    const weeks = weeklyAspectShares(
      series([
        day("2026-10-04", [50, 0], { performance: 50 }),
        day("2026-10-07", [20, 0], { performance: 5 }),
      ]),
      "neg",
      { from: "2026-10-07", to: "2026-10-09" },
    );

    expect(weeks.map((w) => [w.start, w.n])).toEqual([["2026-10-05", 20]]);
    expect(weeks[0]?.shares[0]).toBeCloseTo(0.25);
  });
});

describe("topComplaint", () => {
  function entry(top: GameSummary["top_complaint_30d"]): GameSummary {
    return {
      appid: 1,
      name: "Example",
      genres: [],
      first_date: null,
      last_date: null,
      latest: null,
      top_complaint_30d: top,
    };
  }

  it("top_complaint_needs_minimum_support", () => {
    const enough = topComplaint(
      entry({ code: "stability_bugs", share: 0.31, n: MIN_COMPLAINT_SUPPORT }),
    );
    expect(enough).toEqual({
      code: "stability_bugs",
      label: "Stability and bugs",
      share: 0.31,
      n: MIN_COMPLAINT_SUPPORT,
    });
    expect(
      topComplaint(entry({ code: "performance", share: 0.9, n: MIN_COMPLAINT_SUPPORT - 1 })),
    ).toBeNull();
    expect(topComplaint(entry(null))).toBeNull();
    const schemaOne = entry(null);
    delete schemaOne.top_complaint_30d;
    expect(topComplaint(schemaOne)).toBeNull();
  });
});
