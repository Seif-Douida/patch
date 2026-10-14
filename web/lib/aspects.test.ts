import { describe, expect, it } from "vitest";

import {
  MIN_COMPLAINT_SUPPORT,
  MIN_WEEK_SCORED,
  MIN_WINDOW_SCORED,
  beforeAfter,
  weakAspects,
  newcombeInterval,
  topComplaint,
  weeklyAspectShares,
} from "./aspects";
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

describe("newcombeInterval", () => {
  it("newcombe_interval_matches_published_values", () => {
    // Newcombe (1998), "Interval estimation for the difference between independent
    // proportions: comparison of eleven methods", Statistics in Medicine 17, 873-890,
    // Table II, method 10 (the hybrid score interval, Wilson-based, no continuity correction).
    const published: [number, number, number, number, number, number][] = [
      [56, 70, 48, 80, 0.0524, 0.3339],
      [9, 10, 3, 10, 0.1705, 0.809],
      [5, 56, 0, 29, -0.0381, 0.1926],
      [0, 10, 0, 20, -0.1611, 0.2775],
      [10, 10, 0, 20, 0.6791, 1],
      [10, 10, 0, 10, 0.6075, 1],
    ];
    for (const [x1, n1, x2, n2, low, high] of published) {
      const [lo, hi] = newcombeInterval(x1, n1, x2, n2);
      expect(lo).toBeCloseTo(low, 4);
      expect(hi).toBeCloseTo(high, 4);
    }
  });
});

describe("beforeAfter", () => {
  // 1 Sep to 30 Sep 2026, one day at a time: `before` before the patch, `after` from its day.
  function around(patch: string, before: AspectDay, after: AspectDay): AspectSeries {
    const days: AspectDay[] = [];
    for (let d = 1; d <= 30; d += 1) {
      const date = `2026-09-${String(d).padStart(2, "0")}`;
      days.push({ ...(date < patch ? before : after), date });
    }
    return series(days);
  }

  it("compares_the_14_days_before_with_the_patch_day_and_13_after", () => {
    // 2 performance mentions in 10 negative reviews a day before; 4 in 10 from the patch day.
    const data = around(
      "2026-09-16",
      day("", [10, 5], { performance: 2 }),
      day("", [10, 5], { performance: 4 }),
    );

    const result = beforeAfter(data, "2026-09-16");

    expect(result.before).toEqual({ days: 14, n: 140 });
    expect(result.after).toEqual({ days: 14, n: 140 });
    expect(result.shortBefore).toBe(false);
    const [performance] = result.aspects;
    expect(performance?.code).toBe("performance");
    expect(performance?.before).toBeCloseTo(0.2);
    expect(performance?.after).toBeCloseTo(0.4);
    expect(performance?.change).toBeCloseTo(0.2);
    expect(performance?.interval).toEqual(newcombeInterval(56, 140, 28, 140));
    expect(result.aspects[1]?.change).toBe(0);
  });

  it("patch_too_close_to_the_data_start_is_flagged", () => {
    const data = around("2026-09-08", day("", [10, 0]), day("", [10, 0]));

    const early = beforeAfter(data, "2026-09-08");
    expect(early.shortBefore).toBe(true);
    expect(early.before.days).toBe(7);
    expect(beforeAfter(data, "2026-09-15").shortBefore).toBe(false);
    // A patch in the last 13 days of data has an incomplete after window too.
    const recent = beforeAfter(data, "2026-09-25");
    expect(recent.shortAfter).toBe(true);
    expect(recent.after.days).toBe(6);
  });

  it("aspects_with_too_few_reviews_show_no_numbers", () => {
    // 2 scored negative reviews a day: 28 before and 28 after, under MIN_WINDOW_SCORED.
    expect(14 * 2).toBeLessThan(MIN_WINDOW_SCORED);
    const data = around(
      "2026-09-16",
      day("", [2, 50], { performance: 1 }),
      day("", [2, 50], { performance: 2 }),
    );

    const result = beforeAfter(data, "2026-09-16");

    expect(result.enough).toBe(false);
    for (const aspect of result.aspects) {
      expect([aspect.before, aspect.after, aspect.change, aspect.interval]).toEqual([
        null,
        null,
        null,
        null,
      ]);
    }
  });
});

describe("weakAspects", () => {
  it("names_aspects_under_the_f1_bar", () => {
    const model = {
      family: "s0",
      version: "v1",
      test_macro_f1: 0.61,
      aspect_f1: { performance: 0.87, ux_controls: 0.48, platform_policy: 0.18 },
    };
    expect([...weakAspects(model)].sort()).toEqual(["platform_policy", "ux_controls"]);
    expect(weakAspects({ ...model, aspect_f1: undefined }).size).toBe(0);
    expect(weakAspects(null).size).toBe(0);
  });
});
