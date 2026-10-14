// Aspect counts (schema 2) turned into what the heatmap and the home page show. Shares are always
// pooled - mentions summed over scored reviews summed - never an average of daily shares.

import type { AspectModel, AspectSeries, GameSummary } from "./types";

export type Polarity = "neg" | "pos";

/** A week with fewer scored reviews of the shown polarity gets no share (hatched). */
export const MIN_WEEK_SCORED = 10;

/** The home page names a top complaint only over at least this many scored negative reviews. */
export const MIN_COMPLAINT_SUPPORT = 30;

/** Upper bounds of the heatmap's colour levels 1-4; level 5 is anything above. Fixed, so a
 * share has the same colour in every game and for both polarities. */
export const SHARE_LEVELS = [0.05, 0.1, 0.2, 0.3] as const;

export const ASPECT_LABELS: Record<string, string> = {
  performance: "Performance",
  stability_bugs: "Stability and bugs",
  gameplay_balance: "Gameplay and balance",
  content: "Content",
  monetization: "Monetization",
  online_servers: "Online and servers",
  story_world: "Story and world",
  ux_controls: "UX and controls",
  platform_policy: "Platform and policy",
  off_topic: "Off topic",
};

export function aspectLabel(code: string): string {
  return ASPECT_LABELS[code] ?? code;
}

/** An aspect the model detects with a gold-test F1 under this is marked as less reliable (the
 * exporter's MIN_ASPECT_F1, which keeps it out of the top complaint). */
export const MIN_ASPECT_F1 = 0.5;

/** The aspects the model detects less reliably; none when it records no per-aspect F1. */
export function weakAspects(model: AspectModel | null | undefined): Set<string> {
  const f1 = model?.aspect_f1 ?? {};
  return new Set(Object.keys(f1).filter((code) => (f1[code] ?? 1) < MIN_ASPECT_F1));
}

export const WEAK_NOTE = `* Detected less reliably: F1 under ${MIN_ASPECT_F1} on our test set.`;

/** The colour level (1-5) of a share; null for a week without one. */
export function shareLevel(share: number | null): number | null {
  if (share === null) return null;
  const below = SHARE_LEVELS.findIndex((bound) => share < bound);
  return below === -1 ? SHARE_LEVELS.length + 1 : below + 1;
}

export type Week = {
  start: string; // the Monday, "2026-10-05"
  n: number; // scored reviews of the polarity
  counts: number[]; // mentions per aspect, in `codes` order
  shares: (number | null)[]; // counts / n; null when n < MIN_WEEK_SCORED
};

const DAY_MS = 86_400_000;

function toTime(date: string): number {
  return Date.parse(`${date}T00:00:00Z`);
}

function toDate(time: number): string {
  return new Date(time).toISOString().slice(0, 10);
}

function monday(date: string): number {
  const time = toTime(date);
  return time - ((new Date(time).getUTCDay() + 6) % 7) * DAY_MS;
}

/** Monday-to-Sunday weeks covering `from`..`to`, every week present so the columns stay on the
 * calendar; only days inside the range count. */
export function weeklyAspectShares(
  series: AspectSeries,
  polarity: Polarity,
  range: { from: string; to: string },
): Week[] {
  const size = series.codes.length;
  const weeks = new Map<string, { n: number; counts: number[] }>();
  for (let time = monday(range.from); time <= toTime(range.to); time += 7 * DAY_MS) {
    weeks.set(toDate(time), { n: 0, counts: new Array<number>(size).fill(0) });
  }
  for (const day of series.days) {
    if (day.date < range.from || day.date > range.to) continue;
    const week = weeks.get(toDate(monday(day.date)));
    if (!week) continue;
    week.n += polarity === "neg" ? day.scored[0] : day.scored[1];
    const mentions = polarity === "neg" ? day.neg : day.pos;
    mentions.forEach((count, i) => {
      week.counts[i] = (week.counts[i] ?? 0) + count;
    });
  }
  return [...weeks].map(([start, { n, counts }]) => ({
    start,
    n,
    counts,
    shares: counts.map((count) => (n >= MIN_WEEK_SCORED ? count / n : null)),
  }));
}

export type Complaint = { code: string; label: string; share: number; n: number };

/** The game's top complaint over the last 30 days, or null without enough scored negatives. */
export function topComplaint(entry: GameSummary): Complaint | null {
  const top = entry.top_complaint_30d;
  if (!top || top.n < MIN_COMPLAINT_SUPPORT) return null;
  return { code: top.code, label: aspectLabel(top.code), share: top.share, n: top.n };
}

/** Each side of a before/after comparison needs this many scored negative reviews. */
export const MIN_WINDOW_SCORED = 30;
export const WINDOW_DAYS = 14;
const Z95 = 1.959963984540054;

function wilson(x: number, n: number): [number, number] {
  const p = x / n;
  const centre = p + (Z95 * Z95) / (2 * n);
  const spread = Z95 * Math.sqrt((p * (1 - p)) / n + (Z95 * Z95) / (4 * n * n));
  const denominator = 1 + (Z95 * Z95) / n;
  return [(centre - spread) / denominator, (centre + spread) / denominator];
}

/** 95% interval for x1/n1 - x2/n2 by Newcombe's hybrid score method (1998, method 10): each
 * proportion's Wilson interval, combined. Well behaved at 0%, 100% and small n. */
export function newcombeInterval(x1: number, n1: number, x2: number, n2: number): [number, number] {
  const p1 = x1 / n1;
  const p2 = x2 / n2;
  const [l1, u1] = wilson(x1, n1);
  const [l2, u2] = wilson(x2, n2);
  const difference = p1 - p2;
  return [
    difference - Math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
    difference + Math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2),
  ];
}

export type Window = { days: number; n: number }; // days of data in the window, scored negatives

export type AspectChange = {
  code: string;
  label: string;
  before: number | null; // share of scored negative reviews mentioning it
  after: number | null;
  change: number | null; // after - before
  interval: [number, number] | null; // 95%, for the change
};

export type PatchComparison = {
  before: Window;
  after: Window;
  shortBefore: boolean; // fewer than `days` days of data before the patch
  shortAfter: boolean; // the after window isn't over yet
  enough: boolean; // both windows have MIN_WINDOW_SCORED scored negative reviews
  aspects: AspectChange[];
};

/** Negative reviews' aspect shares in the `days` days before a patch against its day and the
 * `days - 1` after (the windows of mart.patch_window). Descriptive, not causal. */
export function beforeAfter(
  series: AspectSeries,
  patchDate: string,
  days: number = WINDOW_DAYS,
): PatchComparison {
  const size = series.codes.length;
  const patch = toTime(patchDate);
  const first = series.days[0];
  const last = series.days.at(-1);
  function covered(from: number, to: number): number {
    if (!first || !last) return 0;
    const start = Math.max(from, toTime(first.date));
    const end = Math.min(to, toTime(last.date));
    return end < start ? 0 : (end - start) / DAY_MS + 1;
  }
  function window(from: number, to: number) {
    let n = 0;
    const mentions = new Array<number>(size).fill(0);
    for (const day of series.days) {
      const time = toTime(day.date);
      if (time < from || time > to) continue;
      n += day.scored[0];
      day.neg.forEach((count, i) => {
        mentions[i] = (mentions[i] ?? 0) + count;
      });
    }
    return { n, mentions };
  }
  const beforeFrom = patch - days * DAY_MS;
  const afterTo = patch + (days - 1) * DAY_MS;
  const before = window(beforeFrom, patch - DAY_MS);
  const after = window(patch, afterTo);
  const beforeDays = covered(beforeFrom, patch - DAY_MS);
  const afterDays = covered(patch, afterTo);
  const enough = before.n >= MIN_WINDOW_SCORED && after.n >= MIN_WINDOW_SCORED;
  return {
    before: { days: beforeDays, n: before.n },
    after: { days: afterDays, n: after.n },
    shortBefore: beforeDays < days,
    shortAfter: afterDays < days,
    enough,
    aspects: series.codes.map((code, i) => {
      const label = aspectLabel(code);
      const empty = { code, label, before: null, after: null, change: null, interval: null };
      if (!enough) return empty;
      const x1 = after.mentions[i] ?? 0;
      const x0 = before.mentions[i] ?? 0;
      return {
        ...empty,
        before: x0 / before.n,
        after: x1 / after.n,
        change: x1 / after.n - x0 / before.n,
        interval: newcombeInterval(x1, after.n, x0, before.n),
      };
    }),
  };
}
