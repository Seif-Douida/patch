// Aspect counts (schema 2) turned into what the heatmap and the home page show. Shares are always
// pooled - mentions summed over scored reviews summed - never an average of daily shares.

import type { AspectSeries, GameSummary } from "./types";

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
