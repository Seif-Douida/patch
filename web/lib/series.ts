// Pure helpers that turn the site data into what the charts draw. Dates are ISO "YYYY-MM-DD"
// strings and all date arithmetic is in UTC, so a day never shifts with the viewer's timezone.

import type { DayPoint, Patch } from "./types";

const DAY_MS = 86_400_000;

function toTime(date: string): number {
  return Date.parse(`${date}T00:00:00Z`);
}

function toDate(time: number): string {
  return new Date(time).toISOString().slice(0, 10);
}

/** Every calendar day from the first to the last point; a missing day has no reviews. */
export function fillDays(series: readonly DayPoint[]): DayPoint[] {
  const first = series[0];
  const last = series.at(-1);
  if (!first || !last) return [];
  const byDate = new Map(series.map((point) => [point.date, point]));
  const filled: DayPoint[] = [];
  for (let time = toTime(first.date); time <= toTime(last.date); time += DAY_MS) {
    const date = toDate(time);
    filled.push(
      byDate.get(date) ?? {
        date,
        n: 0,
        pos: 0,
        share: null,
        lo: null,
        hi: null,
        sale: false,
        discount: null,
      },
    );
  }
  return filled;
}

/** The last `days` calendar days of a filled series; null keeps it all. */
export function lastDays<T>(series: readonly T[], days: number | null): T[] {
  return days === null ? [...series] : series.slice(-days);
}

export type Marker = { date: string; titles: string[]; treatment: boolean };

/** One marker per day that has posts inside the plotted dates; a treatment day wins. */
export function placePatchMarkers(patches: readonly Patch[], dates: readonly string[]): Marker[] {
  const plotted = new Set(dates);
  const byDate = new Map<string, Marker>();
  for (const patch of patches) {
    if (!plotted.has(patch.date)) continue;
    const marker = byDate.get(patch.date) ?? { date: patch.date, titles: [], treatment: false };
    marker.titles.push(patch.title);
    marker.treatment ||= patch.treatment;
    byDate.set(patch.date, marker);
  }
  return [...byDate.values()].sort((a, b) => a.date.localeCompare(b.date));
}

export type DateRange = { from: string; to: string };

/** Consecutive sale days, joined into ranges (the series must be gap-filled). */
export function saleRanges(series: readonly DayPoint[]): DateRange[] {
  const ranges: DateRange[] = [];
  for (const point of series) {
    const open = ranges.at(-1);
    if (!point.sale) continue;
    if (open && toTime(point.date) - toTime(open.to) === DAY_MS) open.to = point.date;
    else ranges.push({ from: point.date, to: point.date });
  }
  return ranges;
}

export type SharePoint = { date: string; share: number | null };

/** Positive share pooled over a trailing window of days (reviews summed, not shares averaged). */
export function rollingShare(series: readonly DayPoint[], window: number): SharePoint[] {
  return series.map((point, index) => {
    const span = series.slice(Math.max(0, index - window + 1), index + 1);
    const n = span.reduce((sum, d) => sum + d.n, 0);
    const pos = span.reduce((sum, d) => sum + d.pos, 0);
    return { date: point.date, share: n ? pos / n : null };
  });
}

/** Calendar ticks both charts share: the 1st and 15th, each month, or each quarter by span. */
export function dateTicks(dates: readonly string[]): string[] {
  const keep =
    dates.length <= 120
      ? (date: string) => date.endsWith("-01") || date.endsWith("-15")
      : dates.length <= 400
        ? (date: string) => date.endsWith("-01")
        : (date: string) => /-(01|04|07|10)-01$/.test(date);
  return dates.filter(keep);
}

/** "1 Sep" on a short range; "Sep 2026" once ticks are monthly or wider. */
export function tickLabel(date: string, span: number): string {
  const [year, month, day] = date.split("-").map(Number);
  const name = MONTHS[(month ?? 1) - 1];
  return span <= 120 ? `${day} ${name}` : `${name} ${year}`;
}

export function formatShare(share: number | null): string {
  return share === null ? "–" : `${Math.round(share * 100)}%`;
}

// Spelled out rather than Intl: browsers' locale data differs from Node's ("Sep" or "Sept"), and
// the static HTML must match what the browser renders.
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export function formatDate(date: string): string {
  const [year, month, day] = date.split("-").map(Number);
  return `${day} ${MONTHS[(month ?? 1) - 1]} ${year}`;
}

const COUNT_FORMAT = new Intl.NumberFormat("en-GB");

export function formatCount(n: number): string {
  return COUNT_FORMAT.format(n);
}

export const PATCH_LABELS: Record<Patch["type"], string> = {
  major_update: "Major update",
  patch: "Patch",
  hotfix: "Hotfix",
  event: "Event",
  marketing: "Announcement",
  other: "Other post",
};
