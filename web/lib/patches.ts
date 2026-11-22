// The game page's patch list (design F7): tested patches grouped by their estimate window, newest
// first. Patches within 7 days of the first share one estimate and one report (the exporter's
// bundles), so they share one row.

import { type EstimatedHorizon, points, shownHorizon } from "./effects";
import { joinWords } from "./reports";
import { formatDate } from "./series";
import type { Patch, PatchEffect } from "./types";

export type Verdict = "up" | "down" | "unclear" | "none";

export const VERDICT_LABELS: Record<Verdict, string> = {
  up: "Clearly up",
  down: "Clearly down",
  unclear: "No clear change",
  none: "No estimate",
};

export const FIRST_ROWS = 8; // rows shown before "Show 10 more"
export const MORE_ROWS = 10;
export const SCALE_LIMIT = 100; // the shared scale runs from −100 to +100 points

export type PatchWindow = {
  start: string; // the window's first day
  patches: Patch[]; // oldest first
  title: string;
  effect: PatchEffect | null;
  horizon: EstimatedHorizon | null; // the estimate the row shows, as the sentence states it
  verdict: Verdict;
  report: string | null;
};

/** Clear only when the whole 90% range is on one side of zero; touching zero isn't clear. */
export function verdictOf(horizon: EstimatedHorizon | null): Verdict {
  if (horizon === null) return "none";
  if (horizon.lo90 > 0) return "up";
  if (horizon.hi90 < 0) return "down";
  return "unclear";
}

/** "Devoid of Liberty: 7.2.0 and 7.2.1" for titles that differ only in their last word;
 * otherwise the first title and how many more. */
export function windowTitle(titles: readonly string[]): string {
  const [first] = titles;
  if (first === undefined) return "";
  if (titles.length === 1) return first;
  const head = (title: string) => {
    const space = title.lastIndexOf(" ");
    return space > 0 ? title.slice(0, space) : "";
  };
  const shared = head(first);
  if (shared !== "" && titles.every((title) => head(title) === shared)) {
    return `${shared} ${joinWords(titles.map((title) => title.slice(shared.length + 1)))}`;
  }
  return `${first} and ${titles.length - 1} more`;
}

/** Tested patches by estimate window, newest first; a patch without an estimate stands alone. */
export function patchWindows(patches: readonly Patch[]): PatchWindow[] {
  const groups = new Map<string, Patch[]>();
  for (const patch of patches) {
    if (!patch.treatment) continue;
    const key = patch.effect?.start ?? patch.date;
    groups.set(key, [...(groups.get(key) ?? []), patch]);
  }
  return [...groups.entries()]
    .map(([start, group]) => {
      const sorted = [...group].sort((a, b) => a.date.localeCompare(b.date));
      const effect = sorted.find((patch) => patch.effect)?.effect ?? null;
      const horizon = shownHorizon(effect);
      return {
        start,
        patches: sorted,
        title: windowTitle(sorted.map((patch) => patch.title)),
        effect,
        horizon,
        verdict: verdictOf(horizon),
        report: sorted.find((patch) => patch.report)?.report ?? null,
      };
    })
    .sort((a, b) => b.start.localeCompare(a.start));
}

/** A value on the shared scale, and whether it was cut to fit. */
export function clampToScale(value: number): { value: number; clipped: boolean } {
  const clamped = Math.max(-SCALE_LIMIT, Math.min(SCALE_LIMIT, value));
  return { value: clamped, clipped: clamped !== value };
}

/** The row's accessible name: what a sighted reader gets from its date, title and dot. */
export function windowName(entry: PatchWindow): string {
  const head = `${formatDate(entry.start)}, ${entry.title}.`;
  const { horizon } = entry;
  if (horizon === null) return `${head} ${VERDICT_LABELS.none}`;
  return (
    `${head} ${VERDICT_LABELS[entry.verdict]}: ${points(horizon.effect)} points, ` +
    `90% range ${points(horizon.lo90)} to ${points(horizon.hi90)}, over ${horizon.days} days`
  );
}
