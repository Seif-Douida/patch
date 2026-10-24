// A treatment patch's estimated effect on the positive share (schema 3) as one sentence and
// short notes, for the line above its before-and-after table.

import { formatDate } from "./series";
import type { CausalMethod, HorizonEffect, PatchEffect } from "./types";

export const METHOD_NAMES: Record<CausalMethod, string> = {
  its: "interrupted time series",
  synth: "synthetic control",
};

export type EffectText = { line: string; notes: string[] };

/** Percentage points with one decimal and their sign: "+1.1", "−4.1", "0.0". */
export function points(value: number): string {
  const text = Math.abs(value).toFixed(1);
  if (text === "0.0") return text;
  return value > 0 ? `+${text}` : `−${text}`;
}

function capitalised(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

// The estimators' terse reasons in plain words; the others are already written for readers
// ("too few comparison games without their own patch (3 of 5 needed)").
const PLAIN: readonly (readonly [string, string])[] = [
  ["too few reviews", "too few reviews (under 10 a day)"],
  ["pre-period under 28 days", "less than 28 days of reviews before the patch"],
  ["too little quiet history", "too little quiet history before the patch to check the range"],
  ["too few days with data", "too few days of data before the patch"],
  ["singular design", "the model couldn't be fitted"],
  ["non-finite result", "the model couldn't be fitted"],
  ["interval wider than the grid", "the 90% range was too wide to state"],
  ["interval empty", "the comparison games didn't fit the days after the patch"],
  ["window not complete", "the days after the patch haven't ended"],
];

export function plainReason(reason: string): string {
  return PLAIN.find(([start]) => reason.startsWith(start))?.[1] ?? reason;
}

type Estimated = HorizonEffect & {
  effect: number;
  lo90: number;
  hi90: number;
  method: CausalMethod;
};

// An effect is only shown with its range.
function isEstimated(horizon: HorizonEffect | undefined): horizon is Estimated {
  return (
    horizon !== undefined &&
    horizon.effect !== null &&
    horizon.lo90 !== null &&
    horizon.hi90 !== null &&
    horizon.method !== null
  );
}

function absence(horizon: HorizonEffect | undefined): string {
  if (horizon === undefined) return "No estimate.";
  switch (horizon.reason) {
    case "window_not_complete":
      return `No estimate yet: the ${horizon.days} days after the patch end on ${formatDate(horizon.ends)}.`;
    case "not_estimated_yet":
      return "No estimate yet: the nightly update hasn't reached this patch.";
    case "another_patch_within_14_days":
      return "No estimate: another patch came out within 14 days.";
    case null:
      return "No estimate.";
    default:
      return `No estimate: ${plainReason(horizon.reason)}.`;
  }
}

function fourteenDays(horizon: HorizonEffect): string | null {
  switch (horizon.reason) {
    case "window_not_complete":
      return `The 14-day estimate comes after ${formatDate(horizon.ends)}.`;
    case "not_estimated_yet":
      return "The 14-day estimate is waiting for the nightly update.";
    case "another_patch_within_14_days":
      return "No 14-day estimate: another patch came out within 14 days.";
    case null:
      return null;
    default:
      return `No 14-day estimate: ${plainReason(horizon.reason)}.`;
  }
}

function flagNotes(flags: readonly string[], discountsFrom: string | null): string[] {
  const notes: string[] = [];
  if (flags.includes("confounded")) {
    notes.push(
      "A Steam sale or a discount fell within 3 days of the patch, so part of the change may " +
        "be the price.",
    );
  }
  if (flags.includes("discounts_unknown")) {
    notes.push(
      discountsFrom === null
        ? "Game discounts around the patch aren't known."
        : `Game discounts before ${formatDate(discountsFrom)} aren't known.`,
    );
  }
  if (flags.includes("volume_spike")) notes.push("Reviews spiked after the patch.");
  return notes; // `binned` (quiet days grouped in threes) changes nothing a reader acts on
}

/**
 * The 14-day estimate, or the 7-day one when there's no 14-day estimate; without either, why.
 * `discountsFrom` is the first day the game's price is known (its first price snapshot).
 */
export function describeEffect(
  effect: PatchEffect,
  options: { patchDate: string; defaultMethod: CausalMethod | null; discountsFrom: string | null },
): EffectText {
  const seven = effect.horizons.find((horizon) => horizon.days === 7);
  const fourteen = effect.horizons.find((horizon) => horizon.days === 14);
  const shown = isEstimated(fourteen) ? fourteen : isEstimated(seven) ? seven : null;
  const notes: string[] = [];
  let line: string;
  if (shown === null) {
    line = absence(seven ?? effect.horizons[0]);
  } else {
    line =
      `Estimated effect on positive reviews over ${shown.days} days: ${points(shown.effect)} ` +
      `points (90% range ${points(shown.lo90)} to ${points(shown.hi90)}), ` +
      `${METHOD_NAMES[shown.method]}.`;
    const { defaultMethod } = options;
    if (defaultMethod !== null && shown.method !== defaultMethod && shown.reason !== null) {
      notes.push(
        `${capitalised(METHOD_NAMES[defaultMethod])} wasn't possible: ${plainReason(shown.reason)}.`,
      );
    }
    notes.push(...flagNotes(shown.flags, options.discountsFrom));
    const later = shown.days === 7 && fourteen !== undefined ? fourteenDays(fourteen) : null;
    if (later !== null) notes.push(later);
  }
  if (effect.start !== options.patchDate) {
    notes.push(
      `Estimated together with the patch of ${formatDate(effect.start)}: patches within 7 days ` +
        "count as one change.",
    );
  }
  return { line, notes };
}
