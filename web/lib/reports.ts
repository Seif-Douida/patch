// Plain words for a report's parts (schema 4): the five explanations, their statuses, the
// verifier's rules, and how the claims are filed. The agent's own titles are in
// src/patchpulse/agent/verifier.py (HYPOTHESIS_TITLES); `long` matches them.

import { formatDate } from "./series";
import type { Hypothesis, HypothesisCode, Patch, PublicReport, ReportClaim } from "./types";

export const HYPOTHESIS_ORDER: readonly HypothesisCode[] = ["H1", "H2", "H3", "H4", "H5"];

// `short` fits a tile; `long` heads the explanation's claims.
export const EXPLANATIONS: Record<HypothesisCode, { short: string; long: string }> = {
  H1: { short: "The patch", long: "The patch" },
  H2: { short: "Price or sale", long: "A price change or sale" },
  H3: { short: "Off-topic", long: "Off-topic reviews or a review bomb" },
  H4: { short: "Who reviewed", long: "A change in who reviewed" },
  H5: { short: "Noise", long: "Noise or seasonality" },
};

// The rules set each status from the data, never the model.
export const STATUS_LABELS: Record<Hypothesis["status"], string> = {
  supported: "Supported",
  rejected: "Ruled out",
  inconclusive: "Inconclusive",
};

export const CONFIDENCE_LABELS: Record<PublicReport["confidence"], string> = {
  low: "Low confidence",
  medium: "Medium confidence",
  high: "High confidence",
};

// What each verifier rule stops, worded to follow "1 claim" or "3 claims". V6 checks the summary,
// so it never withholds a claim.
const WITHHELD_KINDS: Record<string, string> = {
  V1: "citing evidence the analysis never produced",
  V2: "with a number that didn't match the data",
  V3: "with a quote that didn't match its review",
  V4: "of cause that the estimate didn't support",
  V5: "that implied a cause without an estimate behind it",
  V7: "that repeated a review's own words",
};

function claims(n: number): string {
  return `${n} ${n === 1 ? "claim" : "claims"}`;
}

/** "2 claims with a number that didn't match the data", in rule order. */
export function withheldLines(withheld: Record<string, number>): string[] {
  return Object.entries(withheld)
    .filter(([, n]) => n > 0)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([rule, n]) => `${claims(n)} ${WITHHELD_KINDS[rule] ?? "the verifier stopped"}`);
}

export function withheldTotal(withheld: Record<string, number>): number {
  return Object.values(withheld).reduce((sum, n) => sum + n, 0);
}

/** "supported by 1 review"; nothing for a claim that cites no reviews. */
export function supportLine(n: number): string | null {
  if (n <= 0) return null;
  return `supported by ${n} ${n === 1 ? "review" : "reviews"}`;
}

export type FiledClaims = {
  hypothesis: Hypothesis;
  claims: ReportClaim[];
};

/**
 * The verified claims under the explanation each backs, in hypothesis order; a claim filed under
 * two shows under the first. Claims no finding files are returned apart, to show at the end.
 */
export function fileClaims(report: PublicReport): { filed: FiledClaims[]; unfiled: ReportClaim[] } {
  const byId = new Map(report.claims.map((c) => [c.claim_id, c]));
  const placed = new Set<string>();
  const filed: FiledClaims[] = [];
  for (const code of HYPOTHESIS_ORDER) {
    const hypothesis = report.hypotheses.find((h) => h.code === code);
    if (!hypothesis) continue;
    const found: ReportClaim[] = [];
    for (const finding of report.findings.filter((f) => f.hypothesis === code)) {
      for (const id of finding.claim_ids) {
        const claim = byId.get(id);
        if (claim && !placed.has(id)) {
          placed.add(id);
          found.push(claim);
        }
      }
    }
    if (found.length > 0) filed.push({ hypothesis, claims: found });
  }
  return { filed, unfiled: report.claims.filter((c) => !placed.has(c.claim_id)) };
}

export function orderedHypotheses(report: PublicReport): Hypothesis[] {
  return HYPOTHESIS_ORDER.flatMap((code) => report.hypotheses.filter((h) => h.code === code));
}

/** "2 Jul to 15 Jul 2026", or with both years when they differ. */
export function formatRange(start: string, end: string): string {
  const [first, last] = [formatDate(start), formatDate(end)];
  if (start.slice(0, 4) !== end.slice(0, 4)) return `${first} to ${last}`;
  return `${first.slice(0, -5)} to ${last}`;
}

/** "a", "a and b", "a, b and c". */
export function joinWords(words: readonly string[]): string {
  if (words.length <= 1) return words.join("");
  return `${words.slice(0, -1).join(", ")} and ${words[words.length - 1]}`;
}

/** The models that wrote the report, each once, in the order the steps used them. */
export function modelNames(report: PublicReport): string[] {
  return [...new Set(Object.values(report.model_ids))];
}

/** The patches whose bundle this report covers, oldest first. */
export function bundlePatches(patches: readonly Patch[], bundleId: string): Patch[] {
  return patches
    .filter((patch) => patch.report === bundleId)
    .sort((a, b) => a.date.localeCompare(b.date));
}

export function reportHref(appid: number, bundleId: string): string {
  return `/games/${appid}/reports/${bundleId}/`;
}
