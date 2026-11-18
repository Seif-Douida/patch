// The Monitoring page's words and numbers (Phase 7, design D10): what each statistic is called,
// how its values are written, and the page's plain-language summaries.

import { formatDate } from "./series";
import type { DriftSection, MonitoringFile, PipelineRun } from "./types";

export const ISSUES_URL = "https://github.com/Seif-Douida/patch/issues";

type Statistic = { title: string; note: string; digits: number };

// Each weekly drift statistic, against the champion's training data.
export const STATISTICS: Record<string, Statistic> = {
  language_psi: {
    title: "Language mix",
    note: "Population stability index of the reviews' languages: 0 means the same mix.",
    digits: 3,
  },
  length_psi: {
    title: "Review length",
    note: "Population stability index of the reviews' lengths: 0 means the same spread.",
    digits: 3,
  },
  embedding_auc: {
    title: "What reviews talk about",
    note: "How well a classifier tells this week's reviews from the training data: 0.5 means it can't.",
    digits: 3,
  },
  aspect_max_change: {
    title: "Largest aspect share change",
    note: "Percentage points between this week's share of an aspect and the training data's.",
    digits: 1,
  },
  uncertain_change: {
    title: "Uncertain predictions",
    note: "Change, in percentage points, in the share of predictions close to their threshold.",
    digits: 1,
  },
};

export function statistic(metric: string): Statistic {
  return STATISTICS[metric] ?? { title: metric, note: "", digits: 3 };
}

export function formatValue(value: number | null, digits = 3): string {
  return value === null ? "–" : value.toFixed(digits);
}

export function formatPercent(value: number | null): string {
  return value === null ? "–" : `${Math.round(value * 100)}%`;
}

export function dayOf(iso: string | null): string | null {
  return iso === null ? null : iso.slice(0, 10);
}

export function formatMinutes(seconds: number | null): string | null {
  return seconds === null ? null : `${Math.max(1, Math.round(seconds / 60))} min`;
}

export function runLabel(run: PipelineRun): string {
  const day = dayOf(run.started_at);
  const when = day === null ? "" : `, ${formatDate(day)}`;
  const took = formatMinutes(run.duration_s);
  const verb = run.status === "succeeded" ? "in" : "after";
  return `Night ${run.run_id}${when}: ${run.status}${took === null ? "" : ` ${verb} ${took}`}`;
}

export function firstCheckMessage(generatedAt: string): string {
  return `The first weekly check runs on the first night after ${formatDate(generatedAt.slice(0, 10))}.`;
}

export function hasDriftPoints(drift: readonly DriftSection[]): boolean {
  return drift.some((section) => section.points.length > 0);
}

/** "20261012T223218Z" to "12 October 2026". */
export function trainedOn(modelVersion: string): string | null {
  const match = /^(\d{4})(\d{2})(\d{2})T/.exec(modelVersion);
  return match ? formatDate(`${match[1]}-${match[2]}-${match[3]}`) : null;
}

export function retrainSummary(retrain: MonitoringFile["retrain"]): string {
  const decision = retrain.last_decision;
  if (decision === null) {
    const request = retrain.last_request;
    return request === null
      ? "No retrain yet. One is asked for when a drift issue opens, and on the first night of each month."
      : `A retrain was asked for on ${formatDate(request.date ?? "")} (${request.reason}); no decision yet.`;
  }
  // "promote" opens a pull request; the challenger is the champion only once it's merged.
  const verdict =
    decision.decision === "promote"
      ? "Proposed a challenger for promotion (a pull request)"
      : "Kept the champion";
  const gain =
    decision.recent_gain === null
      ? ""
      : `: the challenger's macro-F1 on recent reviews was ${decision.recent_gain >= 0 ? "+" : ""}${decision.recent_gain.toFixed(3)} against it`;
  const day = decision.date === null ? "" : ` on ${formatDate(decision.date)}`;
  return `${verdict}${day}${gain}${decision.n === null ? "" : ` (${decision.n} reviews)`}.`;
}

/** "faithfulness" to "Faithfulness", "answers_question" to "Answers question". */
export function scoreName(key: string): string {
  const words = key.replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}
