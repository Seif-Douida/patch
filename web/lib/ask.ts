// The Ask page (Phase 5, design D11): which games a visitor can ask about, and what the API's
// POST /ask answers (src/patchpulse/api/app.py).

import type { PublicReport, SiteIndex } from "./types";

export type AskableGame = { appid: number; name: string };

// The agent refuses a game without reviews in its data, so the page doesn't offer one.
export function askableGames(index: SiteIndex): AskableGame[] {
  return index.games
    .filter((game) => game.first_date !== null)
    .map(({ appid, name }) => ({ appid, name }))
    .sort((a, b) => a.name.localeCompare(b.name, "en"));
}

export const SUGGESTIONS: readonly string[] = [
  "What changed last month?",
  "Did the latest patch help?",
  "What do players complain about?",
];

export const MAX_QUESTION_CHARS = 500; // the agent refuses longer ones
export const ASK_TIMEOUT_MS = 60_000;
export const WARM_DELAY_MS = 1_500; // a wake-up slower than this shows the warming-up notice

// 200: a report or a refusal; the count only when this question was counted against the limits.
// 429 and 503 carry the reason as `detail`.
export type AskResponse = {
  data_version?: number;
  report?: PublicReport;
  refusal?: string;
  questions_left?: number;
  questions_per_hour?: number;
  detail?: unknown;
};

/** The API's address without a trailing slash, or null when the build had none. */
export function apiBase(url: string | null | undefined): string | null {
  const trimmed = (url ?? "").trim().replace(/\/+$/, "");
  return trimmed === "" ? null : trimmed;
}
