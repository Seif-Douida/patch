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

// The wait between the click and the answer (design F8): what the analyst does for a question,
// in the agent's order. They don't track the server's real step (the page can't see it): they
// play in order, then the last seven repeat until the answer arrives.
export const WAIT_PHRASES: readonly string[] = [
  "Reading your question",
  "Pulling the review numbers",
  "Lining up the patches",
  "Deciphering the patch notes",
  "Weighing possible causes",
  "Searching the reviews",
  "Sifting what players said",
  "Thinking it through",
  "Drafting the answer",
  "Checking every number against the data",
  "Checking each claim",
  "Writing it up",
];
export const PHRASE_MS = 2_800;
export const REPEAT_FROM = 5; // "Searching the reviews"
export const SLOW_NOTE_MS = 8_000;
export const SLOW_NOTE = "A question takes up to a minute, a little longer after a quiet spell.";
export const WORKING = "Working on your answer."; // said once, by the form's status region

/** The phrase shown `elapsedMs` after the click. */
export function phraseAt(elapsedMs: number): string {
  const step = Math.floor(elapsedMs / PHRASE_MS);
  const count = WAIT_PHRASES.length;
  const index = step < count ? step : REPEAT_FROM + ((step - count) % (count - REPEAT_FROM));
  return WAIT_PHRASES[index] ?? "";
}

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
