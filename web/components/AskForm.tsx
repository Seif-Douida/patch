"use client";

import { type FormEvent, useRef, useState } from "react";

import {
  ASK_TIMEOUT_MS,
  type AskResponse,
  type AskableGame,
  MAX_QUESTION_CHARS,
  SUGGESTIONS,
  WARM_DELAY_MS,
  apiBase,
} from "@/lib/ask";
import type { PublicReport } from "@/lib/types";

import { ReportView } from "./ReportView";

const NO_ANSWER = "The analysis service didn't answer. Try again in a minute.";
const TIMED_OUT = "The analysis service didn't answer within a minute. Try again in a moment.";
const NOT_ASKED = "That question couldn't be asked. Try rewording it.";

type Answer = { question: string; report: PublicReport };
type Left = { left: number; perHour: number };

type Props = {
  games: AskableGame[];
  apiUrl: string | null; // NEXT_PUBLIC_API_URL at build time
  fetcher?: typeof fetch;
  warmDelayMs?: number;
  timeoutMs?: number;
};

/**
 * Ask (Phase 5, design D11): pick a game, type a question or start from a suggestion; the answer
 * is a report in the same layout as the patch reports, from the API. The API sleeps when idle, so
 * the visitor's first move wakes it, and a slow wake-up says so.
 */
export function AskForm({
  games,
  apiUrl,
  fetcher,
  warmDelayMs = WARM_DELAY_MS,
  timeoutMs = ASK_TIMEOUT_MS,
}: Props) {
  const base = apiBase(apiUrl);
  const call: typeof fetch = fetcher ?? ((...args) => fetch(...args));
  const [appid, setAppid] = useState(games[0]?.appid ?? 0);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [warming, setWarming] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [left, setLeft] = useState<Left | null>(null);
  const woken = useRef(false);

  function wake() {
    if (woken.current || base === null) return;
    woken.current = true;
    let answered = false;
    const slow = setTimeout(() => {
      if (!answered) setWarming(true);
    }, warmDelayMs);
    call(`${base}/health`)
      .catch(() => undefined) // a failed wake-up shows when the question is asked
      .finally(() => {
        answered = true;
        clearTimeout(slow);
        setWarming(false);
      });
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (base === null || busy) return;
    const text = question.trim();
    if (text === "") {
      setMessage("Type a question first, or pick one of the suggestions.");
      return;
    }
    wake();
    setBusy(true);
    setMessage(null);
    setAnswer(null);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await call(`${base}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ appid, question: text }),
        signal: controller.signal,
      });
      const body = (await response.json().catch(() => ({}))) as AskResponse;
      if (typeof body.questions_left === "number" && typeof body.questions_per_hour === "number") {
        setLeft({ left: body.questions_left, perHour: body.questions_per_hour });
      }
      if (response.ok && body.report) {
        setAnswer({ question: text, report: body.report });
      } else if (response.ok && body.refusal) {
        setMessage(body.refusal);
      } else {
        setMessage(typeof body.detail === "string" ? body.detail : NOT_ASKED);
      }
    } catch {
      setMessage(controller.signal.aborted ? TIMED_OUT : NO_ANSWER);
    } finally {
      clearTimeout(timer);
      setBusy(false);
    }
  }

  const status = busy
    ? "Working on it: the analyst reads the data and the reviews, which takes up to a minute."
    : message;

  return (
    <div className="ask">
      <form className="ask-form" onSubmit={submit}>
        <div className="ask-field">
          <label htmlFor="ask-game">Game</label>
          <select
            id="ask-game"
            value={appid}
            onChange={(event) => setAppid(Number(event.target.value))}
            onFocus={wake}
          >
            {games.map((game) => (
              <option key={game.appid} value={game.appid}>
                {game.name}
              </option>
            ))}
          </select>
        </div>
        <div className="ask-field ask-field-question">
          <label htmlFor="ask-question">Question</label>
          <textarea
            id="ask-question"
            rows={3}
            maxLength={MAX_QUESTION_CHARS}
            placeholder="Why did reviews drop after the latest update?"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onFocus={wake}
          />
        </div>
        <div className="ask-suggestions">
          <span id="ask-try">Try</span>
          <ul aria-labelledby="ask-try">
            {SUGGESTIONS.map((suggestion) => (
              <li key={suggestion}>
                <button
                  type="button"
                  onClick={() => {
                    wake();
                    setQuestion(suggestion);
                  }}
                >
                  {suggestion}
                </button>
              </li>
            ))}
          </ul>
        </div>
        <div className="ask-actions">
          <button type="submit" className="ask-submit" disabled={base === null || busy}>
            Ask
          </button>
          <p className="ask-left">
            {left === null
              ? "Each visitor can ask a few questions an hour."
              : `${left.left} of ${left.perHour} questions left this hour`}
          </p>
        </div>
      </form>

      {base === null && (
        <p className="ask-unavailable">
          Ask isn&apos;t available on this copy of the site: it was built without the analysis
          service&apos;s address. The patch reports are on each game&apos;s page.
        </p>
      )}
      {warming && (
        <p className="ask-warming">
          Warming up the analysis service: the first question after a quiet spell takes up to half
          a minute.
        </p>
      )}
      <p className="ask-status" role="status">
        {status}
      </p>

      {answer && (
        <section className="ask-answer" aria-label="Answer">
          <p className="ask-asked">{answer.question}</p>
          <ReportView report={answer.report} />
        </section>
      )}
    </div>
  );
}
