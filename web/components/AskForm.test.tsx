import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { askableGames } from "@/lib/ask";
import type { GameSummary, PublicReport, SiteIndex } from "@/lib/types";

import { AskForm } from "./AskForm";

const API = "https://api.example.test";
const GAMES = [
  { appid: 900001, name: "Example Arena (sample data)" },
  { appid: 900002, name: "Example Colony (sample data)" },
];

const REPORT: PublicReport = {
  report_id: "q-1a2b3c4d5e6f",
  appid: 900002,
  scope: "question",
  bundle_id: "5011",
  horizon: 14,
  question: "Did the latest patch help?",
  window: { start: "2026-07-02", end: "2026-07-15" },
  data_version: 42,
  model_ids: { scope: "model-a", gather: "model-a", draft: "model-a", summary: "model-a" },
  prompt_version: "5a3c9e1f0b72",
  summary: "Reviews stayed about as positive after the patch; a sale ran over the same days.",
  confidence: "medium",
  basic: false,
  hypotheses: (["H1", "H2", "H3", "H4", "H5"] as const).map((code) => ({
    code,
    status: code === "H2" ? "supported" : "inconclusive",
    rule: `${code}.test`,
    evidence: {},
  })),
  findings: [{ title: "The sale", hypothesis: "H2", claim_ids: ["c1"] }],
  claims: [
    {
      claim_id: "c1",
      text: "Reviews per day doubled during the sale, from 45 to 92.",
      claim_type: "comparative",
      value: 47,
      unit: "reviews",
      reworded: false,
      supported_by: 2,
    },
  ],
  withheld: {},
  rule_pass_rates: {},
};

function json(status: number, body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

/** A fetch whose /health and /ask answers the test decides; it records every call. */
function api(answers: { health?: () => Promise<Response>; ask?: () => Promise<Response> }) {
  const calls: { url: string; init?: RequestInit }[] = [];
  const fetcher = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
    calls.push({ url: String(url), init });
    if (String(url).endsWith("/health")) {
      return (answers.health ?? (async () => json(200, { status: "ok" })))();
    }
    return (answers.ask ?? (async () => json(200, { data_version: 42, report: REPORT })))();
  });
  return { fetcher: fetcher as unknown as typeof fetch, calls };
}

function ask(question: string, game = "Example Colony (sample data)") {
  fireEvent.change(screen.getByLabelText("Game"), {
    target: { value: String(GAMES.find((g) => g.name === game)?.appid) },
  });
  fireEvent.change(screen.getByLabelText("Question"), { target: { value: question } });
  fireEvent.click(screen.getByRole("button", { name: "Ask" }));
}

describe("AskForm", () => {
  afterEach(cleanup);

  it("shows_warming_up_until_the_api_answers", async () => {
    let wake: (response: Response) => void = () => undefined;
    const { fetcher, calls } = api({
      health: () => new Promise<Response>((resolve) => (wake = resolve)),
    });
    render(<AskForm games={GAMES} apiUrl={API} fetcher={fetcher} warmDelayMs={0} />);
    expect(calls).toEqual([]); // the page alone doesn't wake the API

    fireEvent.focus(screen.getByLabelText("Question")); // the visitor starts: wake it now

    expect(calls[0]?.url).toBe(`${API}/health`);
    expect(await screen.findByText(/^Warming up the analysis service/)).toBeTruthy();
    wake(json(200, { status: "ok" }));
    await waitFor(() => expect(screen.queryByText(/^Warming up/)).toBeNull());
  });

  it("renders_the_answer_as_a_report", async () => {
    const { fetcher, calls } = api({
      ask: async () =>
        json(200, { data_version: 42, report: REPORT, questions_left: 2, questions_per_hour: 3 }),
    });
    render(<AskForm games={GAMES} apiUrl={`${API}/`} fetcher={fetcher} />);
    expect(screen.getByText(/^Each visitor can ask a few questions an hour/)).toBeTruthy();

    ask("Did the latest patch help?");

    const answer = await screen.findByRole("region", { name: "Answer" });
    expect(within(answer).getByRole("list", { name: "Explanations checked" })).toBeTruthy();
    expect(within(answer).getByText(/doubled during the sale/)).toBeTruthy();
    expect(screen.getByText("2 of 3 questions left this hour")).toBeTruthy();
    const post = calls.find((call) => call.url === `${API}/ask`);
    expect(post?.init?.method).toBe("POST");
    expect(JSON.parse(String(post?.init?.body))).toEqual({
      appid: 900002,
      question: "Did the latest patch help?",
    });
  });

  it("shows_limits_and_refusals_in_plain_words", async () => {
    const cases: [() => Promise<Response>, string][] = [
      [
        async () => json(200, { data_version: 42, refusal: "Ask about at most 92 days at a time." }),
        "Ask about at most 92 days at a time.",
      ],
      [
        async () =>
          json(
            429,
            { detail: "You've asked 3 questions this hour; try again in 60 minutes." },
            { "Retry-After": "3600" },
          ),
        "You've asked 3 questions this hour; try again in 60 minutes.",
      ],
      [
        async () =>
          json(503, { detail: "Ask is unavailable right now; the nightly reports are still here." }),
        "Ask is unavailable right now; the nightly reports are still here.",
      ],
      [
        async () => Promise.reject(new TypeError("Failed to fetch")),
        "The analysis service didn't answer. Try again in a minute.",
      ],
    ];
    for (const [answer, words] of cases) {
      const { fetcher } = api({ ask: answer });
      render(<AskForm games={GAMES} apiUrl={API} fetcher={fetcher} />);

      ask("Why did reviews drop over the whole of last year?");

      // One live region carries progress and outcomes, so screen readers hear them.
      await waitFor(() => expect(screen.getByRole("status").textContent).toContain(words));
      expect(screen.queryByRole("region", { name: "Answer" })).toBeNull();
      cleanup();
    }

    // Without an API address (a build without NEXT_PUBLIC_API_URL), Ask says so up front.
    render(<AskForm games={GAMES} apiUrl={null} />);
    expect(screen.getByText(/^Ask isn't available on this copy of the site/)).toBeTruthy();
    expect((screen.getByRole("button", { name: "Ask" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("only_tracked_games_are_offered", () => {
    const game = (appid: number, name: string, first: string | null): GameSummary => ({
      appid,
      name,
      genres: [],
      first_date: first,
      last_date: first,
      latest: null,
    });
    const index = {
      schema_version: 4,
      data_version: 42,
      generated_at: "2026-09-29T03:32:00Z",
      attribution: "Data from Steam",
      games: [
        game(900002, "Example Colony (sample data)", "2026-06-01"),
        game(900003, "Example Newcomer (sample data)", null), // no reviews collected yet
        game(900001, "Example Arena (sample data)", "2026-06-01"),
      ],
    } satisfies SiteIndex;

    render(<AskForm games={askableGames(index)} apiUrl={API} />);

    const options = within(screen.getByLabelText("Game")).getAllByRole("option");
    expect(options.map((option) => option.textContent)).toEqual([
      "Example Arena (sample data)",
      "Example Colony (sample data)",
    ]);
  });
});
