import type { Metadata } from "next";

import { AskForm } from "@/components/AskForm";
import { askableGames } from "@/lib/ask";
import { loadIndex } from "@/lib/data";

export const metadata: Metadata = {
  title: "Ask",
  description: "Ask why a game's Steam reviews moved; the answer is checked against the data.",
};

/**
 * Ask (Phase 5, design D11). The page is static: the games come from the index at build time and
 * the API's address from NEXT_PUBLIC_API_URL (publish-site.yml reads pp-api's); the form asks the
 * API from the browser.
 */
export default function AskPage() {
  return (
    <>
      <header className="ask-header">
        <h1>Ask why reviews moved</h1>
        <p>
          Pick a game and ask what changed in its reviews, and when. The answer is written from the
          same data as the patch reports, and every number and claim in it is checked before you see
          it.
        </p>
      </header>
      <AskForm games={askableGames(loadIndex())} apiUrl={process.env.NEXT_PUBLIC_API_URL ?? null} />
    </>
  );
}
