import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

describe("AskPage", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllEnvs();
  });

  it("offers_the_tracked_games_and_the_build_time_api", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_URL", "https://api.example.test");
    vi.resetModules();
    const { default: AskPage } = await import("./page");

    render(<AskPage />);

    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("Ask why reviews moved");
    const games = within(screen.getByLabelText("Game")).getAllByRole("option");
    expect(games.map((game) => game.textContent)).toEqual([
      "Example Arena (sample data)",
      "Example Colony (sample data)",
    ]);
    expect((screen.getByRole("button", { name: "Ask" }) as HTMLButtonElement).disabled).toBe(false);
  });
});
