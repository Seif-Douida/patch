import { mkdirSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import ReportPage, { generateStaticParams } from "./page";

// The fixtures with every patch's report link removed: the live data before the first report.
function withoutReports(): string {
  const dir = mkdtempSync(path.join(tmpdir(), "site-data-"));
  const index = JSON.parse(readFileSync("fixtures/index.json", "utf8"));
  writeFileSync(path.join(dir, "index.json"), JSON.stringify({ ...index, reports: 0 }));
  mkdirSync(path.join(dir, "games"));
  for (const file of readdirSync("fixtures/games")) {
    const game = JSON.parse(readFileSync(path.join("fixtures/games", file), "utf8"));
    for (const patch of game.patches) patch.report = null;
    writeFileSync(path.join(dir, "games", file), JSON.stringify(game));
  }
  return dir;
}

// The invented sample in fixtures/: Example Colony's two patches each link their bundle's report.
describe("ReportPage", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllEnvs();
  });

  it("a_page_per_linked_report", async () => {
    // Both params come from here: the game's segment sets its params in its page, not a layout.
    expect(generateStaticParams()).toEqual([
      { appid: "900001", bundle: "5001" },
      { appid: "900002", bundle: "5011" },
      { appid: "900002", bundle: "5012" },
    ]);

    render(await ReportPage({ params: Promise.resolve({ appid: "900002", bundle: "5011" }) }));

    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("After Patch 0.9.4");
    const crumbs = screen.getByRole("navigation", { name: "Breadcrumb" });
    const game = within(crumbs).getByRole("link", { name: "Example Colony (sample data)" });
    expect(game.getAttribute("href")).toMatch(/^\/games\/900002\/?$/); // the build adds the slash
    expect(screen.getByText(/^The 14 days from 2 Jul to 15 Jul 2026/)).toBeTruthy();
    expect(screen.getByRole("list", { name: "Explanations checked" })).toBeTruthy();
  });

  it("builds_before_the_first_report", async () => {
    // A static export needs one page per dynamic route: a placeholder that is a 404.
    vi.stubEnv("SITE_DATA_DIR", withoutReports());
    vi.resetModules(); // the data folder is read when lib/data loads
    const page = await import("./page");

    const params = page.generateStaticParams();

    expect(params).toEqual([{ appid: "900001", bundle: page.NO_REPORTS }]);
    const only = params[0];
    if (!only) throw new Error("no placeholder");
    await expect(page.default({ params: Promise.resolve(only) })).rejects.toThrow(
      /NEXT_HTTP_ERROR_FALLBACK;404/,
    );
  });
});
