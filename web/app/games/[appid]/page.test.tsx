import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

// The charts need a browser layout engine (recharts measures its container); this test is about
// the page's data, so they render nothing here.
vi.mock("@/components/GameCharts", () => ({ GameCharts: () => null }));

// Site data without aspects: schema 1, or schema 2 from a night without a model (nulls).
function siteData(schema: 1 | 2): string {
  const dir = mkdtempSync(path.join(tmpdir(), "site-data-"));
  const header = {
    schema_version: schema,
    data_version: 7,
    generated_at: "2026-10-14T03:30:00Z",
    attribution: "Data from Steam",
  };
  const v2 = schema === 2;
  const summary = {
    appid: 900001,
    name: "Example Arena (sample data)",
    genres: ["Action"],
    first_date: "2026-10-12",
    last_date: "2026-10-13",
    latest: { days: 30, to: "2026-10-13", n: 4, share: 0.75 },
    ...(v2 ? { top_complaint_30d: null } : {}),
  };
  writeFileSync(
    path.join(dir, "index.json"),
    JSON.stringify({ ...header, games: [summary], ...(v2 ? { aspect_model: null } : {}) }),
  );
  mkdirSync(path.join(dir, "games"));
  writeFileSync(
    path.join(dir, "games", "900001.json"),
    JSON.stringify({
      ...header,
      appid: 900001,
      name: summary.name,
      genres: summary.genres,
      series: [],
      patches: [],
      ...(v2 ? { aspects: null } : {}),
    }),
  );
  return dir;
}

describe("GamePage", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllEnvs();
  });

  it("game_page_renders_without_aspects", async () => {
    for (const schema of [1, 2] as const) {
      vi.stubEnv("SITE_DATA_DIR", siteData(schema));
      vi.resetModules(); // the data folder is read when lib/data loads
      const { default: GamePage } = await import("./page");

      render(await GamePage({ params: Promise.resolve({ appid: "900001" }) }));

      expect(screen.getByRole("heading", { level: 1 }).textContent).toBe(
        "Example Arena (sample data)",
      );
      expect(screen.getByText("Patches and posts")).toBeTruthy();
      cleanup();
    }
  });
});
