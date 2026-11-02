import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

// The charts need a browser layout engine (recharts measures its container); this test is about
// the page's data, so they render nothing here.
vi.mock("@/components/GameCharts", () => ({ GameCharts: () => null }));

// Site data from a night without a model (aspect fields null): schema 3 with its causal block and
// a patch that isn't a treatment candidate (no estimate), or schema 4 with no reports yet.
function siteData(schema: 3 | 4): string {
  const dir = mkdtempSync(path.join(tmpdir(), "site-data-"));
  const header = {
    schema_version: schema,
    data_version: 7,
    generated_at: "2026-10-14T03:30:00Z",
    attribution: "Data from Steam",
  };
  const v4 = schema === 4;
  const summary = {
    appid: 900001,
    name: "Example Arena (sample data)",
    genres: ["Action"],
    first_date: "2026-10-12",
    last_date: "2026-10-13",
    latest: { days: 30, to: "2026-10-13", n: 4, share: 0.75 },
    top_complaint_30d: null,
  };
  const causal = { default_method: "its", effects_version: "e2", benchmark_report: null };
  writeFileSync(
    path.join(dir, "index.json"),
    JSON.stringify({
      ...header,
      games: [summary],
      aspect_model: null,
      causal,
      ...(v4 ? { reports: 0 } : {}),
    }),
  );
  const post = {
    date: "2026-10-12",
    title: "Dev blog (sample data)",
    url: "https://example.com/blog",
    type: "other",
    treatment: false,
  };
  mkdirSync(path.join(dir, "games"));
  writeFileSync(
    path.join(dir, "games", "900001.json"),
    JSON.stringify({
      ...header,
      appid: 900001,
      name: summary.name,
      genres: summary.genres,
      series: [],
      patches: [v4 ? { ...post, effect: null, report: null } : { ...post, effect: null }],
      aspects: null,
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
    for (const schema of [3, 4] as const) {
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
