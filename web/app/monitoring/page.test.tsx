import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

// The charts need a browser layout engine (recharts measures its container): here they render
// their table views only, which is what these tests read.
const FIXTURES = path.resolve(__dirname, "../../fixtures");

async function renderPage(dataDir: string) {
  vi.stubEnv("SITE_DATA_DIR", dataDir);
  vi.resetModules(); // the data folder is read when lib/data loads
  const { default: MonitoringPage } = await import("./page");
  render(<MonitoringPage />);
}

describe("Monitoring", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllEnvs();
  });

  it("renders_every_section_from_the_sample", async () => {
    await renderPage(FIXTURES);

    const headings = screen.getAllByRole("heading", { level: 2 }).map((h) => h.textContent);
    expect(headings).toEqual(["Open alerts", "Pipeline", "Model", "Agent"]);
    const alerts = within(screen.getByRole("list", { name: "Open alerts" }));
    expect(alerts.getAllByRole("listitem")).toHaveLength(2);
    expect(alerts.getByRole("link", { name: /#101/ }).getAttribute("href")).toBe(
      "https://github.com/Seif-Douida/patch/issues/101",
    );
    // The last 30 nights, the failed one named in words, not by colour alone.
    const strip = screen.getByRole("list", { name: "Last 30 nights" });
    expect(within(strip).getAllByRole("listitem")).toHaveLength(30);
    expect(within(strip).getAllByText("failed")).toHaveLength(1);
    expect(screen.getByText(/aspect-classifier v3/)).toBeTruthy();
    expect(screen.getAllByRole("figure").length).toBeGreaterThanOrEqual(5 + 1 + 1 + 2);
    expect(screen.getByText(/Kept the champion/)).toBeTruthy();
    // The whole page with every chart: ~1.6 s alone, up to ~5.6 s with the suite in parallel.
  }, 15_000);

  it("shows_the_empty_states_without_data", async () => {
    const dir = mkdtempSync(path.join(tmpdir(), "monitoring-"));
    writeFileSync(
      path.join(dir, "monitoring.json"),
      JSON.stringify({
        schema_version: 1,
        generated_at: "2026-11-18T03:40:00Z",
        pipeline: [],
        freshness: { last_success: null, age_hours: null },
        data_quality: [],
        drift: [
          { rule: "drift_language", metric: "language_psi", threshold: null, points: [] },
        ],
        agreement: { reference: null, threshold: 0.05, weeks: [] },
        online: { nights: [] },
        retrain: { last_request: null, last_decision: null },
        rules: [],
        champion: null,
      }),
    );

    await renderPage(dir);

    expect(screen.getByText("No alerts open")).toBeTruthy();
    expect(
      screen.getByText("The first weekly check runs on the first night after 18 Nov 2026."),
    ).toBeTruthy();
    expect(screen.getByText(/No retrain yet/)).toBeTruthy();
  });

  it("without_any_monitoring_file_the_page_still_builds", async () => {
    await renderPage(mkdtempSync(path.join(tmpdir(), "monitoring-none-")));

    expect(screen.getByText("No alerts open")).toBeTruthy();
    expect(screen.getByText(/monitoring data appears after the next night/i)).toBeTruthy();
  });

  it("labels_the_judge_uncalibrated", async () => {
    await renderPage(FIXTURES);

    expect(screen.getByRole("heading", { name: "Judge (uncalibrated)" })).toBeTruthy();
  });
});
