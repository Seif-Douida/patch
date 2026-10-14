import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { AspectDay, AspectSeries, Patch } from "@/lib/types";

import { PatchAspectTables } from "./PatchAspectTable";

const CODES = [
  "performance",
  "stability_bugs",
  "gameplay_balance",
  "content",
  "monetization",
  "online_servers",
  "story_world",
  "ux_controls",
  "platform_policy",
  "off_topic",
];

// September 2026, `perDay` scored negative reviews a day. With 10 a day, 2 mention performance
// before 16 Sep and 4 from then on; with fewer, none do.
function september(perDay: number): AspectSeries {
  const days: AspectDay[] = [];
  for (let d = 1; d <= 30; d += 1) {
    const date = `2026-09-${String(d).padStart(2, "0")}`;
    const performance = perDay < 10 ? 0 : date < "2026-09-16" ? 2 : 4;
    const neg = CODES.map((code) => (code === "performance" ? performance : 0));
    days.push({ date, scored: [perDay, 5], neg, pos: CODES.map(() => 0) });
  }
  return { codes: CODES, days };
}

function patch(date: string, title: string, treatment: boolean): Patch {
  const type = treatment ? "patch" : "other";
  return { date, title, url: `https://example.com/${date}`, type, treatment };
}

describe("PatchAspectTables", () => {
  afterEach(cleanup);

  it("one_descriptive_table_per_treatment_patch", () => {
    render(
      <PatchAspectTables
        aspects={september(10)}
        patches={[
          patch("2026-09-16", "Patch 1.1", true),
          patch("2026-09-18", "Dev blog", false),
          patch("2026-09-05", "Patch 1.0.1", true),
        ]}
      />,
    );

    const tables = screen.getAllByRole("table");
    expect(tables).toHaveLength(2); // the dev blog isn't a treatment patch
    for (const table of tables) {
      expect(table.closest("details")).not.toBeNull();
      expect(within(table).getByText(/^Descriptive: the change around a patch, not/)).toBeTruthy();
    }
    // Most recent first; its performance row: 20% before, 40% after, +20 points.
    const [latest, earlier] = tables;
    const summary = latest?.closest("details")?.querySelector("summary");
    expect(summary?.textContent).toContain("Patch 1.1");
    const performance = within(latest as HTMLElement).getByRole("row", { name: /Performance/ });
    expect(within(performance).getAllByRole("cell").map((c) => c.textContent)).toEqual([
      "20%",
      "40%",
      "+20",
      "+9 to +30",
    ]);
    // The earlier patch has only 4 days of data before it.
    expect(earlier?.closest("details")?.textContent).toContain("only 4 days of data before");
  });

  it("too_few_reviews_show_dashes_not_numbers", () => {
    const patches = [patch("2026-09-16", "Patch 1.1", true)];
    render(<PatchAspectTables aspects={september(2)} patches={patches} />);

    const table = screen.getByRole("table");
    const cells = within(table).getAllByRole("cell").map((c) => c.textContent);
    expect(cells.every((text) => text === "–")).toBe(true);
    expect(table.closest("details")?.textContent).toContain(
      "Too few scored negative reviews for a comparison (28 before, 28 after)",
    );
  });
});
