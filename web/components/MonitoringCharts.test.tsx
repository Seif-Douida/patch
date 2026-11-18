import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { DriftSection } from "@/lib/types";

import { DriftChart, PointDot, RunStrip } from "./MonitoringCharts";

const LANGUAGE: DriftSection = {
  rule: "drift_language",
  metric: "language_psi",
  threshold: 0.12,
  points: [
    { window_end: "2026-09-13", value: 0.04, drill: false },
    { window_end: "2026-09-20", value: 0.17, drill: false },
    { window_end: "2026-09-20", value: 0.61, drill: true },
  ],
};

describe("MonitoringCharts", () => {
  afterEach(cleanup);

  it("marks_drill_points", () => {
    render(<DriftChart section={LANGUAGE} />);

    // In the table view, a drill's point says so and is told apart from the week it measured.
    const table = within(screen.getByRole("table", { name: /Language mix/ }));
    const rows = table.getAllByRole("row").slice(1).map((row) => row.textContent);
    expect(rows).toEqual([
      "13 Sep 20260.040",
      "20 Sep 20260.170over",
      "20 Sep 2026 (drill)0.610over",
    ]);
    // Drawn hollow: the dot itself carries the drill.
    const { container } = render(
      <svg>
        <PointDot cx={10} cy={10} payload={{ drill: true }} />
        <PointDot cx={20} cy={10} payload={{ drill: false }} />
      </svg>,
    );
    const dots = container.querySelectorAll("circle");
    expect(dots[0]?.getAttribute("class")).toContain("point-drill");
    expect(dots[1]?.getAttribute("class")).toContain("point-week");
  });

  it("a_statistic_without_a_threshold_says_so", () => {
    render(<DriftChart section={{ ...LANGUAGE, threshold: null }} />);

    expect(screen.getByText("No threshold yet: it is set from past weeks.")).toBeTruthy();
  });

  it("the_run_strip_names_each_status_in_words", () => {
    render(
      <RunStrip
        runs={[
          { run_id: 41, started_at: "2026-09-25T03:01:00Z", status: "succeeded", duration_s: 1500, stages_s: {} },
          { run_id: 42, started_at: "2026-09-27T03:01:00Z", status: "failed", duration_s: 900, stages_s: {} },
        ]}
      />,
    );

    const items = within(screen.getByRole("list", { name: "Last 30 nights" })).getAllByRole("listitem");
    expect(items.map((item) => item.getAttribute("aria-label"))).toEqual([
      "Night 41, 25 Sep 2026: succeeded in 25 min",
      "Night 42, 27 Sep 2026: failed after 15 min",
    ]);
  });
});
