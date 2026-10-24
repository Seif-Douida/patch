import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type {
  AspectDay,
  AspectModel,
  AspectSeries,
  CausalInfo,
  HorizonEffect,
  Patch,
  PatchEffect,
} from "@/lib/types";

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
    // No emphasis: with ten aspects a patch, some changes look clear by chance alone.
    expect(document.querySelector("strong, b, .change-clear")).toBeNull();
    expect(document.body.textContent).not.toMatch(/bold/i);
  });

  it("weak_aspects_are_marked", () => {
    const model: AspectModel = {
      family: "s0",
      version: "v1",
      test_macro_f1: 0.61,
      aspect_f1: { performance: 0.87, ux_controls: 0.48 },
    };
    render(
      <PatchAspectTables
        aspects={september(10)}
        patches={[patch("2026-09-16", "Patch 1.1", true)]}
        model={model}
      />,
    );

    const ux = screen.getByRole("rowheader", { name: /UX and controls/ });
    expect(ux.textContent).toContain("UX and controls*");
    expect(ux.textContent).toContain("detected less reliably");
    const performance = screen.getByRole("rowheader", { name: /Performance/ });
    expect(performance.textContent).toBe("Performance");
    const note = /^\* Detected less reliably: F1 under 0.5 on our test set/;
    expect(screen.getByText(note)).toBeTruthy();
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

// Schema 3: each treatment patch's estimated effect on the positive share (Phase 4).
const SYNTH: CausalInfo = {
  default_method: "synth",
  effects_version: "e2",
  benchmark_report: "reports/causal/benchmark.md",
  validated: true,
};
const TOO_FEW = "too few comparison games without their own patch (3 of 5 needed)";

function horizon(days: number, ends: string, values: Partial<HorizonEffect> = {}): HorizonEffect {
  return {
    days,
    ends,
    effect: null,
    lo90: null,
    hi90: null,
    method: null,
    flags: [],
    reason: null,
    ...values,
  };
}

function withEffect(date: string, title: string, effect: PatchEffect | null): Patch {
  return { ...patch(date, title, true), effect };
}

function renderPatches(patches: Patch[], discountsFrom: string | null = null) {
  render(
    <PatchAspectTables
      aspects={september(10)}
      patches={patches}
      causal={SYNTH}
      discountsFrom={discountsFrom}
    />,
  );
}

// The estimate line and its notes in a patch's block.
function estimateOf(title: string): { line: string | null; notes: string[] } {
  const block = [...document.querySelectorAll("details")].find((details) =>
    details.querySelector("summary")?.textContent?.includes(title),
  );
  return {
    line: block?.querySelector(".patch-effect-line")?.textContent ?? null,
    notes: [...(block?.querySelectorAll(".patch-effect-notes li") ?? [])].map(
      (note) => note.textContent ?? "",
    ),
  };
}

describe("PatchAspectTables estimates", () => {
  afterEach(cleanup);

  it("shows_the_estimate_with_its_range_and_method", () => {
    renderPatches([
      withEffect("2026-09-16", "Patch 1.1", {
        start: "2026-09-16",
        horizons: [
          horizon(7, "2026-09-22", { effect: -5.04, lo90: -7.2, hi90: -2.9, method: "synth" }),
          horizon(14, "2026-09-29", { effect: -4.12, lo90: -6.0, hi90: -2.24, method: "synth" }),
        ],
      }),
      // Estimated with the patch two days before: patches within 7 days are one change.
      withEffect("2026-09-03", "Patch 1.0.1", {
        start: "2026-09-01",
        horizons: [
          horizon(7, "2026-09-07", { effect: 0.04, lo90: -1.5, hi90: 1.6, method: "its" }),
          horizon(14, "2026-09-14", {
            effect: 1.25,
            lo90: -0.5,
            hi90: 3.0,
            method: "its",
            reason: TOO_FEW,
          }),
        ],
      }),
    ]);

    expect(estimateOf("Patch 1.1")).toEqual({
      line:
        "Estimated effect on positive reviews over 14 days: −4.1 points (90% range −6.0 to " +
        "−2.2), synthetic control.",
      notes: [],
    });
    expect(estimateOf("Patch 1.0.1")).toEqual({
      line:
        "Estimated effect on positive reviews over 14 days: +1.3 points (90% range −0.5 to " +
        "+3.0), interrupted time series.",
      notes: [
        "Synthetic control wasn't possible: too few comparison games without their own patch " +
          "(3 of 5 needed).",
        "Estimated together with the patch of 1 Sep 2026: patches within 7 days count as one " +
          "change.",
      ],
    });
    // Above the descriptive table, in the text's ink: no emphasis, no series colour.
    const line = document.querySelector(".patch-effect-line") as HTMLElement;
    const table = line.closest("details")?.querySelector("table") as HTMLElement;
    expect(line.compareDocumentPosition(table) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(document.querySelector("strong, b, [style]")).toBeNull();
  });

  it("falls_back_to_seven_days", () => {
    renderPatches([
      withEffect("2026-09-20", "Patch 1.2", {
        start: "2026-09-20",
        horizons: [
          horizon(7, "2026-09-26", { effect: 1.1, lo90: -2.9, hi90: 5.0, method: "synth" }),
          horizon(14, "2026-10-03", { reason: "window_not_complete" }),
        ],
      }),
      withEffect("2026-09-05", "Patch 1.0.2", {
        start: "2026-09-05",
        horizons: [
          horizon(7, "2026-09-11", { effect: -2.0, lo90: -3.5, hi90: -0.5, method: "synth" }),
          horizon(14, "2026-09-18", { reason: "another_patch_within_14_days" }),
        ],
      }),
    ]);

    expect(estimateOf("Patch 1.2")).toEqual({
      line:
        "Estimated effect on positive reviews over 7 days: +1.1 points (90% range −2.9 to " +
        "+5.0), synthetic control.",
      notes: ["The 14-day estimate comes after 3 Oct 2026."],
    });
    expect(estimateOf("Patch 1.0.2")).toEqual({
      line:
        "Estimated effect on positive reviews over 7 days: −2.0 points (90% range −3.5 to " +
        "−0.5), synthetic control.",
      notes: ["No 14-day estimate: another patch came out within 14 days."],
    });
  });

  it("explains_each_absence", () => {
    const none = (title: string, date: string, seven: Partial<HorizonEffect>): Patch =>
      withEffect(date, title, {
        start: date,
        horizons: [horizon(7, "2026-09-28", seven), horizon(14, "2026-10-05", seven)],
      });
    renderPatches([
      none("Patch A", "2026-09-22", { reason: "window_not_complete" }),
      none("Patch B", "2026-09-12", { reason: "not_estimated_yet" }),
      none("Patch C", "2026-09-08", { method: "synth", reason: TOO_FEW }),
      none("Patch D", "2026-09-04", {
        method: "synth",
        reason: "too little quiet history to check the interval (5 of 9 placebo dates)",
      }),
      none("Patch E", "2026-09-02", {
        effect: -3.0,
        method: "synth",
        reason: "interval wider than the grid",
      }),
      none("Patch F", "2026-08-30", {
        effect: -1.0,
        method: "synth",
        reason: "interval empty: no effect size passed the conformal test",
      }),
    ]);

    const lines = ["Patch A", "Patch B", "Patch C", "Patch D", "Patch E", "Patch F"].map(
      (title) => estimateOf(title).line,
    );
    expect(lines).toEqual([
      "No estimate yet: the 7 days after the patch end on 28 Sep 2026.",
      "No estimate yet: the nightly update hasn't reached this patch.",
      "No estimate: too few comparison games without their own patch (3 of 5 needed).",
      "No estimate: too little quiet history before the patch to check the range.",
      // An effect is never shown without its range.
      "No estimate: the 90% range was too wide to state.",
      "No estimate: the comparison games didn't fit the days after the patch.",
    ]);
  });

  it("shows_each_flag", () => {
    renderPatches(
      [
        withEffect("2026-09-16", "Patch 1.1", {
          start: "2026-09-16",
          horizons: [
            horizon(7, "2026-09-22", {
              effect: -4.0,
              lo90: -6.0,
              hi90: -2.0,
              method: "synth",
              flags: ["binned", "confounded", "discounts_unknown", "volume_spike"],
            }),
            horizon(14, "2026-09-29", { reason: "window_not_complete" }),
          ],
        }),
      ],
      "2026-09-30",
    );

    expect(estimateOf("Patch 1.1").notes).toEqual([
      "A Steam sale or a discount fell within 3 days of the patch, so part of the change may " +
        "be the price.",
      "Game discounts before 30 Sep 2026 aren't known.",
      "Reviews spiked after the patch.",
      "The 14-day estimate comes after 29 Sep 2026.",
    ]);
  });

  it("no_line_for_schema_2", () => {
    renderPatches([
      patch("2026-09-16", "Patch 1.1", true), // schema 2: no `effect` at all
      withEffect("2026-09-05", "Patch 1.0.1", null),
    ]);

    expect(document.querySelector(".patch-effect")).toBeNull();
    expect(screen.getAllByRole("table")).toHaveLength(2); // the descriptive tables stay
  });
});
