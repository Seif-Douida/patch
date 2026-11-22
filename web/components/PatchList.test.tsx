import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type {
  AspectDay,
  AspectSeries,
  CausalInfo,
  HorizonEffect,
  Patch,
  PatchEffect,
} from "@/lib/types";

import { PatchList } from "./PatchList";

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

// September 2026: 10 scored negative reviews a day; 2 mention performance before 16 Sep, 4 after.
function september(): AspectSeries {
  const days: AspectDay[] = [];
  for (let d = 1; d <= 30; d += 1) {
    const date = `2026-09-${String(d).padStart(2, "0")}`;
    const neg = CODES.map((code) => (code === "performance" ? (date < "2026-09-16" ? 2 : 4) : 0));
    days.push({ date, scored: [10, 5], neg, pos: CODES.map(() => 0) });
  }
  return { codes: CODES, days };
}

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

function post(date: string, title: string, treatment: boolean, extra: Partial<Patch> = {}): Patch {
  return {
    date,
    title,
    url: `https://example.com/${date}/${title.replace(/\W+/g, "-")}`,
    type: treatment ? "patch" : "other",
    treatment,
    ...extra,
  };
}

function withEffect(date: string, title: string, effect: PatchEffect | null, extra: Partial<Patch> = {}) {
  return post(date, title, true, { effect, report: null, ...extra });
}

function estimated(start: string, effect = -4): PatchEffect {
  return {
    start,
    horizons: [
      horizon(7, "2026-09-30", { effect, lo90: effect - 2, hi90: effect + 2, method: "synth" }),
      horizon(14, "2026-10-07", { reason: "window_not_complete" }),
    ],
  };
}

function renderList(
  patches: Patch[],
  { aspects = september(), discountsFrom = null }: { aspects?: AspectSeries | null; discountsFrom?: string | null } = {},
) {
  return render(
    <PatchList
      appid={553850}
      patches={patches}
      aspects={aspects}
      model={null}
      causal={SYNTH}
      discountsFrom={discountsFrom}
    />,
  );
}

function rows(): HTMLElement[] {
  return screen.getAllByRole("button").filter((button) => button.hasAttribute("aria-expanded"));
}

function rowOf(title: string): HTMLElement {
  const row = rows().find((button) => button.textContent?.includes(title));
  if (!row) throw new Error(`no row for ${title}`);
  return row;
}

// Opens a row (if it isn't) and returns its panel.
function open(title: string): HTMLElement {
  const row = rowOf(title);
  if (row.getAttribute("aria-expanded") !== "true") fireEvent.click(row);
  const panel = document.getElementById(row.getAttribute("aria-controls") ?? "");
  if (!panel) throw new Error(`no panel for ${title}`);
  return panel;
}

function estimateOf(title: string): { line: string | null; notes: string[] } {
  const panel = open(title);
  return {
    line: panel.querySelector(".patch-effect-line")?.textContent ?? null,
    notes: [...panel.querySelectorAll(".patch-effect-notes li")].map((li) => li.textContent ?? ""),
  };
}

describe("PatchList", () => {
  afterEach(cleanup);

  it("lists_one_row_per_estimate_newest_first", () => {
    const together = estimated("2026-09-10");
    renderList([
      withEffect("2026-09-01", "Patch 1.0", estimated("2026-09-01")),
      withEffect("2026-09-12", "Update 2: 1.1.1", together),
      withEffect("2026-09-10", "Update 2: 1.1.0", together),
      post("2026-09-15", "Dev blog", false),
      withEffect("2026-09-20", "Patch 1.2", estimated("2026-09-20")),
    ]);

    expect(screen.getByRole("heading", { level: 2, name: "Patches" })).toBeTruthy();
    expect(
      screen.getByText("Newest first. Patches a few days apart share one estimate and one report."),
    ).toBeTruthy();
    expect(rows().map((row) => row.getAttribute("aria-label"))).toEqual([
      "20 Sep 2026, Patch 1.2. Clearly down: −4.0 points, 90% range −6.0 to −2.0, over 7 days",
      "10 Sep 2026, Update 2: 1.1.0 and 1.1.1. Clearly down: −4.0 points, 90% range −6.0 to " +
        "−2.0, over 7 days",
      "1 Sep 2026, Patch 1.0. Clearly down: −4.0 points, 90% range −6.0 to −2.0, over 7 days",
    ]);
  });

  it("the_newest_row_starts_open", () => {
    renderList([
      withEffect("2026-09-01", "Patch 1.0", estimated("2026-09-01")),
      withEffect("2026-09-20", "Patch 1.2", estimated("2026-09-20")),
    ]);

    const [newest, older] = rows();
    expect(newest?.getAttribute("aria-expanded")).toBe("true");
    expect(older?.getAttribute("aria-expanded")).toBe("false");
    const panel = document.getElementById(newest?.getAttribute("aria-controls") ?? "");
    expect(panel?.querySelector(".patch-effect-line")?.textContent).toMatch(
      /^Estimated effect on positive reviews over 7 days: −4.0 points/,
    );
  });

  it("shows_eight_rows_then_ten_more", () => {
    const patches = Array.from({ length: 20 }, (_, i) => {
      const date = `2026-08-${String(i + 1).padStart(2, "0")}`;
      return withEffect(date, `Patch 0.${i + 1}`, estimated(date));
    });
    renderList(patches);

    expect(rows()).toHaveLength(8);
    expect(screen.getByText("12 older tested patches")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Show 10 more" }));
    expect(rows()).toHaveLength(18);
    fireEvent.click(screen.getByRole("button", { name: "Show 2 more" }));
    expect(rows()).toHaveLength(20);
    expect(screen.queryByRole("button", { name: /^Show \d+ more$/ })).toBeNull();
  });

  it("switching_to_all_posts_lists_every_post_and_starts_again_at_eight", () => {
    const tested = Array.from({ length: 20 }, (_, i) => {
      const date = `2026-08-${String(i + 1).padStart(2, "0")}`;
      return withEffect(date, `Patch 0.${i + 1}`, estimated(date));
    });
    const others = Array.from({ length: 5 }, (_, i) =>
      post(`2026-09-0${i + 1}`, `Dev blog ${i + 1}`, false),
    );
    renderList([...tested, ...others]);

    const chips = within(screen.getByRole("group", { name: "Patches shown" }));
    const testedChip = chips.getByRole("button", { name: "Tested patches (20)" });
    const allChip = chips.getByRole("button", { name: "All posts (25)" });
    expect(testedChip.getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(screen.getByRole("button", { name: "Show 10 more" }));

    fireEvent.click(allChip);
    expect(allChip.getAttribute("aria-pressed")).toBe("true");
    const table = screen.getByRole("table");
    const body = within(table).getAllByRole("row").slice(1);
    expect(body).toHaveLength(8);
    // Newest first: the five dev blogs, then the latest patches.
    const first = within(body[0] as HTMLElement).getByRole("link");
    expect(first.textContent).toBe("Dev blog 5");
    expect(first.getAttribute("href")).toBe("https://example.com/2026-09-05/Dev-blog-5");
    expect(first.getAttribute("rel")).toBe("noopener noreferrer");
    expect(body[5]?.textContent).toContain("Patch");
    expect(screen.getByText("17 older posts")).toBeTruthy();
    expect(screen.getByText(/^Each post is labelled by its title\./)).toBeTruthy();

    fireEvent.click(testedChip);
    expect(rows()).toHaveLength(8);
  });

  it("an_opened_row_lists_its_posts_and_its_report", () => {
    const together = estimated("2026-09-10");
    renderList([
      withEffect("2026-09-12", "Update 2: 1.1.1", together, { report: "b7" }),
      withEffect("2026-09-10", "Update 2: 1.1.0", together, { report: "b7" }),
    ]);

    const panel = open("Update 2");
    const links = within(panel).getAllByRole("link");
    expect(links.map((link) => link.textContent)).toEqual([
      "Update 2: 1.1.0",
      "Update 2: 1.1.1",
      "Read the report",
    ]);
    // Next adds the trailing slash in the build (trailingSlash: true), not under Vitest.
    expect(links[2]?.getAttribute("href")).toMatch(/^\/games\/553850\/reports\/b7\/?$/);
    expect(panel.textContent).toContain("10 Sep 2026");
    expect(panel.querySelector("figure")).not.toBeNull(); // the before-and-after chart
  });

  it("a_merged_row_drops_the_estimated_together_note", () => {
    const together: PatchEffect = {
      start: "2026-09-01",
      horizons: [
        horizon(7, "2026-09-07", { effect: 0.04, lo90: -1.5, hi90: 1.6, method: "its" }),
        horizon(14, "2026-09-14", { effect: 1.25, lo90: -0.5, hi90: 3.0, method: "its", reason: TOO_FEW }),
      ],
    };
    renderList([
      withEffect("2026-09-01", "Patch 1.0.0", together),
      withEffect("2026-09-03", "Patch 1.0.1", together),
    ]);

    expect(estimateOf("Patch 1.0.0 and 1.0.1")).toEqual({
      line:
        "Estimated effect on positive reviews over 14 days: +1.3 points (90% range −0.5 to " +
        "+3.0), interrupted time series.",
      notes: [
        "Synthetic control wasn't possible: too few comparison games without their own patch " +
          "(3 of 5 needed).",
      ],
    });
    expect(document.body.textContent).not.toContain("Estimated together with");
  });

  it("an_opened_row_states_its_verdict_in_words", () => {
    renderList([
      withEffect("2026-09-20", "Patch 1.2", estimated("2026-09-20", -4)),
      withEffect("2026-09-10", "Patch 1.1", {
        start: "2026-09-10",
        horizons: [horizon(7, "2026-09-16", { effect: 1, lo90: -2, hi90: 4, method: "synth" })],
      }),
      withEffect("2026-09-01", "Patch 1.0", null),
    ]);

    // Colour is never the only carrier: the opened row says the verdict too.
    expect(open("Patch 1.2").querySelector(".patch-verdict")?.textContent).toBe("Clearly down");
    expect(open("Patch 1.1").querySelector(".patch-verdict")?.textContent).toBe("No clear change");
    expect(open("Patch 1.0").querySelector(".patch-verdict")).toBeNull(); // its sentence says so
  });

  it("show_more_moves_focus_to_the_first_new_row", () => {
    const patches = Array.from({ length: 12 }, (_, i) => {
      const date = `2026-08-${String(i + 1).padStart(2, "0")}`;
      return withEffect(date, `Patch 0.${i + 1}`, estimated(date));
    });
    patches.push(post("2026-09-01", "Dev blog", false));
    renderList(patches);

    const more = screen.getByRole("button", { name: "Show 4 more" });
    more.focus();
    fireEvent.click(more);
    // The button is gone (nothing left); focus lands on the first row it revealed, not the page.
    expect(document.activeElement).toBe(rows()[8]);

    fireEvent.click(screen.getByRole("button", { name: "All posts (13)" }));
    fireEvent.click(screen.getByRole("button", { name: "Show 5 more" }));
    const links = within(screen.getByRole("table")).getAllByRole("link");
    expect(document.activeElement).toBe(links[8]);
  });

  it("empty_states_say_so", () => {
    renderList([post("2026-09-15", "Dev blog", false)]);
    expect(screen.getByText("No tested patches in this period.")).toBeTruthy();
    cleanup();

    renderList([]);
    expect(screen.getByText("No tested patches in this period.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "All posts (0)" }));
    expect(screen.getByText("No official posts in this period.")).toBeTruthy();
  });

  it("schema_3_and_no_aspects_still_list_the_patches", () => {
    const old = withEffect("2026-09-16", "Patch 1.1", estimated("2026-09-16"));
    delete old.report; // schema 3 has no report field at all
    const { container } = renderList([old, post("2026-09-02", "Patch 1.0", true)], {
      aspects: null,
    });

    expect(rows()).toHaveLength(2);
    const panel = open("Patch 1.1");
    expect(within(panel).queryByRole("link", { name: "Read the report" })).toBeNull();
    expect(container.querySelector("figure")).toBeNull();
    // A tested patch without any estimate (older data) says so when opened.
    expect(open("Patch 1.0").querySelector(".patch-effect-line")?.textContent).toBe("No estimate.");
  });
});

// The estimate sentences, ported from PatchAspectTable's tests: the same words in the opened row.
describe("PatchList estimates", () => {
  afterEach(cleanup);

  it("shows_the_estimate_with_its_range_and_method", () => {
    renderList([
      withEffect("2026-09-16", "Patch 1.1", {
        start: "2026-09-16",
        horizons: [
          horizon(7, "2026-09-22", { effect: -5.04, lo90: -7.2, hi90: -2.9, method: "synth" }),
          horizon(14, "2026-09-29", { effect: -4.12, lo90: -6.0, hi90: -2.24, method: "synth" }),
        ],
      }),
    ]);

    expect(estimateOf("Patch 1.1")).toEqual({
      line:
        "Estimated effect on positive reviews over 14 days: −4.1 points (90% range −6.0 to " +
        "−2.2), synthetic control.",
      notes: [],
    });
    // In the text's ink: no emphasis.
    expect(document.querySelector(".patch-effect strong, .patch-effect b")).toBeNull();
  });

  it("falls_back_to_seven_days", () => {
    renderList([
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
    renderList([
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
      "No estimate yet: the updates haven't reached this patch.",
      "No estimate: too few comparison games without their own patch (3 of 5 needed).",
      "No estimate: too little quiet history before the patch to check the range.",
      // An effect is never shown without its range.
      "No estimate: the 90% range was too wide to state.",
      "No estimate: the comparison games didn't fit the days after the patch.",
    ]);
  });

  it("shows_each_flag", () => {
    renderList(
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
      { discountsFrom: "2026-09-30" },
    );

    expect(estimateOf("Patch 1.1").notes).toEqual([
      "A Steam sale or a discount fell within 3 days of the patch, so part of the change may " +
        "be the price.",
      "Game discounts before 30 Sep 2026 aren't known.",
      "Reviews spiked after the patch.",
      "The 14-day estimate comes after 29 Sep 2026.",
    ]);
  });

  it("links_the_report_when_there_is_one", () => {
    renderList([
      withEffect("2026-09-16", "Patch 1.1", estimated("2026-09-16"), { report: "5001" }),
      withEffect("2026-09-05", "Patch 1.0.1", estimated("2026-09-05"), { report: null }),
    ]);

    const link = within(open("Patch 1.1")).getByRole("link", { name: "Read the report" });
    expect(link.getAttribute("href")).toMatch(/^\/games\/553850\/reports\/5001\/?$/);
    expect(within(open("Patch 1.0.1")).queryByRole("link", { name: /report/ })).toBeNull();
  });
});
