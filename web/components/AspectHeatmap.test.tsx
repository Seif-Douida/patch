import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { AspectDay, AspectSeries } from "@/lib/types";

import { AspectHeatmap } from "./AspectHeatmap";

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

function counts(values: Partial<Record<string, number>>): number[] {
  return CODES.map((code) => values[code] ?? 0);
}

function day(date: string, scored: [number, number], neg = {}, pos = {}): AspectDay {
  return { date, scored, neg: counts(neg), pos: counts(pos) };
}

// Three weeks: two with enough reviews, one thin.
const ASPECTS: AspectSeries = {
  codes: CODES,
  days: [
    day("2026-09-28", [40, 20], { performance: 10, content: 2 }, { content: 10 }),
    day("2026-10-05", [3, 30], { performance: 3 }, { performance: 3, story_world: 15 }),
    day("2026-10-12", [20, 20], { stability_bugs: 7 }, { content: 1 }),
  ],
};

function renderHeatmap() {
  return render(<AspectHeatmap aspects={ASPECTS} from="2026-09-28" to="2026-10-18" />);
}

// Each cell's colour class, keyed by the share it shows: the same share must keep its colour.
function coloursByShare(): Map<string, string> {
  const map = new Map<string, string>();
  for (const cell of document.querySelectorAll<HTMLElement>("[data-share]")) {
    map.set(cell.dataset.share ?? "", cell.dataset.level ?? "");
  }
  return map;
}

describe("AspectHeatmap", () => {
  afterEach(cleanup);

  it("heatmap_has_ten_rows_and_a_table_view", () => {
    renderHeatmap();

    const grid = screen.getByRole("grid", { name: /negative reviews/i });
    expect(within(grid).getAllByRole("row")).toHaveLength(10);
    expect(within(grid).getAllByRole("rowheader").map((h) => h.textContent)[0]).toBe(
      "Performance",
    );
    expect(within(grid).getAllByRole("gridcell")).toHaveLength(30);
    expect(grid.querySelectorAll("[data-thin]")).toHaveLength(10); // the thin week, every row
    const table = screen.getByRole("table");
    expect(table.closest("details")).not.toBeNull();
    expect(within(table).getAllByRole("columnheader")).toHaveLength(2 + CODES.length);
    expect(within(table).getAllByRole("row")).toHaveLength(1 + 3);
  });

  it("toggle_changes_data_not_colours", () => {
    renderHeatmap();
    const negative = screen.getByRole("button", { name: "Negative reviews" });
    const positive = screen.getByRole("button", { name: "Positive reviews" });
    expect(negative.getAttribute("aria-pressed")).toBe("true");
    const legend = screen.getByLabelText("Colour key").innerHTML;
    const before = coloursByShare();
    const performance = document.querySelector<HTMLElement>(
      '[data-aspect-row="performance"] [data-week="2026-09-28"]',
    );
    expect(performance?.dataset.share).toBe("0.25");

    fireEvent.click(positive);

    expect(positive.getAttribute("aria-pressed")).toBe("true");
    expect(negative.getAttribute("aria-pressed")).toBe("false");
    expect(screen.getByRole("grid", { name: /positive reviews/i })).toBeTruthy();
    const after = document.querySelector<HTMLElement>(
      '[data-aspect-row="performance"] [data-week="2026-09-28"]',
    );
    expect(after?.dataset.share).toBe("0");
    // The scale is fixed: the key is unchanged, and a share keeps its colour across the toggle.
    expect(screen.getByLabelText("Colour key").innerHTML).toBe(legend);
    for (const [share, level] of coloursByShare()) {
      if (before.has(share)) expect(level).toBe(before.get(share));
    }
  });

  it("hover_readout_leads_with_the_value", () => {
    renderHeatmap();
    const cell = document.querySelector<HTMLElement>(
      '[data-aspect-row="performance"] [data-week="2026-09-28"]',
    ) as HTMLElement;
    const thin = document.querySelector<HTMLElement>(
      '[data-aspect-row="performance"] [data-week="2026-10-05"]',
    ) as HTMLElement;
    const readout = screen.getByTestId("heatmap-readout");

    fireEvent.pointerEnter(cell);

    expect(readout.querySelector("strong")?.textContent).toBe("25%");
    expect(readout.textContent).toContain("of 40 negative reviews mention performance");
    expect(readout.textContent).toContain("Week of 28 Sep 2026");

    // Keyboard focus shows the same; a thin week says why it has no share.
    fireEvent.focus(thin);
    expect(readout.querySelector("strong")?.textContent).toBe("3");
    expect(readout.textContent).toContain("negative reviews scored, too few for a share");
  });
});

describe("AspectHeatmap, more", () => {
  afterEach(cleanup);

  it("keyboard_entry_survives_a_shorter_range", () => {
    const { rerender } = renderHeatmap();
    const last = document.querySelector<HTMLElement>(
      '[data-aspect-row="performance"] [data-week="2026-10-12"]',
    ) as HTMLElement;
    fireEvent.focus(last);

    rerender(<AspectHeatmap aspects={ASPECTS} from="2026-09-28" to="2026-10-04" />);

    const entries = document.querySelectorAll('[role="gridcell"][tabindex="0"]');
    expect(entries).toHaveLength(1);
  });

  it("weak_aspects_are_marked", () => {
    render(
      <AspectHeatmap
        aspects={ASPECTS}
        from="2026-09-28"
        to="2026-10-18"
        weak={["ux_controls"]}
      />,
    );

    const headers = screen.getAllByRole("rowheader");
    const ux = headers.find((h) => h.textContent?.startsWith("UX and controls"));
    expect(ux?.textContent).toContain("UX and controls*");
    expect(headers[0]?.textContent).toBe("Performance");
    const note = /^\* Detected less reliably: F1 under 0.5 on our test set/;
    expect(screen.getByText(note)).toBeTruthy();
    // One child holds the name and its mark, so a name that wraps on a phone keeps the mark
    // beside its last word (narrow screens lay the label out as a flex row).
    expect(ux?.children).toHaveLength(1);
    expect(ux?.children[0]?.textContent).toBe("UX and controls* (detected less reliably)");
  });
});
