import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { beforeAfter, weakAspects } from "@/lib/aspects";
import type { EstimatedHorizon } from "@/lib/effects";
import type { AspectDay, AspectModel, AspectSeries } from "@/lib/types";

import { AspectShift } from "./AspectShift";
import { EstimateMark } from "./EstimateMark";

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
// before 16 Sep and 4 from then on; with fewer, none do. (Ported from PatchAspectTable's tests.)
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

function estimate(effect: number, lo90: number, hi90: number): EstimatedHorizon {
  return { days: 7, ends: "2026-09-23", effect, lo90, hi90, method: "its", flags: [], reason: null };
}

function renderShift(perDay = 10, model: AspectModel | null = null) {
  return render(
    <AspectShift comparison={beforeAfter(september(perDay), "2026-09-16")} weak={weakAspects(model)} />,
  );
}

describe("EstimateMark", () => {
  afterEach(cleanup);

  it("the_mark_fills_only_a_clear_change", () => {
    const { container, rerender } = render(<EstimateMark horizon={estimate(6, 1, 11)} />);
    const svg = () => container.querySelector("svg");
    expect(svg()?.getAttribute("data-verdict")).toBe("up");
    expect(svg()?.getAttribute("aria-hidden")).toBe("true");
    expect(container.querySelector("circle.mark-up")).not.toBeNull();

    rerender(<EstimateMark horizon={estimate(-8, -15, -1)} />);
    expect(container.querySelector("circle.mark-down")).not.toBeNull();

    rerender(<EstimateMark horizon={estimate(2, -3, 7)} />);
    expect(container.querySelector("circle.mark-unclear")).not.toBeNull();

    rerender(<EstimateMark horizon={null} />);
    expect(svg()?.getAttribute("data-verdict")).toBe("none");
    expect(container.querySelector("circle")).toBeNull();
    expect(container.textContent).toBe("No estimate");
  });

  it("a_clipped_range_is_drawn_open", () => {
    const { container, rerender } = render(<EstimateMark horizon={estimate(-95, -130, -60)} />);
    expect(container.querySelector("svg")?.getAttribute("data-clipped")).toBe("low");

    rerender(<EstimateMark horizon={estimate(10, -120, 140)} />);
    expect(container.querySelector("svg")?.getAttribute("data-clipped")).toBe("both");

    rerender(<EstimateMark horizon={estimate(10, -20, 40)} />);
    expect(container.querySelector("svg")?.hasAttribute("data-clipped")).toBe(false);
  });
});

describe("AspectShift", () => {
  afterEach(cleanup);

  it("faded_rows_are_the_ones_that_could_be_chance", () => {
    const { container } = renderShift();

    // Performance moved clearly (+9 to +30); content didn't move (its interval spans 0).
    const row = (code: string) => container.querySelector(`[data-aspect="${code}"]`);
    expect(row("performance")?.getAttribute("data-faded")).toBe("false");
    expect(row("content")?.getAttribute("data-faded")).toBe("true");
    expect(screen.getByText("Faded rows: the change could be chance.")).toBeTruthy();
    expect(
      screen.getByText("With ten aspects, about one change in twenty looks clear by chance alone."),
    ).toBeTruthy();
    // The windows are named: a reader can't weigh "before" and "after" without them.
    expect(
      screen.getByText(
        "What negative reviews mention, before (hollow) and after (solid): the 14 days before " +
          "the patch, and its day and the 13 after.",
      ),
    ).toBeTruthy();
    const figure = container.querySelector("figure");
    expect(figure?.querySelector("figcaption")?.textContent).toBe(
      "Descriptive: the change around a patch, not proof the patch caused it.",
    );
    // The drawing is for the eye; its numbers reach screen readers as the list below.
    expect(figure?.querySelector(".shift-chart")?.getAttribute("aria-hidden")).toBe("true");
    // Labels are page text, so they keep their size on a phone instead of shrinking with a drawing.
    expect(row("performance")?.querySelector(".shift-label")?.tagName).toBe("SPAN");
  });

  it("each_aspect_is_read_out_with_its_interval", () => {
    const { container } = renderShift();

    const items = [...container.querySelectorAll("ul.visually-hidden li")].map((li) => li.textContent);
    expect(items).toHaveLength(CODES.length);
    expect(items[0]).toBe(
      "Performance: 20% before, 40% after, change +20 points (95% interval +9 to +30)",
    );
    // The drawn change, beside each row.
    const change = container.querySelector('[data-aspect="performance"] .shift-change');
    expect(change?.textContent).toBe("+20 pts");
  });

  it("too_few_reviews_show_the_note_not_the_chart", () => {
    const { container } = renderShift(2);

    expect(container.textContent).toContain(
      "Too few scored negative reviews for a comparison (28 before, 28 after)",
    );
    expect(container.querySelector("figure")).toBeNull();
    expect(container.querySelector("details")).toBeNull();
  });

  it("weak_aspects_are_marked", () => {
    const model: AspectModel = {
      family: "s0",
      version: "v1",
      test_macro_f1: 0.61,
      aspect_f1: { performance: 0.87, ux_controls: 0.48 },
    };
    const { container } = renderShift(10, model);

    const table = within(container.querySelector("details") as HTMLElement);
    const ux = table.getByRole("rowheader", { name: /UX and controls/ });
    expect(ux.textContent).toContain("UX and controls*");
    expect(ux.textContent).toContain("detected less reliably");
    expect(table.getByRole("rowheader", { name: /Performance/ }).textContent).toBe("Performance");
    expect(container.querySelector('[data-aspect="ux_controls"] .shift-label')?.textContent).toBe(
      "UX and controls*",
    );
    expect(screen.getByText(/^\* Detected less reliably: F1 under 0.5 on our test set/)).toBeTruthy();
  });

  it("the_numbers_fold_keeps_the_five_columns", () => {
    const { container } = renderShift();

    const fold = container.querySelector("details.table-view") as HTMLElement;
    expect(fold.querySelector("summary")?.textContent).toBe("Show these numbers as a table");
    const headers = within(fold).getAllByRole("columnheader").map((th) => th.textContent);
    expect(headers).toEqual(["Aspect", "Before", "After", "Change (pts)", "95% interval (pts)"]);
    const performance = within(fold).getByRole("row", { name: /Performance/ });
    expect(within(performance).getAllByRole("cell").map((c) => c.textContent)).toEqual([
      "20%",
      "40%",
      "+20",
      "+9 to +30",
    ]);
  });
});
