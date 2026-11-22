import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { DayPoint } from "@/lib/types";

import { GameCharts, Key, SaleMark, VolumeDetails } from "./GameCharts";

// The charts need a browser's layout to draw; these tests only need the props each chart is given.
const calls = vi.hoisted(() => ({
  composed: [] as Record<string, unknown>[],
  line: [] as Record<string, unknown>[],
  bar: [] as Record<string, unknown>[],
}));

vi.mock("recharts", async (importOriginal) => {
  const actual = await importOriginal<typeof import("recharts")>();
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => children,
    ComposedChart: (props: Record<string, unknown>) => {
      calls.composed.push(props);
      return null;
    },
    LineChart: (props: Record<string, unknown>) => {
      calls.line.push(props);
      return null;
    },
    BarChart: (props: Record<string, unknown>) => {
      calls.bar.push(props);
      return null;
    },
  };
});

function day(date: string, n: number, sale = false, discount: number | null = null): DayPoint {
  const pos = Math.round(n * 0.8);
  return { date, n, pos, share: n ? pos / n : null, lo: 0.7, hi: 0.9, sale, discount };
}

// The readout leads with the value in bold, so its line is split across two text nodes.
function paragraph(text: string) {
  return screen.getByText((_, element) => element?.tagName === "P" && element.textContent === text);
}

describe("GameCharts", () => {
  beforeEach(() => {
    calls.composed.length = 0;
    calls.line.length = 0;
    calls.bar.length = 0;
  });
  afterEach(cleanup);

  it("the_reviews_readout_shows_its_own_day", () => {
    const { rerender } = render(<VolumeDetails row={day("2026-09-28", 1234, true, 50)} />);
    expect(screen.getByText("28 Sep 2026")).toBeTruthy();
    expect(paragraph("1,234 reviews")).toBeTruthy();
    expect(screen.getByText("Steam sale, 50% off")).toBeTruthy();

    rerender(<VolumeDetails row={day("2026-09-28", 1)} />);
    expect(paragraph("1 review")).toBeTruthy();
    expect(screen.queryByText(/Steam sale/)).toBeNull();

    rerender(<VolumeDetails row={day("2026-09-28", 5, true)} />);
    expect(screen.getByText("Steam sale")).toBeTruthy();
  });

  it("the_sale_mark_names_a_wide_band_only", () => {
    const { container, rerender } = render(
      <svg>
        <SaleMark viewBox={{ x: 10, y: 8, width: 40, height: 200 }} />
      </svg>,
    );
    expect(container.querySelectorAll("line.sale-edge")).toHaveLength(1);
    expect(container.querySelector("text.sale-label")?.textContent).toBe("Sale");

    rerender(
      <svg>
        <SaleMark viewBox={{ x: 10, y: 8, width: 20, height: 200 }} />
      </svg>,
    );
    expect(container.querySelectorAll("line.sale-edge")).toHaveLength(1);
    expect(container.querySelector("text.sale-label")).toBeNull();
  });

  it("the_charts_are_not_linked", () => {
    render(
      <GameCharts
        series={[day("2026-09-26", 10), day("2026-09-27", 12, true, 30), day("2026-09-28", 9)]}
        patches={[]}
      />,
    );

    // Each chart has its own readout: no syncId ties a hover in one to the other.
    expect(calls.composed.length).toBeGreaterThan(0);
    expect(calls.line.length).toBeGreaterThan(0);
    for (const props of [...calls.composed, ...calls.line]) {
      expect(props).not.toHaveProperty("syncId");
    }
    expect(calls.bar).toHaveLength(0); // reviews per day is a line now
  });

  it("the_daily_table_is_labelled_as_the_charts", () => {
    const { container } = render(
      <GameCharts series={[day("2026-09-27", 12), day("2026-09-28", 9)]} patches={[]} />,
    );

    const fold = container.querySelector("details.table-view");
    expect(fold?.querySelector("summary")?.textContent).toBe("Show these charts as a table");
    const headers = within(fold as HTMLElement)
      .getAllByRole("columnheader")
      .map((th) => th.textContent);
    expect(headers).toEqual(["Date", "Reviews", "Positive", "90% range", "Sale"]);
  });

  it("the_key_shows_the_sale_band", () => {
    render(<Key />);

    const item = screen.getByText("Steam sale").closest("li");
    expect(item?.querySelector(".key-sale")).not.toBeNull();
  });
});
