import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { Footer } from "./Footer";

function renderFooter(attribution = "Data from Steam") {
  return render(
    <Footer attribution={attribution} generatedAt="2026-09-29T03:32:00Z" dataVersion={42} />,
  );
}

describe("Footer", () => {
  afterEach(cleanup);

  it("attribution_is_rendered", () => {
    renderFooter();

    expect(screen.getByText("Data from Steam.")).toBeTruthy();
    expect(screen.getByText("Data as of 29 Sep 2026, 03:32 UTC (version 42)")).toBeTruthy();
  });

  it("an_attribution_with_its_full_stop_keeps_one", () => {
    renderFooter("Data from Steam.");

    expect(screen.getByText("Data from Steam.")).toBeTruthy();
    expect(screen.queryByText("Data from Steam..")).toBeNull();
  });

  it("footer_has_the_three_columns", () => {
    const { container } = renderFooter();

    const headings = [...container.querySelectorAll(".footer-heading")].map((p) => p.textContent);
    expect(headings).toEqual(["PatchPulse", "Pages", "Project"]);
    expect(
      screen.getByText(
        "How players' Steam reviews moved around each game's patches, updated every other night.",
      ),
    ).toBeTruthy();
    const pages = within(screen.getByRole("navigation", { name: "Footer" }));
    const links = pages.getAllByRole("link").map((link) => [link.textContent, link.getAttribute("href")]);
    expect(links.map(([text]) => text)).toEqual(["Games", "Ask", "Monitoring"]);
    expect(links[0]?.[1]).toBe("/");
    expect(links[1]?.[1]).toMatch(/^\/ask\/?$/);
    expect(links[2]?.[1]).toMatch(/^\/monitoring\/?$/);
    expect(
      screen.getByRole("link", { name: "Source and method on GitHub" }).getAttribute("href"),
    ).toBe("https://github.com/Seif-Douida/patch");
    // Not a heading element: the footer doesn't add to each page's outline.
    expect(within(container).queryAllByRole("heading")).toEqual([]);
  });

  it("no_valve_line", () => {
    renderFooter();

    expect(screen.queryByText(/Valve/)).toBeNull();
  });
});
