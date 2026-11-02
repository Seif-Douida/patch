import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { SiteHeader } from "./SiteHeader";

describe("SiteHeader", () => {
  afterEach(cleanup);

  it("the_menu_offers_the_games_and_ask", () => {
    render(<SiteHeader />);

    const menu = within(screen.getByRole("navigation", { name: "Site" }));
    const links = menu.getAllByRole("link").map((link) => [link.textContent, link.getAttribute("href")]);
    // Next adds the trailing slash in the build (trailingSlash: true), not under Vitest.
    expect(links.map(([text]) => text)).toEqual(["Games", "Ask"]);
    expect(links[1]?.[1]).toMatch(/^\/ask\/?$/);
  });
});
