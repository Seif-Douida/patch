import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SiteHeader } from "./SiteHeader";

let pathname: string | null = "/";
vi.mock("next/navigation", () => ({ usePathname: () => pathname }));

describe("SiteHeader", () => {
  afterEach(() => {
    cleanup();
    pathname = "/";
    Object.defineProperty(window, "scrollY", { value: 0, configurable: true });
  });

  it("the_menu_offers_the_games_and_ask", () => {
    render(<SiteHeader />);

    const menu = within(screen.getByRole("navigation", { name: "Site" }));
    const links = menu.getAllByRole("link").map((link) => [link.textContent, link.getAttribute("href")]);
    // Next adds the trailing slash in the build (trailingSlash: true), not under Vitest.
    expect(links.map(([text]) => text)).toEqual(["Games", "Ask", "Monitoring"]);
    expect(links[1]?.[1]).toMatch(/^\/ask\/?$/);
  });

  it("links_monitoring_from_the_header", () => {
    render(<SiteHeader />);

    const link = screen.getByRole("link", { name: "Monitoring" });
    expect(link.getAttribute("href")).toMatch(/^\/monitoring\/?$/);
  });

  it("marks_the_current_page", () => {
    pathname = "/games/900001/";
    render(<SiteHeader />);

    const menu = within(screen.getByRole("navigation", { name: "Site" }));
    expect(menu.getByRole("link", { name: "Games" }).getAttribute("aria-current")).toBe("page");
    expect(menu.getByRole("link", { name: "Ask" }).getAttribute("aria-current")).toBeNull();
    expect(menu.getByRole("link", { name: "Monitoring" }).getAttribute("aria-current")).toBeNull();
  });

  it("the_wordmark_keeps_its_name", () => {
    render(<SiteHeader />);

    // The pulse mark beside it is decorative: the link reads "PatchPulse" and nothing more.
    const wordmark = screen.getByRole("link", { name: "PatchPulse" });
    expect(wordmark.getAttribute("href")).toBe("/");
    expect(wordmark.querySelector("svg")?.getAttribute("aria-hidden")).toBe("true");
  });

  it("the_bar_gains_its_shadow_once_scrolled", () => {
    const { container } = render(<SiteHeader />);
    const bar = container.querySelector("header.site-header");
    expect(bar?.getAttribute("data-scrolled")).toBe("false");

    Object.defineProperty(window, "scrollY", { value: 120, configurable: true });
    act(() => {
      fireEvent.scroll(window);
    });

    expect(bar?.getAttribute("data-scrolled")).toBe("true");
  });
});
