import { describe, expect, it } from "vitest";

import { NAV_LINKS, SITE_DESCRIPTION, currentSection } from "./site";

describe("site", () => {
  it("current_section_follows_the_path", () => {
    expect(currentSection("/")).toBe("games");
    expect(currentSection("/games/553850/")).toBe("games");
    expect(currentSection("/games/553850/reports/b1/")).toBe("games");
    expect(currentSection("/ask/")).toBe("ask");
    expect(currentSection("/monitoring/")).toBe("monitoring");
    expect(currentSection("/nope/")).toBeNull();
    expect(currentSection(null)).toBeNull();
  });

  it("the_menu_lists_the_three_pages_in_order", () => {
    expect(NAV_LINKS.map((link) => [link.label, link.href, link.section])).toEqual([
      ["Games", "/", "games"],
      ["Ask", "/ask/", "ask"],
      ["Monitoring", "/monitoring/", "monitoring"],
    ]);
    expect(SITE_DESCRIPTION).toBe(
      "How players' Steam reviews moved around each game's patches, updated every other night.",
    );
  });
});
