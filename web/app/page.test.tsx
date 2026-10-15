import path from "node:path";

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

// The sample data (web/fixtures) is schema 2 with a model, so the aspect column is on.
const FIXTURES = path.resolve(__dirname, "../fixtures");

describe("Home", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllEnvs();
  });

  it("names_the_aspect_column_by_what_it_counts", async () => {
    vi.stubEnv("SITE_DATA_DIR", FIXTURES);
    vi.resetModules(); // the data folder is read when lib/data loads
    const { default: Home } = await import("./page");

    render(<Home />);

    // The model finds mentions, not whether a mention is praise or a complaint.
    const header = screen.getByRole("columnheader", { name: /negative reviews/ });
    expect(header.textContent).toBe("Most mentioned in negative reviews (30 days)");
    expect(document.body.textContent).not.toMatch(/complaint/i);
    expect(
      screen.getByText(/^The most mentioned aspect leaves out off-topic and the aspects/),
    ).toBeTruthy();
  });
});
