import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Footer } from "./Footer";

describe("Footer", () => {
  it("attribution_is_rendered", () => {
    render(
      <Footer attribution="Data from Steam" generatedAt="2026-09-29T03:32:00Z" dataVersion={42} />,
    );

    expect(screen.getByText("Data from Steam")).toBeTruthy();
    expect(screen.getByText("Data as of 29 Sep 2026, 03:32 UTC (version 42)")).toBeTruthy();
  });
});
