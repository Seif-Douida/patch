import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { EVAL_REPORT_URL, EstimateNote, MethodNote, REPO_URL } from "./MethodNote";

describe("MethodNote", () => {
  afterEach(cleanup);

  it("method_note_shows_the_model_and_its_score", () => {
    render(
      <MethodNote
        model={{ family: "s0", version: "20261012T223218Z", test_macro_f1: 0.6139 }}
      />,
    );

    const note = screen.getByText(/Aspects are detected by a model/);
    expect(note.textContent).toContain(
      "Aspects are detected by a model (S0 20261012T223218Z, macro-F1 0.61 on our test set of " +
        "English and French reviews)",
    );
    const link = screen.getByRole("link", { name: "How the model was evaluated" });
    expect(link.getAttribute("href")).toBe(EVAL_REPORT_URL);
    expect(EVAL_REPORT_URL).toBe(
      "https://github.com/Seif-Douida/patch/blob/main/reports/aspects/phase-2-eval.md",
    );
  });

  it("no_model_no_note", () => {
    const { container } = render(<MethodNote model={null} />);
    expect(container.textContent).toBe("");
  });
});

describe("EstimateNote", () => {
  afterEach(cleanup);

  it("method_note_links_the_benchmark", () => {
    render(
      <EstimateNote
        causal={{
          default_method: "synth",
          effects_version: "e2",
          benchmark_report: "reports/causal/benchmark.md",
        }}
      />,
    );

    const text = document.body.textContent ?? "";
    // What the estimate is, what the range means, what it can't rule out, and the method.
    expect(text).toContain(
      "from the game's own history (interrupted time series) or from similar games without a " +
        "patch of their own (synthetic control)",
    );
    expect(text).toContain("The 90% range should hold the true effect about 9 times in 10");
    expect(text).toContain("can't rule out something else that changed at the same time");
    expect(text).toContain("Estimates use synthetic control where it's possible");
    const link = screen.getByRole("link", { name: "How the methods were tested" });
    expect(link.getAttribute("href")).toBe(`${REPO_URL}/blob/main/reports/causal/benchmark.md`);
    expect(REPO_URL).toBe("https://github.com/Seif-Douida/patch");
  });

  it("says_the_methods_are_not_yet_compared", () => {
    render(
      <EstimateNote
        causal={{ default_method: "its", effects_version: "e2", benchmark_report: null }}
      />,
    );

    expect(document.body.textContent).toContain(
      "Estimates use interrupted time series for now; the comparison of the two methods isn't " +
        "published yet.",
    );
    expect(screen.queryByRole("link")).toBeNull();
  });

  it("no_estimates_no_note", () => {
    const { container } = render(<EstimateNote causal={null} />);
    expect(container.textContent).toBe("");
  });
});
