import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { EVAL_REPORT_URL, MethodNote } from "./MethodNote";

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
      "Aspects are detected by a model (S0 20261012T223218Z, macro-F1 0.61 on our test set)",
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
