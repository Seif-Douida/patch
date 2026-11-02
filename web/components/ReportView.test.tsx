import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { Hypothesis, PublicReport, ReportClaim } from "@/lib/types";

import { ReportView } from "./ReportView";

function hypothesis(code: Hypothesis["code"], status: Hypothesis["status"]): Hypothesis {
  return { code, status, rule: `${code}.test`, evidence: {} };
}

function claim(id: string, text: string, extra: Partial<ReportClaim> = {}): ReportClaim {
  return {
    claim_id: id,
    text,
    claim_type: "comparative",
    value: null,
    unit: null,
    reworded: false,
    supported_by: 0,
    ...extra,
  };
}

// Invented: a sale over the patch's days, the patch's own effect not established.
function report(extra: Partial<PublicReport> = {}): PublicReport {
  return {
    report_id: "5011-14d",
    appid: 900002,
    scope: "bundle",
    bundle_id: "5011",
    horizon: 14,
    question: null,
    window: { start: "2026-07-02", end: "2026-07-15" },
    data_version: 42,
    model_ids: { gather: "model-a", draft: "model-a", summary: "model-b" },
    prompt_version: "5a3c9e1f0b72",
    summary: "Reviews stayed about as positive after the patch; a sale ran over the same days.",
    confidence: "medium",
    basic: false,
    // Not in order: the page puts them in order.
    hypotheses: [
      hypothesis("H3", "rejected"),
      hypothesis("H1", "inconclusive"),
      hypothesis("H5", "inconclusive"),
      hypothesis("H2", "supported"),
      hypothesis("H4", "rejected"),
    ],
    findings: [
      { title: "The patch", hypothesis: "H1", claim_ids: ["c2"] },
      { title: "The sale", hypothesis: "H2", claim_ids: ["c1"] },
    ],
    claims: [
      claim("c1", "Reviews per day doubled during the sale, from 45 to 92.", { supported_by: 3 }),
      claim("c2", "The data are consistent with the patch moving the share by about -0.6 points.", {
        claim_type: "causal",
        reworded: true,
        supported_by: 1,
      }),
    ],
    withheld: {},
    rule_pass_rates: { V1: 1, V2: 1 },
    ...extra,
  };
}

function precedes(first: Element, second: Element): boolean {
  return Boolean(first.compareDocumentPosition(second) & Node.DOCUMENT_POSITION_FOLLOWING);
}

describe("ReportView", () => {
  afterEach(cleanup);

  it("tiles_come_first_in_hypothesis_order", () => {
    render(<ReportView report={report()} />);

    const tiles = within(screen.getByRole("list", { name: "Explanations checked" }));
    const items = tiles.getAllByRole("listitem").map((item) => item.textContent);
    expect(items).toEqual([
      "The patchInconclusive",
      "Price or saleSupported",
      "Off-topicRuled out",
      "Who reviewedRuled out",
      "NoiseInconclusive",
    ]);
    const summary = screen.getByText(/^Reviews stayed about as positive/);
    expect(precedes(screen.getByRole("list", { name: "Explanations checked" }), summary)).toBe(
      true,
    );
  });

  it("claims_sit_under_their_explanation", () => {
    render(<ReportView report={report()} />);

    const patch = screen.getByRole("region", { name: "The patch: inconclusive" });
    const sale = screen.getByRole("region", { name: "A price change or sale: supported" });
    expect(within(patch).getByText(/consistent with the patch/)).toBeTruthy();
    expect(within(patch).queryByText(/doubled during the sale/)).toBeNull();
    expect(within(sale).getByText(/doubled during the sale/)).toBeTruthy();
    expect(within(sale).getByText("supported by 3 reviews")).toBeTruthy();
    expect(within(patch).getByText("supported by 1 review")).toBeTruthy();
    // The patch comes before the sale, in hypothesis order; explanations without a claim get no
    // section of their own.
    expect(precedes(patch, sale)).toBe(true);
    expect(screen.queryByRole("region", { name: /Off-topic reviews/ })).toBeNull();
    // A claim the findings don't file still shows, at the end.
    cleanup();
    render(
      <ReportView
        report={report({
          claims: [...report().claims, claim("c3", "The game sold 40% more copies.")],
        })}
      />,
    );
    const other = screen.getByRole("region", { name: "Also found" });
    expect(within(other).getByText("The game sold 40% more copies.")).toBeTruthy();
  });

  it("withheld_shows_kinds_and_counts_only", () => {
    render(<ReportView report={report({ withheld: { V2: 2, V4: 1, V7: 1 } })} />);

    const withheld = screen.getByText("Withheld by the verifier (4)").closest("details");
    expect(withheld).not.toBeNull();
    const lines = within(withheld as HTMLElement)
      .getAllByRole("listitem")
      .map((item) => item.textContent);
    expect(lines).toEqual([
      "2 claims with a number that didn't match the data",
      "1 claim of cause that the estimate didn't support",
      "1 claim that repeated a review's own words",
    ]);
    // Folded at the bottom: after every claim.
    expect(precedes(screen.getByText(/doubled during the sale/), withheld as Element)).toBe(true);

    cleanup();
    render(<ReportView report={report()} />);
    expect(screen.queryByText(/Withheld by the verifier/)).toBeNull();
  });

  it("reworded_claims_say_so", () => {
    render(<ReportView report={report()} />);

    const reworded = screen.getByText(/consistent with the patch/).closest("li");
    const plain = screen.getByText(/doubled during the sale/).closest("li");
    expect(within(reworded as HTMLElement).getByText(/^Reworded by the verifier/)).toBeTruthy();
    expect(within(plain as HTMLElement).queryByText(/Reworded/)).toBeNull();
  });

  it("basic_report_says_so", () => {
    render(<ReportView report={report({ basic: true, model_ids: {} })} />);

    const notice = screen.getByRole("note");
    expect(notice.textContent).toMatch(/^Written without the language model/);
    expect(precedes(notice, screen.getByRole("list", { name: "Explanations checked" }))).toBe(
      true,
    );
    expect(screen.getByText(/without a language model, from data version 42/)).toBeTruthy();

    cleanup();
    render(<ReportView report={report()} />);
    expect(screen.queryByRole("note")).toBeNull();
    expect(screen.getByText(/by model-a and model-b, from data version 42/)).toBeTruthy();
    expect(screen.getByText(/prompt version 5a3c9e1f0b72/)).toBeTruthy();
  });
});
