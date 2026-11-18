import { describe, expect, it } from "vitest";

import { retrainSummary } from "./monitoring";

describe("retrainSummary", () => {
  it("says_a_passing_challenger_was_proposed_not_promoted", () => {
    // Final review (18 Nov): "promote" means the challenger passed the rule and a pull request
    // was opened; it is the champion only once that pull request is merged.
    const summary = retrainSummary({
      last_request: { date: "2026-11-23", reason: "drift" },
      last_decision: { date: "2026-11-23", decision: "promote", recent_gain: 0.021, n: 812 },
    });

    expect(summary).toMatch(/^Proposed a challenger for promotion \(a pull request\)/);
    expect(summary).not.toMatch(/Promoted/);
  });
});
