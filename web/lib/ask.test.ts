import { describe, expect, it } from "vitest";

import { PHRASE_MS, SLOW_NOTE, WAIT_PHRASES, WORKING, phraseAt } from "./ask";

describe("the Ask wait", () => {
  it("phrases_play_in_order_then_repeat_from_the_search", () => {
    expect(WAIT_PHRASES).toEqual([
      "Reading your question",
      "Pulling the review numbers",
      "Lining up the patches",
      "Deciphering the patch notes",
      "Weighing possible causes",
      "Searching the reviews",
      "Sifting what players said",
      "Thinking it through",
      "Drafting the answer",
      "Checking every number against the data",
      "Checking each claim",
      "Writing it up",
    ]);
    expect(PHRASE_MS).toBe(2_800);
    expect(phraseAt(0)).toBe("Reading your question");
    expect(phraseAt(2_799)).toBe("Reading your question");
    expect(phraseAt(2_800)).toBe("Pulling the review numbers");
    expect(phraseAt(11 * 2_800)).toBe("Writing it up");
    expect(phraseAt(12 * 2_800)).toBe("Searching the reviews");
    expect(phraseAt(18 * 2_800)).toBe("Writing it up");
    expect(phraseAt(19 * 2_800)).toBe("Searching the reviews");
    expect(SLOW_NOTE).toBe("A question takes up to a minute, a little longer after a quiet spell.");
    expect(WORKING).toBe("Working on your answer.");
  });
});
