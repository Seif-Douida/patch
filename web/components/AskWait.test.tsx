import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AskWait } from "./AskWait";

describe("AskWait", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("counts_seconds_and_changes_phrase", () => {
    render(<AskWait />);
    expect(screen.getByText("Reading your question")).toBeTruthy();
    expect(screen.getByText("0 s")).toBeTruthy();

    act(() => {
      vi.advanceTimersByTime(2_800);
    });

    expect(screen.getByText("Pulling the review numbers")).toBeTruthy();
    expect(screen.getByText("2 s")).toBeTruthy();
  });

  it("the_slow_note_appears_after_eight_seconds", () => {
    render(<AskWait />);
    const note = "A question takes up to a minute, a little longer after a quiet spell.";

    act(() => {
      vi.advanceTimersByTime(7_999);
    });
    expect(screen.queryByText(note)).toBeNull();

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(screen.getByText(note)).toBeTruthy();
  });

  it("the_wait_is_hidden_from_screen_readers", () => {
    const { container } = render(<AskWait />);

    // The form's status region says "Working on your answer." once; the phrases aren't read out.
    const root = container.querySelector(".ask-wait");
    expect(root?.getAttribute("aria-hidden")).toBe("true");
    expect(root?.querySelector("svg .pulse-beat")).not.toBeNull();
  });
});
