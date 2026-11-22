import { describe, expect, it } from "vitest";

import { shownHorizon } from "./effects";
import {
  FIRST_ROWS,
  MORE_ROWS,
  VERDICT_LABELS,
  clampToScale,
  patchWindows,
  verdictOf,
  windowName,
  windowTitle,
} from "./patches";
import type { HorizonEffect, Patch, PatchEffect } from "./types";

function horizon(days: number, values: Partial<HorizonEffect> = {}): HorizonEffect {
  return {
    days,
    ends: "2026-10-30",
    effect: -12.3,
    lo90: -20.1,
    hi90: -4.5,
    method: "its",
    flags: [],
    reason: null,
    ...values,
  };
}

const NONE = { effect: null, lo90: null, hi90: null, method: null, reason: "too few reviews" };

function effect(start: string, horizons: HorizonEffect[]): PatchEffect {
  return { start, horizons };
}

function patch(date: string, title: string, values: Partial<Patch> = {}): Patch {
  return {
    date,
    title,
    url: `https://example.com/${date}`,
    type: "patch",
    treatment: true,
    effect: null,
    report: null,
    ...values,
  };
}

// Devoid of Liberty and its follow-up a day later: one estimate, one report (night 63's data).
const DEVOID = effect("2026-10-20", [horizon(7), horizon(14, NONE)]);

describe("patchWindows", () => {
  it("patches_a_week_apart_share_a_row", () => {
    const windows = patchWindows([
      patch("2026-10-21", "Devoid of Liberty: 7.2.1", { effect: DEVOID, report: "b7" }),
      patch("2026-10-20", "Devoid of Liberty: 7.2.0", { effect: DEVOID, report: "b7" }),
    ]);

    expect(windows).toHaveLength(1);
    const [window] = windows;
    expect(window?.patches.map((p) => p.date)).toEqual(["2026-10-20", "2026-10-21"]);
    expect(window?.title).toBe("Devoid of Liberty: 7.2.0 and 7.2.1");
    expect(window?.start).toBe("2026-10-20");
    expect(window?.verdict).toBe("down");
    expect(window?.report).toBe("b7");
  });

  it("windows_are_newest_first_and_posts_are_left_out", () => {
    const windows = patchWindows([
      patch("2026-09-01", "Escalation of Freedom", { effect: effect("2026-09-01", [horizon(7)]) }),
      patch("2026-09-15", "Dev blog", { type: "other", treatment: false }),
      patch("2026-10-02", "Hotfix 7.1.4", { effect: effect("2026-10-02", [horizon(7)]) }),
    ]);

    expect(windows.map((w) => w.title)).toEqual(["Hotfix 7.1.4", "Escalation of Freedom"]);
  });

  it("a_patch_without_an_effect_is_a_row_of_its_own", () => {
    const missing = patch("2026-08-01", "Old patch");
    delete missing.effect;
    const windows = patchWindows([missing, patch("2026-08-03", "Older data", { effect: null })]);

    expect(windows.map((w) => [w.start, w.verdict, w.horizon])).toEqual([
      ["2026-08-03", "none", null],
      ["2026-08-01", "none", null],
    ]);
  });

  it("a_window_takes_its_report_from_any_of_its_patches", () => {
    const [window] = patchWindows([
      patch("2026-10-20", "Devoid of Liberty: 7.2.0", { effect: DEVOID, report: null }),
      patch("2026-10-21", "Devoid of Liberty: 7.2.1", { effect: DEVOID, report: "b7" }),
    ]);

    expect(window?.report).toBe("b7");
  });
});

describe("windowTitle", () => {
  it("titles_join_like_a_sentence", () => {
    expect(windowTitle(["Hotfix 7.1.4"])).toBe("Hotfix 7.1.4");
    expect(
      windowTitle([
        "Devoid of Liberty: 7.2.0",
        "Devoid of Liberty: 7.2.1",
        "Devoid of Liberty: 7.2.2",
      ]),
    ).toBe("Devoid of Liberty: 7.2.0, 7.2.1 and 7.2.2");
    expect(windowTitle(["Escalation of Freedom", "Hotfix 7.1.4"])).toBe(
      "Escalation of Freedom and 1 more",
    );
    expect(windowTitle(["Hotfix", "Hotfix"])).toBe("Hotfix and 1 more");
  });
});

describe("estimates", () => {
  it("the_row_shows_fourteen_days_when_there_are_fourteen", () => {
    expect(shownHorizon(effect("2026-10-20", [horizon(7), horizon(14)]))?.days).toBe(14);
    expect(shownHorizon(effect("2026-10-20", [horizon(7), horizon(14, NONE)]))?.days).toBe(7);
    expect(shownHorizon(effect("2026-10-20", [horizon(7, NONE), horizon(14, NONE)]))).toBeNull();
    expect(shownHorizon(null)).toBeNull();
    expect(shownHorizon(undefined)).toBeNull();
  });

  it("the_verdict_needs_the_whole_range_on_one_side", () => {
    const shown = (lo90: number, hi90: number) =>
      shownHorizon(effect("2026-10-20", [horizon(7, { effect: (lo90 + hi90) / 2, lo90, hi90 })]));

    expect(verdictOf(shown(0.5, 3))).toBe("up");
    expect(verdictOf(shown(-4, -0.1))).toBe("down");
    expect(verdictOf(shown(0, 2))).toBe("unclear");
    expect(verdictOf(shown(-2, 0))).toBe("unclear");
    expect(verdictOf(null)).toBe("none");
    expect(VERDICT_LABELS).toEqual({
      up: "Clearly up",
      down: "Clearly down",
      unclear: "No clear change",
      none: "No estimate",
    });
  });

  it("an_effect_without_its_range_is_no_estimate", () => {
    const [window] = patchWindows([
      patch("2026-10-20", "Hotfix 7.2.3", {
        effect: effect("2026-10-20", [horizon(7, { effect: 5, lo90: null })]),
      }),
    ]);

    expect(window?.horizon).toBeNull();
    expect(window?.verdict).toBe("none");
  });

  it("ranges_beyond_the_scale_are_clipped", () => {
    expect(clampToScale(-130)).toEqual({ value: -100, clipped: true });
    expect(clampToScale(130)).toEqual({ value: 100, clipped: true });
    expect(clampToScale(50)).toEqual({ value: 50, clipped: false });
    expect(clampToScale(-100)).toEqual({ value: -100, clipped: false });
  });
});

describe("windowName", () => {
  it("the_row_name_reads_the_verdict_and_its_numbers", () => {
    const [window] = patchWindows([
      patch("2026-10-20", "Devoid of Liberty: 7.2.0", { effect: DEVOID }),
      patch("2026-10-21", "Devoid of Liberty: 7.2.1", { effect: DEVOID }),
    ]);

    expect(window && windowName(window)).toBe(
      "20 Oct 2026, Devoid of Liberty: 7.2.0 and 7.2.1. Clearly down: −12.3 points, " +
        "90% range −20.1 to −4.5, over 7 days",
    );
  });

  it("a_row_without_an_estimate_says_so", () => {
    const [window] = patchWindows([
      patch("2026-10-20", "Hotfix 7.2.3", { effect: effect("2026-10-20", [horizon(7, NONE)]) }),
    ]);

    expect(window && windowName(window)).toBe("20 Oct 2026, Hotfix 7.2.3. No estimate");
    expect([FIRST_ROWS, MORE_ROWS]).toEqual([8, 10]);
  });
});
