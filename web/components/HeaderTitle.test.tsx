import { act, cleanup, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { HeaderTitleProvider, WatchedTitle } from "./HeaderTitle";
import { SiteHeader } from "./SiteHeader";

vi.mock("next/navigation", () => ({ usePathname: () => "/games/900001/" }));

type Callback = (entries: Partial<IntersectionObserverEntry>[]) => void;

// Keeps the callback and the options, so a test can say what the browser would have seen.
class StubObserver {
  static last: StubObserver | null = null;
  constructor(
    readonly callback: Callback,
    readonly options?: IntersectionObserverInit,
  ) {
    StubObserver.last = this;
  }
  observe() {}
  disconnect() {}
  unobserve() {}
}

function renderGamePage() {
  return render(
    <HeaderTitleProvider>
      <SiteHeader />
      <WatchedTitle name="Example Arena">
        <h1>Example Arena</h1>
      </WatchedTitle>
    </HeaderTitleProvider>,
  );
}

function see(entry: Partial<IntersectionObserverEntry>) {
  act(() => StubObserver.last?.callback([entry]));
}

describe("HeaderTitle", () => {
  const original = globalThis.IntersectionObserver;
  beforeEach(() => {
    StubObserver.last = null;
    globalThis.IntersectionObserver = StubObserver as unknown as typeof IntersectionObserver;
  });
  afterEach(() => {
    cleanup();
    globalThis.IntersectionObserver = original;
  });

  it("names_the_game_once_its_title_scrolls_away", () => {
    const { container } = renderGamePage();
    expect(container.querySelector(".header-here")).toBeNull();
    // Measured under the bar: the title counts as gone once it's behind the 56 px header.
    expect(StubObserver.last?.options?.rootMargin).toBe("-56px 0px 0px 0px");

    see({ isIntersecting: false, boundingClientRect: { top: -40 } as DOMRectReadOnly });
    expect(container.querySelector(".header-here")?.textContent).toBe("/ Example Arena");

    see({ isIntersecting: true, boundingClientRect: { top: 120 } as DOMRectReadOnly });
    expect(container.querySelector(".header-here")).toBeNull();
  });

  it("the_name_is_hidden_from_screen_readers", () => {
    const { container } = renderGamePage();

    see({ isIntersecting: false, boundingClientRect: { top: -40 } as DOMRectReadOnly });

    expect(container.querySelector(".header-here")?.getAttribute("aria-hidden")).toBe("true");
  });

  it("without_intersection_observer_the_name_stays_hidden", () => {
    // @ts-expect-error: an older browser, or jsdom, has no IntersectionObserver at all
    delete globalThis.IntersectionObserver;

    const { container } = renderGamePage();

    expect(container.querySelector("h1")?.textContent).toBe("Example Arena");
    expect(container.querySelector(".header-here")).toBeNull();
  });
});
