"use client";

import {
  type ReactNode,
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

// The header names the game once the page's own title has scrolled away under it (design F1).
// The game page wraps its title in `WatchedTitle`; the header reads the name with `useHeaderTitle`.

export const BAR_HEIGHT_PX = 56; // the header's height (globals.css .site-header)

type TitleState = { name: string | null; setName: (name: string | null) => void };

const TitleContext = createContext<TitleState>({ name: null, setName: () => undefined });

export function HeaderTitleProvider({ children }: { children: ReactNode }) {
  const [name, setName] = useState<string | null>(null);
  const value = useMemo(() => ({ name, setName }), [name]);
  return <TitleContext.Provider value={value}>{children}</TitleContext.Provider>;
}

/** The name the header shows, or null while the page's title is in view (or there is none). */
export function useHeaderTitle(): string | null {
  return useContext(TitleContext).name;
}

/** Watches the page's title: gone behind the bar, the header names it. A browser without
 * IntersectionObserver keeps the header as it is. */
export function WatchedTitle({ name, children }: { name: string; children: ReactNode }) {
  const { setName } = useContext(TitleContext);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = ref.current;
    if (element === null || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry) return;
        const gone = !entry.isIntersecting && entry.boundingClientRect.top < BAR_HEIGHT_PX;
        setName(gone ? name : null);
      },
      { rootMargin: `-${BAR_HEIGHT_PX}px 0px 0px 0px` },
    );
    observer.observe(element);
    return () => {
      observer.disconnect();
      setName(null);
    };
  }, [name, setName]);

  return (
    <div ref={ref} className="watched-title">
      {children}
    </div>
  );
}
