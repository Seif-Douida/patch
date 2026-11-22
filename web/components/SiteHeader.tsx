"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { useHeaderTitle } from "@/components/HeaderTitle";
import { PulseMark } from "@/components/PulseMark";
import { NAV_LINKS, currentSection } from "@/lib/site";

/**
 * The bar that stays at the top (design F1): the wordmark, the game's name once its title has
 * scrolled away, and the menu with the current page marked. A soft shadow shows once the page has
 * moved under it.
 */
export function SiteHeader() {
  const section = currentSection(usePathname());
  const here = useHeaderTitle();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const update = () => setScrolled(window.scrollY > 0);
    update();
    window.addEventListener("scroll", update, { passive: true });
    return () => window.removeEventListener("scroll", update);
  }, []);

  return (
    <header className="site-header" data-scrolled={scrolled ? "true" : "false"}>
      <div className="site-header-inner">
        <Link href="/" className="wordmark">
          <PulseMark className="wordmark-mark" />
          PatchPulse
        </Link>
        {here !== null && (
          <span className="header-here" aria-hidden="true">
            / <b>{here}</b>
          </span>
        )}
        <nav aria-label="Site" className="site-menu">
          {NAV_LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              prefetch={link.section === "games" ? undefined : false}
              aria-current={link.section === section ? "page" : undefined}
            >
              {link.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
