// The site's pages and its one-line description, shared by the header, the footer and the
// metadata, so the three never drift apart.

export const SITE_DESCRIPTION =
  "How players' Steam reviews moved around each game's patches, updated every other night.";

export type Section = "games" | "ask" | "monitoring";

export const NAV_LINKS: readonly { label: string; href: string; section: Section }[] = [
  { label: "Games", href: "/", section: "games" },
  { label: "Ask", href: "/ask/", section: "ask" },
  { label: "Monitoring", href: "/monitoring/", section: "monitoring" },
];

function under(pathname: string, page: string): boolean {
  return pathname === page || pathname.startsWith(`${page}/`);
}

/** The menu entry a page belongs to: a game page and its reports count as Games. */
export function currentSection(pathname: string | null): Section | null {
  if (pathname === null) return null;
  if (pathname === "/" || under(pathname, "/games")) return "games";
  if (under(pathname, "/ask")) return "ask";
  if (under(pathname, "/monitoring")) return "monitoring";
  return null;
}
