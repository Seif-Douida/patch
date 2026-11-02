import Link from "next/link";

/** The wordmark and the site's menu: the games (the home page) and Ask. */
export function SiteHeader() {
  return (
    <header className="site-header">
      <Link href="/" className="wordmark">
        PatchPulse
      </Link>
      <nav aria-label="Site" className="site-menu">
        <Link href="/">Games</Link>
        <Link href="/ask/" prefetch={false}>
          Ask
        </Link>
      </nav>
    </header>
  );
}
