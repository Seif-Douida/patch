import Link from "next/link";

import { formatDate } from "@/lib/series";
import { NAV_LINKS, SITE_DESCRIPTION } from "@/lib/site";

type Props = {
  attribution: string;
  generatedAt: string; // UTC, "2026-09-29T03:32:00Z"
  dataVersion: number;
};

/** "Data from Steam" or "Data from Steam." both end with exactly one full stop. */
function sentence(text: string): string {
  return text.endsWith(".") ? text : `${text}.`;
}

/** Every page's footer (design F2): what the site is, its pages, its source, and the data's age. */
export function Footer({ attribution, generatedAt, dataVersion }: Props) {
  const asOf = `${formatDate(generatedAt.slice(0, 10))}, ${generatedAt.slice(11, 16)} UTC`;
  return (
    <footer className="site-footer">
      <div className="site-footer-inner">
        <div className="footer-columns">
          <div className="footer-about">
            <p className="footer-heading">PatchPulse</p>
            <p>{SITE_DESCRIPTION}</p>
          </div>
          <div>
            <p className="footer-heading">Pages</p>
            <nav aria-label="Footer">
              <ul>
                {NAV_LINKS.map((link) => (
                  <li key={link.href}>
                    <Link href={link.href} prefetch={link.section === "games" ? undefined : false}>
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </nav>
          </div>
          <div>
            <p className="footer-heading">Project</p>
            <ul>
              <li>
                <a href="https://github.com/Seif-Douida/patch">Source and method on GitHub</a>
              </li>
            </ul>
          </div>
        </div>
        <div className="footer-base">
          <p>{sentence(attribution)}</p>
          <p>{`Data as of ${asOf} (version ${dataVersion})`}</p>
        </div>
      </div>
    </footer>
  );
}
