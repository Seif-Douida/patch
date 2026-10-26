import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import { ReportView } from "@/components/ReportView";
import { loadGame, loadIndex, loadReport } from "@/lib/data";
import { bundlePatches, joinWords } from "@/lib/reports";

type Params = { appid: string; bundle: string };
type Props = { params: Promise<Params> };

export const dynamicParams = false;

// A static export needs at least one page per dynamic route. Until the first report is published,
// the route has this one, which is a 404.
export const NO_REPORTS = "none";

/**
 * A page per report a patch links (the export writes only those), for every game: both params
 * come from here, since the game's segment sets its params in its page, not a layout.
 */
export function generateStaticParams(): Params[] {
  const games = loadIndex().games;
  const params = games.flatMap((summary) => {
    const bundles = loadGame(summary.appid)
      .patches.map((patch) => patch.report)
      .filter((bundle): bundle is string => Boolean(bundle));
    return [...new Set(bundles)].sort().map((bundle) => ({ appid: String(summary.appid), bundle }));
  });
  if (params.length > 0) return params;
  return [{ appid: String(games[0]?.appid ?? 0), bundle: NO_REPORTS }];
}

function heading(appid: string, bundle: string): string {
  const titles = bundlePatches(loadGame(Number(appid)).patches, bundle).map((p) => p.title);
  return titles.length > 0 ? `After ${joinWords(titles)}` : "Report";
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { appid, bundle } = await params;
  if (bundle === NO_REPORTS) return {};
  return { title: `${heading(appid, bundle)}, ${loadGame(Number(appid)).name}` };
}

export default async function ReportPage({ params }: Props) {
  const { appid, bundle } = await params;
  if (bundle === NO_REPORTS) notFound();
  const game = loadGame(Number(appid));
  const { report } = loadReport(bundle);
  if (report.appid !== game.appid) notFound();

  return (
    <>
      <nav aria-label="Breadcrumb" className="breadcrumb">
        <Link href="/">All games</Link>
        <span aria-hidden="true"> / </span>
        <Link href={`/games/${game.appid}/`} prefetch={false}>
          {game.name}
        </Link>
      </nav>
      <header className="report-header">
        <h1>{heading(appid, bundle)}</h1>
      </header>
      <ReportView report={report} />
    </>
  );
}
