import type { Metadata } from "next";
import Link from "next/link";

import { GameCharts } from "@/components/GameCharts";
import { WatchedTitle } from "@/components/HeaderTitle";
import { EstimateNote, MethodNote } from "@/components/MethodNote";
import { PatchList } from "@/components/PatchList";
import { weakAspects } from "@/lib/aspects";
import { loadGame, loadIndex } from "@/lib/data";
import { formatCount, formatShare } from "@/lib/series";

type Props = { params: Promise<{ appid: string }> };

export const dynamicParams = false;

export function generateStaticParams() {
  return loadIndex().games.map((game) => ({ appid: String(game.appid) }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { appid } = await params;
  return { title: loadGame(Number(appid)).name };
}

export default async function GamePage({ params }: Props) {
  const { appid } = await params;
  const game = loadGame(Number(appid));
  const index = loadIndex();
  const latest = index.games.find((g) => g.appid === game.appid)?.latest ?? null;
  const aspects = game.aspects && game.aspects.days.length > 0 ? game.aspects : null;
  // The first day with a price snapshot: discounts before it aren't known.
  const discountsFrom = game.series.find((day) => day.discount !== null)?.date ?? null;

  return (
    <>
      <nav aria-label="Breadcrumb" className="breadcrumb">
        <Link href="/">All games</Link>
      </nav>
      <header className="game-header">
        <WatchedTitle name={game.name}>
          <h1>{game.name}</h1>
        </WatchedTitle>
        <p className="game-genres">{game.genres.join(", ")}</p>
        {latest && (
          <p className="game-latest">
            <span className="share-figure">{formatShare(latest.share)}</span>
            <span>
              of {formatCount(latest.n)} reviews in the last {latest.days} days recommend it
            </span>
          </p>
        )}
      </header>

      <GameCharts
        series={game.series}
        patches={game.patches}
        aspects={aspects}
        weak={[...weakAspects(index.aspect_model)]}
      />

      <PatchList
        appid={game.appid}
        patches={game.patches}
        aspects={aspects}
        model={index.aspect_model ?? null}
        causal={index.causal ?? null}
        discountsFrom={discountsFrom}
      />

      {aspects && (
        <div className="method-notes">
          <EstimateNote causal={index.causal} />
          <MethodNote model={index.aspect_model} />
        </div>
      )}
    </>
  );
}
