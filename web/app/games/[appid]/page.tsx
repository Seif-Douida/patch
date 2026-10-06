import type { Metadata } from "next";
import Link from "next/link";

import { GameCharts } from "@/components/GameCharts";
import { loadGame, loadIndex } from "@/lib/data";
import { PATCH_LABELS, formatCount, formatDate, formatShare } from "@/lib/series";

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
  const latest = loadIndex().games.find((g) => g.appid === game.appid)?.latest ?? null;
  const posts = [...game.patches].reverse();

  return (
    <>
      <nav aria-label="Breadcrumb" className="breadcrumb">
        <Link href="/">All games</Link>
      </nav>
      <header className="game-header">
        <h1>{game.name}</h1>
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

      <GameCharts series={game.series} patches={game.patches} />

      <section className="posts">
        <h2>Patches and posts</h2>
        {posts.length === 0 ? (
          <p>No official posts in this period.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th scope="col">Date</th>
                <th scope="col">Post</th>
                <th scope="col">Kind</th>
              </tr>
            </thead>
            <tbody>
              {posts.map((post) => (
                <tr key={`${post.date}-${post.url}`}>
                  <td>{formatDate(post.date)}</td>
                  <td>
                    <a href={post.url} rel="noopener noreferrer">
                      {post.title}
                    </a>
                  </td>
                  <td>
                    <span
                      className={post.treatment ? "kind kind-treatment" : "kind"}
                      aria-hidden="true"
                    />
                    {PATCH_LABELS[post.type]}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="method-note">
          Each post is labelled by its title. Major updates and patches are the ones PatchPulse
          tests for an effect on reviews; the rest are shown for context.
        </p>
      </section>
    </>
  );
}
