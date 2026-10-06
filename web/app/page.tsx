import Link from "next/link";

import { Pulse } from "@/components/Pulse";
import { loadGame, loadIndex } from "@/lib/data";
import { fillDays, formatCount, formatShare, lastDays, rollingShare } from "@/lib/series";

const PULSE_DAYS = 90;

export default function Home() {
  const index = loadIndex();
  const games = index.games.map((game) => {
    const series = fillDays(loadGame(game.appid).series);
    return { game, pulse: lastDays(rollingShare(series, 7), PULSE_DAYS) };
  });

  return (
    <>
      <section className="intro">
        <h1>How players reacted to each patch</h1>
        <p>
          PatchPulse reads every Steam review of these games each night and tracks the share that
          recommend the game. Open a game to see that share day by day, with its patches and
          Steam sales marked.
        </p>
      </section>

      <table className="game-list">
        <caption className="visually-hidden">Tracked games</caption>
        <thead>
          <tr>
            <th scope="col">Game</th>
            <th scope="col">Last {PULSE_DAYS} days</th>
            <th scope="col">Recommend it (30 days)</th>
          </tr>
        </thead>
        <tbody>
          {games.map(({ game, pulse }) => (
            <tr key={game.appid}>
              <th scope="row">
                {/* No prefetch: Next's static export misnames the prefetch files of dynamic
                    routes, so prefetching only produces 404s; a click loads the page normally. */}
                <Link href={`/games/${game.appid}/`} className="game-name" prefetch={false}>
                  {game.name}
                </Link>
                <span className="game-genres">{game.genres.join(", ")}</span>
              </th>
              <td className="game-pulse">
                <Pulse points={pulse} />
              </td>
              <td className="game-share">
                <span className="share-figure">{formatShare(game.latest?.share ?? null)}</span>
                <span className="share-count">
                  {game.latest ? `of ${formatCount(game.latest.n)} reviews` : "no reviews yet"}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
