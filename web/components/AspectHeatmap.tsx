"use client";

import { type CSSProperties, type KeyboardEvent, useMemo, useRef, useState } from "react";

import { WeakMark } from "@/components/PatchAspectTable";
import {
  MIN_WEEK_SCORED,
  type Polarity,
  SHARE_LEVELS,
  WEAK_NOTE,
  type Week,
  aspectLabel,
  shareLevel,
  weeklyAspectShares,
} from "@/lib/aspects";
import { formatCount, formatDate, formatShare } from "@/lib/series";
import type { AspectSeries } from "@/lib/types";

const POLARITIES: { value: Polarity; label: string }[] = [
  { value: "neg", label: "Negative reviews" },
  { value: "pos", label: "Positive reviews" },
];

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

type Cell = { row: number; col: number };

function round4(value: number): number {
  return Math.round(value * 1e4) / 1e4;
}

/** "performance", "stability and bugs", but "UX and controls": for use mid-sentence. */
function inSentence(label: string): string {
  return /^[A-Z]{2}/.test(label) ? label : label.charAt(0).toLowerCase() + label.slice(1);
}

/** The week's column label: the month that starts in it (with the year on long ranges). */
function axisLabel(week: Week, weeks: number): string | null {
  const start = Date.parse(`${week.start}T00:00:00Z`);
  for (let offset = 0; offset < 7; offset += 1) {
    const day = new Date(start + offset * 86_400_000);
    if (day.getUTCDate() !== 1) continue;
    const month = day.getUTCMonth();
    if (weeks <= 30) return MONTHS[month] ?? null;
    return month % 3 === 0 ? `${MONTHS[month]} ${day.getUTCFullYear()}` : null;
  }
  return null;
}

function levelRange(level: number): string {
  const lower = level === 1 ? null : SHARE_LEVELS[level - 2];
  const upper = SHARE_LEVELS[level - 1];
  if (lower === null || lower === undefined) return `Under ${formatShare(upper ?? 0)}`;
  if (upper === undefined) return `${formatShare(lower)} or more`;
  return `${formatShare(lower)} to ${formatShare(upper)}`;
}

function Readout({
  week,
  code,
  share,
  noun,
}: {
  week: Week;
  code: string;
  share: number | null;
  noun: string;
}) {
  return (
    <>
      {share === null ? (
        <>
          <strong>{formatCount(week.n)}</strong> {noun} reviews scored, too few for a share
        </>
      ) : (
        <>
          <strong>{formatShare(share)}</strong> of {formatCount(week.n)} {noun} reviews mention{" "}
          {inSentence(aspectLabel(code))}
        </>
      )}
      <span className="readout-week">Week of {formatDate(week.start)}</span>
    </>
  );
}

export function AspectHeatmap({
  aspects,
  from,
  to,
  weak = [],
}: {
  aspects: AspectSeries;
  from: string;
  to: string;
  weak?: readonly string[]; // aspects the model detects less reliably
}) {
  const [polarity, setPolarity] = useState<Polarity>("neg");
  const [active, setActive] = useState<Cell | null>(null);
  const grid = useRef<HTMLDivElement>(null);
  const weeks = useMemo(
    () => weeklyAspectShares(aspects, polarity, { from, to }),
    [aspects, polarity, from, to],
  );
  const noun = polarity === "neg" ? "negative" : "positive";
  const rows = aspects.codes.length;
  // The keyboard enters the grid on the last active cell, or the latest week of the first
  // aspect; a shorter range can leave the last active week outside the grid, so it's clamped.
  const inRange = active !== null && active.col < weeks.length ? active : null;
  const current = inRange ?? { row: Math.min(active?.row ?? 0, rows - 1), col: weeks.length - 1 };
  const shown = inRange ? weeks[inRange.col] : undefined;

  function move(event: KeyboardEvent<HTMLDivElement>, cell: Cell) {
    const steps: Record<string, Cell> = {
      ArrowRight: { row: cell.row, col: Math.min(weeks.length - 1, cell.col + 1) },
      ArrowLeft: { row: cell.row, col: Math.max(0, cell.col - 1) },
      ArrowDown: { row: Math.min(rows - 1, cell.row + 1), col: cell.col },
      ArrowUp: { row: Math.max(0, cell.row - 1), col: cell.col },
      Home: { row: cell.row, col: 0 },
      End: { row: cell.row, col: weeks.length - 1 },
    };
    const next = steps[event.key];
    if (!next) return;
    event.preventDefault();
    setActive(next);
    grid.current
      ?.querySelector<HTMLElement>(`[data-row="${next.row}"][data-col="${next.col}"]`)
      ?.focus();
  }

  return (
    <>
      <figure className="chart chart-aspects">
        <figcaption>
          <h2>What reviews talk about, week by week</h2>
          <p className="chart-note">
            The share of each week&apos;s {noun} reviews that mention each aspect. Weeks with
            fewer than {MIN_WEEK_SCORED} scored {noun} reviews are hatched.
          </p>
        </figcaption>

        <div className="range-filter polarity-filter" role="group" aria-label="Reviews shown">
          {POLARITIES.map((option) => (
            <button
              key={option.value}
              type="button"
              aria-pressed={polarity === option.value}
              onClick={() => setPolarity(option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>

        <div className="heatmap-scroll" style={{ "--weeks": weeks.length } as CSSProperties}>
          <div
            ref={grid}
            role="grid"
            aria-label={`Share of ${noun} reviews mentioning each aspect, by week`}
            className="heatmap"
            onPointerLeave={() => setActive(null)}
          >
            {aspects.codes.map((code, row) => (
              <div role="row" key={code} className="heatmap-row" data-aspect-row={code}>
                <div role="rowheader" className="heatmap-label">
                  {/* One child: narrow screens lay the label out as a flex row, and a name that
                      wraps must keep its mark beside its last word. */}
                  <span>
                    {aspectLabel(code)}
                    {weak.includes(code) && <WeakMark />}
                  </span>
                </div>
                {weeks.map((week, col) => {
                  const share = week.shares[row] ?? null;
                  const level = shareLevel(share);
                  const isActive = active?.row === row && active.col === col;
                  const where = `${aspectLabel(code)}, week of ${formatDate(week.start)}`;
                  const label =
                    share === null
                      ? `${where}: ${week.n} ${noun} reviews scored, too few for a share`
                      : `${where}: ${formatShare(share)} of ${week.n} ${noun} reviews`;
                  return (
                    <div
                      key={week.start}
                      role="gridcell"
                      aria-label={label}
                      tabIndex={current.row === row && current.col === col ? 0 : -1}
                      className={[
                        "heat-cell",
                        level === null ? "heat-thin" : `heat-${level}`,
                        isActive ? "is-active" : "",
                      ].join(" ")}
                      data-row={row}
                      data-col={col}
                      data-week={week.start}
                      data-share={share === null ? undefined : String(round4(share))}
                      data-level={level ?? undefined}
                      data-thin={share === null ? "" : undefined}
                      onPointerEnter={() => setActive({ row, col })}
                      onFocus={() => setActive({ row, col })}
                      onKeyDown={(event) => move(event, { row, col })}
                    />
                  );
                })}
              </div>
            ))}
          </div>
          <div className="heatmap-axis" aria-hidden="true">
            <span />
            {weeks.map((week) => (
              <span key={week.start}>{axisLabel(week, weeks.length)}</span>
            ))}
          </div>
        </div>

        <p className="heatmap-readout" data-testid="heatmap-readout" aria-live="polite">
          {inRange && shown ? (
            <Readout
              week={shown}
              code={aspects.codes[inRange.row] ?? ""}
              share={shown.shares[inRange.row] ?? null}
              noun={noun}
            />
          ) : (
            <span className="day-quiet">
              Point at a week for its numbers, or move through them with the arrow keys.
            </span>
          )}
        </p>

        <ul className="chart-key heatmap-key" aria-label="Colour key">
          {[1, 2, 3, 4, 5].map((level) => (
            <li key={level}>
              <span className={`key-heat heat-${level}`} aria-hidden="true" />
              {levelRange(level)}
            </li>
          ))}
          <li>
            <span className="key-heat heat-thin" aria-hidden="true" />
            Under {MIN_WEEK_SCORED} scored reviews
          </li>
        </ul>
        {aspects.codes.some((code) => weak.includes(code)) && (
          <p className="chart-note">{WEAK_NOTE}</p>
        )}
      </figure>

      <details className="table-view">
        <summary>Show the weekly aspect numbers ({noun} reviews)</summary>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th scope="col">Week of</th>
                <th scope="col">Scored</th>
                {aspects.codes.map((code) => (
                  <th scope="col" key={code}>
                    {aspectLabel(code)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...weeks].reverse().map((week) => (
                <tr key={week.start}>
                  <td>{formatDate(week.start)}</td>
                  <td>{formatCount(week.n)}</td>
                  {week.shares.map((share, i) => (
                    <td key={aspects.codes[i]}>{formatShare(share)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </>
  );
}
