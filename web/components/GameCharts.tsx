"use client";

import { useMemo, useState } from "react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  dateTicks,
  fillDays,
  formatCount,
  formatDate,
  formatShare,
  lastDays,
  type Marker,
  placePatchMarkers,
  saleRanges,
  tickLabel,
} from "@/lib/series";
import type { DayPoint, Patch } from "@/lib/types";

const RANGES: { label: string; days: number | null }[] = [
  { label: "Last 90 days", days: 90 },
  { label: "Last year", days: 365 },
  { label: "All", days: null },
];

type Row = DayPoint & { band: [number, number] | null; marker: Marker | null };

function DayDetails({ row }: { row: Row }) {
  return (
    <div className="day-details">
      <p className="day-date">{formatDate(row.date)}</p>
      {row.share === null ? (
        <p>No reviews</p>
      ) : (
        <>
          <p>
            <strong>{formatShare(row.share)}</strong> positive
          </p>
          <p className="day-quiet">
            90% range {formatShare(row.lo)} to {formatShare(row.hi)}
          </p>
          <p className="day-quiet">{formatCount(row.n)} reviews</p>
        </>
      )}
      {row.sale && (
        <p className="day-quiet">Steam sale{row.discount ? `, ${row.discount}% off` : ""}</p>
      )}
      {row.marker && (
        <ul className="day-posts">
          {row.marker.titles.map((title) => (
            <li key={title}>{title}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ShareTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: readonly { payload?: Row }[];
}) {
  const row = payload?.[0]?.payload;
  if (!active || !row) return null;
  return <DayDetails row={row} />;
}

function Key() {
  return (
    <ul className="chart-key" aria-label="Chart key">
      <li>
        <span className="key-line" aria-hidden="true" />
        Positive share
      </li>
      <li>
        <span className="key-band" aria-hidden="true" />
        90% range
      </li>
      <li>
        <span className="key-marker key-marker-treatment" aria-hidden="true" />
        Major update or patch
      </li>
      <li>
        <span className="key-marker" aria-hidden="true" />
        Other post
      </li>
      <li>
        <span className="key-sale" aria-hidden="true" />
        Steam sale
      </li>
    </ul>
  );
}

export function GameCharts({ series, patches }: { series: DayPoint[]; patches: Patch[] }) {
  const filled = useMemo(() => fillDays(series), [series]);
  const [days, setDays] = useState<number | null>(90);

  const { rows, markers, sales } = useMemo(() => {
    const shown = lastDays(filled, days);
    const placed = placePatchMarkers(
      patches,
      shown.map((d) => d.date),
    );
    const byDate = new Map(placed.map((m) => [m.date, m]));
    return {
      rows: shown.map<Row>((d) => ({
        ...d,
        band: d.lo !== null && d.hi !== null ? [d.lo, d.hi] : null,
        marker: byDate.get(d.date) ?? null,
      })),
      markers: placed,
      sales: saleRanges(shown),
    };
  }, [filled, patches, days]);

  if (rows.length === 0) {
    return <p>No reviews have been collected for this game yet. They appear after the next nightly run.</p>;
  }

  // Both charts get the same calendar ticks, so their dates line up.
  const xAxis = (
    <XAxis
      dataKey="date"
      ticks={dateTicks(rows.map((row) => row.date))}
      tickFormatter={(date: string) => tickLabel(date, rows.length)}
      minTickGap={24}
      tickLine={false}
      tickMargin={8}
    />
  );
  const decorations = [
    ...sales.map((s) => (
      <ReferenceArea
        key={`sale-${s.from}`}
        x1={s.from}
        x2={s.to}
        className="sale-band"
        ifOverflow="hidden"
      />
    )),
    ...markers.map((m) => (
      <ReferenceLine
        key={`post-${m.date}`}
        x={m.date}
        className={m.treatment ? "marker marker-treatment" : "marker"}
      />
    )),
  ];

  return (
    <div className="game-charts">
      <div className="range-filter" role="group" aria-label="Time range">
        {RANGES.map((range) => (
          <button
            key={range.label}
            type="button"
            aria-pressed={days === range.days}
            onClick={() => setDays(range.days)}
          >
            {range.label}
          </button>
        ))}
      </div>

      <figure className="chart chart-share">
        <figcaption>
          <h2>Share of each day&apos;s reviews that recommend the game</h2>
          <Key />
        </figcaption>
        <div className="chart-frame chart-frame-share">
          <ResponsiveContainer>
            <ComposedChart data={rows} syncId="pulse" margin={{ top: 8, right: 12, left: 0 }}>
              <CartesianGrid vertical={false} />
              {decorations}
              {xAxis}
              <YAxis
                domain={[0, 1]}
                ticks={[0, 0.25, 0.5, 0.75, 1]}
                tickFormatter={formatShare}
                width={48}
                tickLine={false}
                axisLine={false}
              />
              <Area
                dataKey="band"
                className="share-band"
                isAnimationActive={false}
                activeDot={false}
                connectNulls={false}
              />
              <Line
                dataKey="share"
                className="share-line"
                dot={false}
                activeDot={{ r: 4, className: "share-dot" }}
                isAnimationActive={false}
                connectNulls={false}
              />
              <Tooltip
                content={ShareTooltip}
                cursor={{ className: "crosshair" }}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </figure>

      <figure className="chart chart-volume">
        <figcaption>
          <h2>Reviews per day</h2>
          <p className="chart-note">A day with few reviews gives a wide 90% range above.</p>
        </figcaption>
        <div className="chart-frame chart-frame-volume">
          <ResponsiveContainer>
            <BarChart data={rows} syncId="pulse" margin={{ top: 8, right: 12, left: 0 }}>
              <CartesianGrid vertical={false} />
              {decorations}
              {xAxis}
              <YAxis
                width={48}
                tickLine={false}
                axisLine={false}
                tickFormatter={(n: number) => formatCount(n)}
                allowDecimals={false}
              />
              <Bar
                dataKey="n"
                className="volume-bar"
                maxBarSize={24}
                radius={[2, 2, 0, 0]}
                isAnimationActive={false}
              />
              <Tooltip content={() => null} cursor={{ className: "crosshair" }} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </figure>

      <details className="table-view">
        <summary>Show the daily numbers for this range</summary>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th scope="col">Date</th>
                <th scope="col">Reviews</th>
                <th scope="col">Positive</th>
                <th scope="col">90% range</th>
                <th scope="col">Sale</th>
              </tr>
            </thead>
            <tbody>
              {[...rows].reverse().map((row) => (
                <tr key={row.date}>
                  <td>{formatDate(row.date)}</td>
                  <td>{formatCount(row.n)}</td>
                  <td>{formatShare(row.share)}</td>
                  <td>
                    {row.lo === null ? "–" : `${formatShare(row.lo)} to ${formatShare(row.hi)}`}
                  </td>
                  <td>{row.sale ? `${row.discount ?? 0}% off` : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}
