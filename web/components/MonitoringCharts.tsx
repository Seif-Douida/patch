"use client";

// The Monitoring page's charts (Phase 7, design D10). One series and one axis each: the drift
// statistics and the judge's scores are small multiples, not lines sharing a scale. Thresholds and
// references are dashed ink lines; the acceptance drill's points are hollow and say "drill" in the
// tooltip and the table. Every chart has a table view.

import type { ReactNode } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  formatPercent,
  formatValue,
  runLabel,
  scoreName,
  statistic,
} from "@/lib/monitoring";
import { formatDate } from "@/lib/series";
import type { AgreementWeek, DriftSection, OnlineNight, PipelineRun } from "@/lib/types";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function shortDate(date: string): string {
  const [, month, day] = date.split("-").map(Number);
  return `${day} ${MONTHS[(month ?? 1) - 1]}`;
}

type DotProps = { cx?: number; cy?: number; payload?: { drill?: boolean } };

/** A point: filled for a weekly check, hollow for the acceptance drill's. */
export function PointDot({ cx, cy, payload }: DotProps) {
  if (cx === undefined || cy === undefined) return null;
  const drill = payload?.drill === true;
  return (
    <circle
      cx={cx}
      cy={cy}
      r={drill ? 5 : 4}
      className={drill ? "point point-drill" : "point point-week"}
    />
  );
}

/** The smallest 1, 2, 2.5 or 5 times a power of ten at or above `value`: a round axis top. */
export function niceTop(value: number): number {
  if (!(value > 0)) return 1;
  const power = 10 ** Math.floor(Math.log10(value));
  const step = [1, 2, 2.5, 5, 10].find((m) => m * power >= value - 1e-12) ?? 10;
  return step * power;
}

function formatTick(value: number): string {
  return Number(value.toPrecision(3)).toString();
}

type DotRender ={ cx?: number; cy?: number; key?: React.Key | null };

function weekDot({ cx, cy, key }: DotRender) {
  return <PointDot key={key ?? undefined} cx={cx} cy={cy} payload={{ drill: false }} />;
}

function drillDot({ cx, cy, key }: DotRender) {
  return <PointDot key={key ?? undefined} cx={cx} cy={cy} payload={{ drill: true }} />;
}

function Readout({ title, lines }: { title: string; lines: ReactNode[] }) {
  return (
    <div className="day-details">
      <p className="day-date">{title}</p>
      {lines.map((line, i) => (
        <p key={i}>{line}</p>
      ))}
    </div>
  );
}

function Frame({ children }: { children: ReactNode }) {
  return (
    <div className="chart-frame chart-frame-small">
      <ResponsiveContainer>{children as React.ReactElement}</ResponsiveContainer>
    </div>
  );
}

function TableView({ summary, caption, children }: { summary: string; caption: string; children: ReactNode }) {
  return (
    <details className="table-view">
      <summary>{summary}</summary>
      <div className="table-scroll">
        <table>
          <caption className="visually-hidden">{caption}</caption>
          {children}
        </table>
      </div>
    </details>
  );
}

const AXIS = { tickLine: false, axisLine: false } as const;

function xAxis() {
  return <XAxis dataKey="date" tickFormatter={shortDate} minTickGap={24} tickLine={false} tickMargin={8} />;
}

type DriftRow = { date: string; week: number | null; drill: number | null };

/** One drift statistic by week, against its threshold. */
export function DriftChart({ section }: { section: DriftSection }) {
  const { title, note, digits } = statistic(section.metric);
  const byDate = new Map<string, DriftRow>();
  for (const point of section.points) {
    const row = byDate.get(point.window_end) ?? { date: point.window_end, week: null, drill: null };
    if (point.drill) row.drill = point.value;
    else row.week = point.value;
    byDate.set(point.window_end, row);
  }
  const rows = [...byDate.values()].sort((a, b) => a.date.localeCompare(b.date));
  const threshold = section.threshold;
  const over = (value: number | null) => threshold !== null && value !== null && value > threshold;
  const values = section.points.map((p) => p.value ?? 0);
  const top = niceTop(Math.max(...values, threshold ?? 0) * 1.1);

  function DriftTooltip({ active, payload }: { active?: boolean; payload?: readonly { payload?: DriftRow }[] }) {
    const row = payload?.[0]?.payload;
    if (!active || !row) return null;
    const lines: ReactNode[] = [];
    if (row.week !== null) lines.push(<><strong>{formatValue(row.week, digits)}</strong> week ending {formatDate(row.date)}</>);
    if (row.drill !== null) lines.push(<>Drill: {formatValue(row.drill, digits)}</>);
    if (threshold !== null) lines.push(<span className="day-quiet">Threshold {formatValue(threshold, digits)}</span>);
    return <Readout title={formatDate(row.date)} lines={lines} />;
  }

  return (
    <figure className="chart chart-small">
      <figcaption>
        <h4>{title}</h4>
        <p className="chart-note">{note}</p>
        {threshold === null && <p className="chart-note">No threshold yet: it is set from past weeks.</p>}
      </figcaption>
      {rows.length > 0 && (
        <Frame>
          <ComposedChart data={rows} margin={{ top: 8, right: 24, left: 0 }}>
            <CartesianGrid vertical={false} />
            {xAxis()}
            <YAxis domain={[0, top]} ticks={[0, top / 2, top]} width={44} tickFormatter={(v: number) => formatTick(v)} {...AXIS} />
            {threshold !== null && (
              <ReferenceLine y={threshold} className="threshold-line" ifOverflow="extendDomain" />
            )}
            <Line dataKey="week" className="series-line" isAnimationActive={false} connectNulls
              dot={weekDot}
              activeDot={false} />
            <Line dataKey="drill" className="series-none" isAnimationActive={false}
              dot={drillDot}
              activeDot={false} />
            <Tooltip content={DriftTooltip} cursor={{ className: "crosshair" }} isAnimationActive={false} />
          </ComposedChart>
        </Frame>
      )}
      {section.points.length > 0 && (
        <TableView summary="Show the weeks" caption={title}>
          <thead>
            <tr>
              <th scope="col">Week ending</th>
              <th scope="col">Value</th>
              <th scope="col">Threshold</th>
            </tr>
          </thead>
          <tbody>
            {section.points.map((point) => (
              <tr key={`${point.window_end}-${point.drill}`}>
                <td>
                  {formatDate(point.window_end)}
                  {point.drill ? " (drill)" : ""}
                </td>
                <td>{formatValue(point.value, digits)}</td>
                <td>{over(point.value) ? "over" : ""}</td>
              </tr>
            ))}
          </tbody>
        </TableView>
      )}
    </figure>
  );
}

type AgreementRow = { date: string; value: number | null; band: [number, number] | null };

/** The weekly teacher-student agreement, its interval, the champion's reference and the alert line. */
export function AgreementChart({
  weeks,
  reference,
  threshold,
}: {
  weeks: AgreementWeek[];
  reference: number | null;
  threshold: number | null;
}) {
  const rows: AgreementRow[] = weeks.map((w) => ({
    date: w.window_end,
    value: w.value,
    band: w.lo !== null && w.hi !== null ? [w.lo, w.hi] : null,
  }));
  const floor = reference !== null && threshold !== null ? reference - threshold : null;

  function AgreementTooltip({ active, payload }: { active?: boolean; payload?: readonly { payload?: AgreementRow }[] }) {
    const row = payload?.[0]?.payload;
    if (!active || !row) return null;
    const lines: ReactNode[] = [<><strong>{formatValue(row.value)}</strong> macro-F1</>];
    if (row.band) lines.push(<span className="day-quiet">95% interval {formatValue(row.band[0])} to {formatValue(row.band[1])}</span>);
    return <Readout title={`Week ending ${formatDate(row.date)}`} lines={lines} />;
  }

  return (
    <figure className="chart chart-small">
      <figcaption>
        <h4>Agreement with the teacher</h4>
        <p className="chart-note">
          Macro-F1 of the champion against the teacher&apos;s labels on a week of new reviews, with
          its 95% interval.
          {reference !== null && ` The dashed line is the champion's own, ${formatValue(reference)}`}
          {floor !== null && `; an alert opens below ${formatValue(floor)}.`}
        </p>
      </figcaption>
      {rows.length > 0 && (
        <Frame>
          <ComposedChart data={rows} margin={{ top: 8, right: 24, left: 0 }}>
            <CartesianGrid vertical={false} />
            {xAxis()}
            <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} width={44} {...AXIS} />
            {reference !== null && <ReferenceLine y={reference} className="threshold-line" />}
            {floor !== null && <ReferenceLine y={floor} className="floor-line" />}
            <Area dataKey="band" className="series-band" isAnimationActive={false} activeDot={false} />
            <Line dataKey="value" className="series-line" isAnimationActive={false}
              dot={weekDot}
              activeDot={false} />
            <Tooltip content={AgreementTooltip} cursor={{ className: "crosshair" }} isAnimationActive={false} />
          </ComposedChart>
        </Frame>
      )}
      {rows.length > 0 && (
        <TableView summary="Show the weeks" caption="Agreement with the teacher">
          <thead>
            <tr>
              <th scope="col">Week ending</th>
              <th scope="col">Macro-F1</th>
              <th scope="col">95% interval</th>
              <th scope="col">Reviews</th>
            </tr>
          </thead>
          <tbody>
            {weeks.map((w) => (
              <tr key={w.window_end}>
                <td>{formatDate(w.window_end)}</td>
                <td>{formatValue(w.value)}</td>
                <td>{w.lo === null || w.hi === null ? "–" : `${formatValue(w.lo)} to ${formatValue(w.hi)}`}</td>
                <td>{w.n ?? "–"}</td>
              </tr>
            ))}
          </tbody>
        </TableView>
      )}
    </figure>
  );
}

type NightRow = { date: string; value: number | null };

function NightChart({
  title,
  note,
  rows,
  percent,
  table,
}: {
  title: string;
  note: string;
  rows: NightRow[];
  percent: boolean;
  table: ReactNode;
}) {
  const format = (v: number | null) => (percent ? formatPercent(v) : formatValue(v, 2));

  function NightTooltip({ active, payload }: { active?: boolean; payload?: readonly { payload?: NightRow }[] }) {
    const row = payload?.[0]?.payload;
    if (!active || !row) return null;
    return <Readout title={formatDate(row.date)} lines={[<strong key="v">{format(row.value)}</strong>]} />;
  }

  return (
    <figure className="chart chart-small">
      <figcaption>
        <h4>{title}</h4>
        <p className="chart-note">{note}</p>
      </figcaption>
      <Frame>
        <ComposedChart data={rows} margin={{ top: 8, right: 24, left: 0 }}>
          <CartesianGrid vertical={false} />
          {xAxis()}
          <YAxis domain={[0, 1]} ticks={[0, 0.5, 1]} width={44}
            tickFormatter={(v: number) => (percent ? formatPercent(v) : v.toFixed(1))} {...AXIS} />
          <Line dataKey="value" className="series-line" isAnimationActive={false} connectNulls
            dot={weekDot}
            activeDot={false} />
          <Tooltip content={NightTooltip} cursor={{ className: "crosshair" }} isAnimationActive={false} />
        </ComposedChart>
      </Frame>
      {table}
    </figure>
  );
}

/** The verifier's pass rate per night, and the night's numbers in a table. */
export function VerifierChart({ nights }: { nights: OnlineNight[] }) {
  return (
    <NightChart
      title="Claims that passed the verifier"
      note="Share of the drafted claims shipped as written, per night of reports."
      rows={nights.map((n) => ({ date: n.date, value: n.pass_rate }))}
      percent
      table={
      <TableView summary="Show the nights" caption="Reports per night">
        <thead>
          <tr>
            <th scope="col">Night</th>
            <th scope="col">Reports</th>
            <th scope="col">Passed</th>
            <th scope="col">Withheld</th>
            <th scope="col">Reworded</th>
            <th scope="col">Written without the model</th>
          </tr>
        </thead>
        <tbody>
          {nights.map((n) => (
            <tr key={n.date}>
              <td>{formatDate(n.date)}</td>
              <td>{n.reports ?? "–"}</td>
              <td>{formatPercent(n.pass_rate)}</td>
              <td>{Object.values(n.withheld).reduce((a, b) => a + b, 0)}</td>
              <td>{n.reworded}</td>
              <td>{n.gemini_errors ?? "–"}</td>
            </tr>
          ))}
        </tbody>
      </TableView>
      }
    />
  );
}

/** The judge's mean score per night, one small chart per score. */
export function JudgeCharts({ nights }: { nights: OnlineNight[] }) {
  const keys = [...new Set(nights.flatMap((n) => Object.keys(n.judge.scores)))].sort();
  return (
    <div className="small-multiples">
      {keys.map((key) => (
        <div key={key}>
          <NightChart
            title={scoreName(key)}
            note="Mean score of the night's reports, 0 to 1."
            rows={nights.map((n) => ({ date: n.date, value: n.judge.scores[key] ?? null }))}
            percent={false}
            table={
          <TableView summary="Show the nights" caption={`Judge: ${scoreName(key)}`}>
            <thead>
              <tr>
                <th scope="col">Night</th>
                <th scope="col">Mean score</th>
              </tr>
            </thead>
            <tbody>
              {nights.map((n) => (
                <tr key={n.date}>
                  <td>{formatDate(n.date)}</td>
                  <td>{formatValue(n.judge.scores[key] ?? null, 2)}</td>
                </tr>
              ))}
            </tbody>
          </TableView>
            }
          />
        </div>
      ))}
    </div>
  );
}

/** The last 30 nights, oldest first: each a mark, a failed one red with a cross and its word. */
export function RunStrip({ runs }: { runs: PipelineRun[] }) {
  return (
    <ul className="run-strip" aria-label="Last 30 nights">
      {runs.map((run) => {
        const failed = run.status !== "succeeded";
        const label = runLabel(run);
        return (
          <li key={run.run_id} className={failed ? "run run-failed" : "run run-ok"} aria-label={label} title={label}>
            <span className="run-mark" aria-hidden="true">
              {failed && (
                <svg viewBox="0 0 10 10" width="10" height="10">
                  <path d="M2 2 8 8M8 2 2 8" />
                </svg>
              )}
            </span>
            {failed && <span className="run-word">{run.status}</span>}
          </li>
        );
      })}
    </ul>
  );
}
