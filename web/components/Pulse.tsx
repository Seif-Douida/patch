import { formatShare, type SharePoint } from "@/lib/series";

type Props = { points: SharePoint[]; width?: number; height?: number };

// A game's recent pulse: its 7-day positive share as a small line, drawn as plain SVG at build
// time (no chart library on the home page). The scale fits the line, so it shows shape, not
// level; the level is the number beside it.
export function Pulse({ points, width = 200, height = 44 }: Props) {
  const values = points.map((p) => p.share).filter((s): s is number => s !== null);
  if (values.length < 2) {
    return <span className="pulse-empty">Not enough reviews yet</span>;
  }
  const low = Math.max(0, Math.min(...values) - 0.03);
  const high = Math.min(1, Math.max(...values) + 0.03);
  const pad = 5; // room for the end dot and its ring
  const x = (i: number) => pad + (i / (points.length - 1)) * (width - 2 * pad);
  const y = (s: number) => pad + (1 - (s - low) / (high - low || 1)) * (height - 2 * pad);

  const segments: string[] = [];
  let current = "";
  points.forEach((point, i) => {
    if (point.share === null) {
      if (current) segments.push(current);
      current = "";
      return;
    }
    current += `${current ? "L" : "M"}${x(i).toFixed(1)},${y(point.share).toFixed(1)}`;
  });
  if (current) segments.push(current);

  const lastIndex = points.findLastIndex((p) => p.share !== null);
  const last = points[lastIndex];
  const first = points.find((p) => p.share !== null);
  const label = `Positive share over the last ${points.length} days, from ${formatShare(
    first?.share ?? null,
  )} to ${formatShare(last?.share ?? null)}`;

  return (
    <svg
      className="pulse"
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      height={height}
      role="img"
      aria-label={label}
    >
      {segments.map((d) => (
        <path key={d} d={d} className="pulse-line" />
      ))}
      {last?.share != null && (
        <circle cx={x(lastIndex)} cy={y(last.share)} r={4} className="pulse-dot" />
      )}
    </svg>
  );
}
