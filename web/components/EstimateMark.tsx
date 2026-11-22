import type { EstimatedHorizon } from "@/lib/effects";
import { SCALE_LIMIT, clampToScale, verdictOf } from "@/lib/patches";

// A patch row's estimate (design F7): a dot and its 90% range on one scale shared by every row,
// from −100 to +100 points, with a zero line. Filled only when the whole range is on one side of
// zero. Decorative for screen readers: the row's name says the same in words.

const WIDTH = 260;
const HEIGHT = 22;
const PAD = 7; // room for the dot and a chevron at either end

function scale(width: number) {
  return (value: number) => PAD + ((width - 2 * PAD) * (value + SCALE_LIMIT)) / (2 * SCALE_LIMIT);
}

export function EstimateMark({
  horizon,
  width = WIDTH,
  height = HEIGHT,
}: {
  horizon: EstimatedHorizon | null;
  width?: number;
  height?: number;
}) {
  const x = scale(width);
  const mid = height / 2;
  const verdict = verdictOf(horizon);
  const zero = <line className="mark-zero" x1={x(0)} x2={x(0)} y1={2} y2={height - 2} />;

  if (horizon === null) {
    return (
      <svg
        className="estimate-mark"
        viewBox={`0 0 ${width} ${height}`}
        aria-hidden="true"
        data-verdict={verdict}
      >
        {zero}
        <text className="mark-none" x={x(0) + 8} y={mid + 4}>
          No estimate
        </text>
      </svg>
    );
  }

  const low = clampToScale(horizon.lo90);
  const high = clampToScale(horizon.hi90);
  const dot = clampToScale(horizon.effect);
  const clipped =
    low.clipped && high.clipped ? "both" : low.clipped ? "low" : high.clipped ? "high" : undefined;
  const chevron = (at: number, towards: -1 | 1) =>
    `${at - towards * 5},${mid - 4} ${at},${mid} ${at - towards * 5},${mid + 4}`;

  return (
    <svg
      className="estimate-mark"
      viewBox={`0 0 ${width} ${height}`}
      aria-hidden="true"
      data-verdict={verdict}
      data-clipped={clipped}
    >
      {zero}
      <line className={`mark-range mark-${verdict}`} x1={x(low.value)} x2={x(high.value)} y1={mid} y2={mid} />
      {low.clipped && <polyline className={`mark-open mark-${verdict}`} points={chevron(x(low.value), -1)} />}
      {high.clipped && <polyline className={`mark-open mark-${verdict}`} points={chevron(x(high.value), 1)} />}
      <circle className={`mark-dot mark-${verdict}`} cx={x(dot.value)} cy={mid} r={5} />
    </svg>
  );
}

/** The scale's labels for the column header: −50, 0 and +50, over the same positions. */
export function ScaleAxis({ width = WIDTH }: { width?: number }) {
  const x = scale(width);
  return (
    <svg className="scale-axis" viewBox={`0 0 ${width} 14`} aria-hidden="true">
      {(
        [
          [-50, "−50"],
          [0, "0"],
          [50, "+50"],
        ] as const
      ).map(([value, label]) => (
        <text key={value} x={x(value)} y={11} textAnchor="middle">
          {label}
        </text>
      ))}
    </svg>
  );
}
