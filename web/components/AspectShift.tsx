import { WeakMark } from "@/components/WeakMark";
import { type PatchComparison, WEAK_NOTE, WINDOW_DAYS } from "@/lib/aspects";
import { formatCount, formatShare } from "@/lib/series";

// What negative reviews mention before and after a patch (design F7), as a dot chart: hollow
// before, solid after, the change in points beside each row, and the rows whose 95% interval
// includes 0 faded. Neutral ink: it describes a change, it doesn't judge it. Screen readers get the
// same numbers as text, and the exact table sits folded under it.

/** Percentage points with their sign: "+20", "−4", "0". */
function signed(value: number): string {
  const points = Math.round(value * 100);
  if (points > 0) return `+${points}`;
  if (points < 0) return `−${-points}`;
  return "0";
}

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

function Notes({ comparison }: { comparison: PatchComparison }) {
  const { before, after, shortBefore, shortAfter, enough } = comparison;
  const notes: string[] = [];
  if (shortBefore) {
    notes.push(`Based on only ${plural(before.days, "day")} of data before the patch.`);
  }
  if (shortAfter) {
    notes.push(`Only ${plural(after.days, "day")} of data since the patch so far.`);
  }
  if (!enough) {
    const counts = `${formatCount(before.n)} before, ${formatCount(after.n)} after`;
    notes.push(`Too few scored negative reviews for a comparison (${counts}).`);
  }
  return notes.length ? (
    <ul className="patch-aspect-notes">
      {notes.map((note) => (
        <li key={note}>{note}</li>
      ))}
    </ul>
  ) : null;
}

const ROW_HEIGHT = 24; // px, each aspect's strip

/** The axis's top: the largest share rounded up to the next 10%, at least 10%. */
function axisTop(shares: number[]): number {
  return Math.max(0.1, Math.ceil(Math.max(0, ...shares) * 10) / 10);
}

export function AspectShift({
  comparison,
  weak,
}: {
  comparison: PatchComparison;
  weak: Set<string>;
}) {
  if (!comparison.enough) return <Notes comparison={comparison} />;

  const { aspects } = comparison;
  const top = axisTop(aspects.flatMap((a) => [a.before ?? 0, a.after ?? 0]));
  const step = top > 0.5 ? 0.2 : 0.1;
  const ticks: number[] = [];
  for (let t = 0; t <= top + 1e-9; t += step) ticks.push(Math.round(t * 100) / 100);
  // Positions as percentages of the strip, so the dots stay round at any width and the labels,
  // which are page text, keep their size on a phone.
  const x = (share: number) => `${2 + (96 * share) / top}%`;
  const label = (code: string, text: string) =>
    weak.has(code) ? `${text} (detected less reliably)` : text;

  return (
    <div className="aspect-shift">
      <Notes comparison={comparison} />
      <p className="chart-note">
        {`What negative reviews mention, before (hollow) and after (solid): the ${WINDOW_DAYS} ` +
          `days before the patch, and its day and the ${WINDOW_DAYS - 1} after.`}
      </p>
      <figure className="shift-figure">
        <div className="shift-chart" aria-hidden="true">
          <div className="shift-row shift-axis">
            <span />
            <span className="shift-ticks">
              {ticks.map((tick) => (
                <span key={tick} className="shift-tick" style={{ left: x(tick) }}>
                  {formatShare(tick)}
                </span>
              ))}
            </span>
            <span />
          </div>
          {aspects.map((aspect) => {
            const [low, high] = aspect.interval ?? [0, 0];
            const faded = aspect.interval === null || (low <= 0 && high >= 0);
            const before = x(aspect.before ?? 0);
            const after = x(aspect.after ?? 0);
            return (
              <div
                key={aspect.code}
                className="shift-row"
                data-aspect={aspect.code}
                data-faded={faded ? "true" : "false"}
              >
                <span className="shift-label">
                  {aspect.label}
                  {weak.has(aspect.code) && "*"}
                </span>
                <svg className="shift-strip" height={ROW_HEIGHT}>
                  {ticks.map((tick) => (
                    <line key={tick} className="shift-grid" x1={x(tick)} x2={x(tick)} y1={0} y2={ROW_HEIGHT} />
                  ))}
                  <line className="shift-link" x1={before} x2={after} y1={ROW_HEIGHT / 2} y2={ROW_HEIGHT / 2} />
                  <circle className="shift-before" cx={before} cy={ROW_HEIGHT / 2} r={4.5} />
                  <circle className="shift-after" cx={after} cy={ROW_HEIGHT / 2} r={4.5} />
                </svg>
                <span className="shift-change">
                  {aspect.change === null ? "–" : `${signed(aspect.change)} pts`}
                </span>
              </div>
            );
          })}
        </div>
        <figcaption>Descriptive: the change around a patch, not proof the patch caused it.</figcaption>
      </figure>
      <ul className="visually-hidden">
        {aspects.map((aspect) => {
          const [low, high] = aspect.interval ?? [0, 0];
          return (
            <li key={aspect.code}>
              {`${label(aspect.code, aspect.label)}: ${formatShare(aspect.before)} before, ` +
                `${formatShare(aspect.after)} after, change ${signed(aspect.change ?? 0)} points ` +
                `(95% interval ${signed(low)} to ${signed(high)})`}
            </li>
          );
        })}
      </ul>
      <p className="chart-note">Faded rows: the change could be chance.</p>
      <p className="chart-note">
        With ten aspects, about one change in twenty looks clear by chance alone.
      </p>
      {aspects.some((aspect) => weak.has(aspect.code)) && <p className="chart-note">{WEAK_NOTE}</p>}
      <details className="table-view">
        <summary>Show these numbers as a table</summary>
        <div className="table-scroll">
          <table>
            <caption>Descriptive: the change around a patch, not proof the patch caused it.</caption>
            <thead>
              <tr>
                <th scope="col">Aspect</th>
                <th scope="col">Before</th>
                <th scope="col">After</th>
                <th scope="col">Change (pts)</th>
                <th scope="col">95% interval (pts)</th>
              </tr>
            </thead>
            <tbody>
              {aspects.map((aspect) => {
                const [low, high] = aspect.interval ?? [0, 0];
                return (
                  <tr key={aspect.code}>
                    <th scope="row">
                      {aspect.label}
                      {weak.has(aspect.code) && <WeakMark />}
                    </th>
                    <td>{formatShare(aspect.before)}</td>
                    <td>{formatShare(aspect.after)}</td>
                    <td>{aspect.change === null ? "–" : signed(aspect.change)}</td>
                    <td>{aspect.interval === null ? "–" : `${signed(low)} to ${signed(high)}`}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}
