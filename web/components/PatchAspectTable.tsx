import {
  type PatchComparison,
  WEAK_NOTE,
  WINDOW_DAYS,
  beforeAfter,
  weakAspects,
} from "@/lib/aspects";
import { formatCount, formatDate, formatShare } from "@/lib/series";
import type { AspectModel, AspectSeries, Patch } from "@/lib/types";

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

/** The less reliable aspects' marker: a visible asterisk, and words for screen readers. */
export function WeakMark() {
  return (
    <>
      <span aria-hidden="true">*</span>
      <span className="visually-hidden"> (detected less reliably)</span>
    </>
  );
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

type TableProps = { patch: Patch; aspects: AspectSeries; open: boolean; weak: Set<string> };

function PatchTable({ patch, aspects, open, weak }: TableProps) {
  const comparison = beforeAfter(aspects, patch.date);
  return (
    <details className="patch-aspect" open={open}>
      <summary>
        <span className="patch-aspect-title">{patch.title}</span>
        <span className="patch-aspect-date">{formatDate(patch.date)}</span>
      </summary>
      <Notes comparison={comparison} />
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
            {comparison.aspects.map((aspect) => {
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
                  <td>
                    {aspect.interval === null ? "–" : `${signed(low)} to ${signed(high)}`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </details>
  );
}

/** One table per major update or patch, newest first (the first one open). */
export function PatchAspectTables({
  aspects,
  patches,
  model = null,
}: {
  aspects: AspectSeries;
  patches: Patch[];
  model?: AspectModel | null;
}) {
  const weak = weakAspects(model);
  const treatments = patches
    .filter((patch) => patch.treatment)
    .sort((a, b) => b.date.localeCompare(a.date));
  if (treatments.length === 0) return null;
  return (
    <section className="patch-aspects">
      <h2>Before and after each patch</h2>
      <p className="chart-note">
        For each major update or patch, the share of scored negative reviews that mention each
        aspect before it (the {WINDOW_DAYS} days before) and after it (its day and the{" "}
        {WINDOW_DAYS - 1} after), and the change in percentage points with its 95% interval. With
        ten aspects a patch, about one change in twenty will look clear by chance alone.
      </p>
      {treatments.map((patch, index) => (
        <PatchTable
          key={`${patch.date}-${patch.url}`}
          patch={patch}
          aspects={aspects}
          open={index === 0}
          weak={weak}
        />
      ))}
      {aspects.codes.some((code) => weak.has(code)) && <p className="chart-note">{WEAK_NOTE}</p>}
    </section>
  );
}
