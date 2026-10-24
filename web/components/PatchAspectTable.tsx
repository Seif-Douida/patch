import {
  type PatchComparison,
  WEAK_NOTE,
  WINDOW_DAYS,
  beforeAfter,
  weakAspects,
} from "@/lib/aspects";
import { describeEffect } from "@/lib/effects";
import { formatCount, formatDate, formatShare } from "@/lib/series";
import type { AspectModel, AspectSeries, CausalInfo, Patch } from "@/lib/types";

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

type EstimateContext = { causal: CausalInfo | null; discountsFrom: string | null };

/** Schema 3: the patch's estimated effect on positive reviews, or why there's none. */
function Estimate({ patch, causal, discountsFrom }: { patch: Patch } & EstimateContext) {
  if (!patch.effect) return null; // schema 2, or not a treatment candidate
  const { line, notes } = describeEffect(patch.effect, {
    patchDate: patch.date,
    defaultMethod: causal?.default_method ?? null,
    discountsFrom,
  });
  return (
    <div className="patch-effect">
      <p className="patch-effect-line">{line}</p>
      {notes.length > 0 && (
        <ul className="patch-effect-notes">
          {notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

type TableProps = {
  patch: Patch;
  aspects: AspectSeries;
  open: boolean;
  weak: Set<string>;
} & EstimateContext;

function PatchTable({ patch, aspects, open, weak, causal, discountsFrom }: TableProps) {
  const comparison = beforeAfter(aspects, patch.date);
  return (
    <details className="patch-aspect" open={open}>
      <summary>
        <span className="patch-aspect-title">{patch.title}</span>
        <span className="patch-aspect-date">{formatDate(patch.date)}</span>
      </summary>
      <Estimate patch={patch} causal={causal} discountsFrom={discountsFrom} />
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

/**
 * One table per major update or patch, newest first (the first one open), each under its
 * estimated effect when the data has one (schema 3). `discountsFrom` is the first day the game's
 * price is known.
 */
export function PatchAspectTables({
  aspects,
  patches,
  model = null,
  causal = null,
  discountsFrom = null,
}: {
  aspects: AspectSeries;
  patches: Patch[];
  model?: AspectModel | null;
  causal?: CausalInfo | null;
  discountsFrom?: string | null;
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
          causal={causal}
          discountsFrom={discountsFrom}
        />
      ))}
      {aspects.codes.some((code) => weak.has(code)) && <p className="chart-note">{WEAK_NOTE}</p>}
    </section>
  );
}
