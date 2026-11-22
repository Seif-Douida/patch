"use client";

import Link from "next/link";
import { useEffect, useId, useMemo, useRef, useState } from "react";

import { AspectShift } from "@/components/AspectShift";
import { EstimateMark, ScaleAxis } from "@/components/EstimateMark";
import { beforeAfter, weakAspects } from "@/lib/aspects";
import { describeEffect } from "@/lib/effects";
import {
  FIRST_ROWS,
  MORE_ROWS,
  type PatchWindow,
  VERDICT_LABELS,
  patchWindows,
  windowName,
} from "@/lib/patches";
import { reportHref } from "@/lib/reports";
import { PATCH_LABELS, formatDate } from "@/lib/series";
import type { AspectModel, AspectSeries, CausalInfo, Patch } from "@/lib/types";

// The game page's one patch section (design F7): every tested patch as a row with its estimate on
// a shared scale, newest first, eight at a time; open a row for its sentence, posts, report and
// what negative reviews mention before and after. A second view lists every post.

type View = "tested" | "all";

type Props = {
  appid: number;
  patches: Patch[];
  aspects: AspectSeries | null;
  model: AspectModel | null;
  causal: CausalInfo | null;
  discountsFrom: string | null;
};

function KeyDot({ kind }: { kind: "up" | "down" | "unclear" }) {
  return (
    <svg className="key-dot" viewBox="0 0 14 14" aria-hidden="true">
      <circle className={`mark-dot mark-${kind}`} cx={7} cy={7} r={5} />
    </svg>
  );
}

function Caret() {
  return (
    <svg className="patch-caret" viewBox="0 0 10 10" aria-hidden="true">
      <polyline points="3,2 7,5 3,8" />
    </svg>
  );
}

function Panel({
  id,
  entry,
  appid,
  aspects,
  weak,
  causal,
  discountsFrom,
}: {
  id: string;
  entry: PatchWindow;
  appid: number;
  aspects: AspectSeries | null;
  weak: Set<string>;
  causal: CausalInfo | null;
  discountsFrom: string | null;
}) {
  const text = entry.effect
    ? describeEffect(entry.effect, {
        patchDate: entry.start,
        defaultMethod: causal?.default_method ?? null,
        discountsFrom,
      })
    : { line: "No estimate.", notes: [] };
  return (
    <div id={id} className="patch-panel">
      {/* The verdict in words, so colour is never its only carrier (design F7). */}
      {entry.horizon && <p className="patch-verdict">{VERDICT_LABELS[entry.verdict]}</p>}
      <div className="patch-effect">
        <p className="patch-effect-line">{text.line}</p>
        {text.notes.length > 0 && (
          <ul className="patch-effect-notes">
            {text.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        )}
      </div>
      <ul className="patch-posts">
        {entry.patches.map((patch) => (
          <li key={`${patch.date}-${patch.url}`}>
            <span className="patch-post-date">{formatDate(patch.date)}</span>{" "}
            <a href={patch.url} rel="noopener noreferrer">
              {patch.title}
            </a>
          </li>
        ))}
      </ul>
      {entry.report && (
        <p className="patch-report">
          <Link href={reportHref(appid, entry.report)} prefetch={false}>
            Read the report
          </Link>
        </p>
      )}
      {aspects && <AspectShift comparison={beforeAfter(aspects, entry.start)} weak={weak} />}
    </div>
  );
}

function More({ left, noun, onMore }: { left: number; noun: string; onMore: () => void }) {
  if (left <= 0) return null;
  return (
    <p className="patch-more">
      <button type="button" onClick={onMore}>
        Show {Math.min(MORE_ROWS, left)} more
      </button>
      <span>
        {left} older {noun}
      </span>
    </p>
  );
}

export function PatchList({ appid, patches, aspects, model, causal, discountsFrom }: Props) {
  const windows = useMemo(() => patchWindows(patches), [patches]);
  const posts = useMemo(
    () => [...patches].sort((a, b) => b.date.localeCompare(a.date)),
    [patches],
  );
  const weak = useMemo(() => weakAspects(model), [model]);
  const [view, setView] = useState<View>("tested");
  const [shown, setShown] = useState(FIRST_ROWS);
  const [open, setOpen] = useState<Set<string>>(
    () => new Set(windows[0] ? [windows[0].start] : []),
  );
  const prefix = useId();

  function choose(next: View) {
    setView(next);
    setShown(FIRST_ROWS);
  }

  function toggle(start: string) {
    setOpen((current) => {
      const next = new Set(current);
      if (next.has(start)) next.delete(start);
      else next.add(start);
      return next;
    });
  }

  // "Show N more" moves focus to the first row it revealed: the button may be gone (nothing left),
  // and a keyboard or screen-reader user keeps their place instead of landing on the page.
  const section = useRef<HTMLElement>(null);
  const focusFrom = useRef<number | null>(null);
  useEffect(() => {
    const index = focusFrom.current;
    if (index === null) return;
    focusFrom.current = null;
    section.current?.querySelectorAll<HTMLElement>("[data-row-focus]")[index]?.focus();
  }, [shown]);

  const more = () => {
    focusFrom.current = shown;
    setShown((count) => count + MORE_ROWS);
  };

  return (
    <section ref={section} className="patch-list" aria-labelledby={`${prefix}-heading`}>
      <h2 id={`${prefix}-heading`}>Patches</h2>
      <p className="chart-note">
        Newest first. Patches a few days apart share one estimate and one report.
      </p>
      <div className="range-filter patch-views" role="group" aria-label="Patches shown">
        <button type="button" aria-pressed={view === "tested"} onClick={() => choose("tested")}>
          Tested patches ({windows.length})
        </button>
        <button type="button" aria-pressed={view === "all"} onClick={() => choose("all")}>
          All posts ({posts.length})
        </button>
      </div>

      {view === "tested" &&
        (windows.length === 0 ? (
          <p>No tested patches in this period.</p>
        ) : (
          <>
            <ul className="chart-key patch-key" aria-label="Estimate key">
              {(["up", "down", "unclear"] as const).map((kind) => (
                <li key={kind}>
                  <KeyDot kind={kind} />
                  {VERDICT_LABELS[kind]}
                </li>
              ))}
              <li className="patch-key-note">
                Line: 90% range. Scale: change in positive share, in points, over the 14 days
                after a patch (7 when there&apos;s no 14-day estimate), from −100 to +100.
              </li>
            </ul>
            <div className="patch-row patch-head" aria-hidden="true">
              <span />
              <span className="patch-date">Date</span>
              <span className="patch-title">Patch</span>
              <ScaleAxis />
            </div>
            <ul className="patch-rows">
              {windows.slice(0, shown).map((entry) => {
                const isOpen = open.has(entry.start);
                const panel = `${prefix}-${entry.start}`;
                return (
                  <li key={entry.start}>
                    <button
                      type="button"
                      className="patch-row"
                      data-row-focus=""
                      aria-expanded={isOpen}
                      aria-controls={isOpen ? panel : undefined}
                      aria-label={windowName(entry)}
                      onClick={() => toggle(entry.start)}
                    >
                      <Caret />
                      <span className="patch-date">{formatDate(entry.start)}</span>
                      <span className="patch-title">{entry.title}</span>
                      <EstimateMark horizon={entry.horizon} />
                    </button>
                    {isOpen && (
                      <Panel
                        id={panel}
                        entry={entry}
                        appid={appid}
                        aspects={aspects}
                        weak={weak}
                        causal={causal}
                        discountsFrom={discountsFrom}
                      />
                    )}
                  </li>
                );
              })}
            </ul>
            <More left={windows.length - shown} noun="tested patches" onMore={more} />
          </>
        ))}

      {view === "all" &&
        (posts.length === 0 ? (
          <p>No official posts in this period.</p>
        ) : (
          <>
            <div className="table-scroll post-table">
              <table>
                <thead>
                  <tr>
                    <th scope="col">Date</th>
                    <th scope="col">Post</th>
                    <th scope="col">Kind</th>
                  </tr>
                </thead>
                <tbody>
                  {posts.slice(0, shown).map((post) => (
                    <tr key={`${post.date}-${post.url}`}>
                      <td>{formatDate(post.date)}</td>
                      <td>
                        <a href={post.url} rel="noopener noreferrer" data-row-focus="">
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
            </div>
            <More left={posts.length - shown} noun="posts" onMore={more} />
            <p className="method-note">
              Each post is labelled by its title. Major updates and patches are the ones
              PatchPulse tests for an effect on reviews; the rest are shown for context.
            </p>
          </>
        ))}
    </section>
  );
}
