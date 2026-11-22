"use client";

import { useEffect, useState } from "react";

import { PulseMark } from "@/components/PulseMark";
import { SLOW_NOTE, SLOW_NOTE_MS, phraseAt } from "@/lib/ask";

/**
 * What the Ask page shows between the click and the answer (design F8): the pulse line beating,
 * a phrase that changes every few seconds, and the seconds so far; after 8 s, a line on how long a
 * question takes. Hidden from screen readers: the form's status region says "Working on your
 * answer." once instead of a phrase every few seconds.
 */
export function AskWait() {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const started = Date.now();
    // 100 ms: a multiple of both 2.8 s (each phrase) and 8 s (the note), so changes land on time.
    const timer = setInterval(() => setElapsed(Date.now() - started), 100);
    return () => clearInterval(timer);
  }, []);

  const phrase = phraseAt(elapsed);
  return (
    <div className="ask-wait" aria-hidden="true">
      <p className="ask-wait-line">
        <PulseMark className="ask-pulse" beating />
        <span key={phrase} className="ask-phrase">
          {phrase}
        </span>
        <span className="ask-seconds">{Math.floor(elapsed / 1000)} s</span>
      </p>
      {elapsed >= SLOW_NOTE_MS && <p className="ask-slow">{SLOW_NOTE}</p>}
    </div>
  );
}
