// The favicon's pulse line as a small inline mark: beside the wordmark, and (beating) while an Ask
// answer is on its way. Decorative, so screen readers skip it. Not the game list's `Pulse`
// sparkline, which draws data.

const PATH = "M4 18h6l3-8 5 14 3-6h7";

export function PulseMark({ className, beating = false }: { className?: string; beating?: boolean }) {
  return (
    <svg
      className={["pulse-mark", className].filter(Boolean).join(" ")}
      viewBox="3 9 26 16"
      aria-hidden="true"
      focusable="false"
    >
      {beating ? (
        <>
          <path className="pulse-bed" d={PATH} />
          <path className="pulse-beat" d={PATH} />
        </>
      ) : (
        <path className="pulse-line" d={PATH} />
      )}
    </svg>
  );
}
