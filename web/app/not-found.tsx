import Link from "next/link";

export default function NotFound() {
  return (
    <section className="intro">
      <h1>This page isn&apos;t here</h1>
      <p>
        PatchPulse only has pages for the games it tracks. <Link href="/">See all games</Link>.
      </p>
    </section>
  );
}
