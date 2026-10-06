import { formatDate } from "@/lib/series";

type Props = {
  attribution: string;
  generatedAt: string; // UTC, "2026-09-29T03:32:00Z"
  dataVersion: number;
};

export function Footer({ attribution, generatedAt, dataVersion }: Props) {
  const asOf = `${formatDate(generatedAt.slice(0, 10))}, ${generatedAt.slice(11, 16)} UTC`;
  return (
    <footer className="site-footer">
      <p>{attribution}</p>
      <p>{`Data as of ${asOf} (version ${dataVersion})`}</p>
      <p>
        <a href="https://github.com/Seif-Douida/patch">Source and method on GitHub</a>
      </p>
    </footer>
  );
}
