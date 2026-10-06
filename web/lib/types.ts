// The site data (schema version 1), as written by src/patchpulse/export/site.py.

export type Header = {
  schema_version: number;
  data_version: number;
  generated_at: string; // UTC, "2026-09-29T03:32:00Z"
  attribution: string;
};

export type Latest = {
  days: number;
  to: string;
  n: number;
  share: number | null;
};

export type GameSummary = {
  appid: number;
  name: string;
  genres: string[];
  first_date: string | null;
  last_date: string | null;
  latest: Latest | null;
};

export type SiteIndex = Header & { games: GameSummary[] };

export type DayPoint = {
  date: string; // "2026-09-28"
  n: number; // reviews that day
  pos: number; // positive reviews
  share: number | null; // pos / n; null on a day without reviews
  lo: number | null; // 90% Wilson interval
  hi: number | null;
  sale: boolean;
  discount: number | null;
};

export type PatchType = "major_update" | "patch" | "hotfix" | "event" | "marketing" | "other";

export type Patch = {
  date: string;
  title: string;
  url: string;
  type: PatchType;
  treatment: boolean; // a major update or a patch: the posts the analysis tests
};

export type GameData = Header & {
  appid: number;
  name: string;
  genres: string[];
  series: DayPoint[];
  patches: Patch[];
};
