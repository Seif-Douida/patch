// The site data (schema versions 1 and 2), as written by src/patchpulse/export/site.py.
// Schema 2 adds the aspect fields; they are absent in schema 1 and null on a night without a
// model, so every one of them is optional.

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

// Over the last 30 days: the aspect mentioned by the largest share of scored negative reviews.
export type TopComplaint = {
  code: string;
  share: number;
  n: number; // scored negative reviews
};

export type GameSummary = {
  appid: number;
  name: string;
  genres: string[];
  first_date: string | null;
  last_date: string | null;
  latest: Latest | null;
  top_complaint_30d?: TopComplaint | null; // schema 2
};

// The model that detects aspects, with its macro-F1 on the gold test set.
export type AspectModel = {
  family: string;
  version: string;
  test_macro_f1: number | null;
};

export type SiteIndex = Header & {
  games: GameSummary[];
  aspect_model?: AspectModel | null; // schema 2
};

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

// One day with scored reviews; days without any are absent.
export type AspectDay = {
  date: string;
  scored: [number, number]; // [negative, positive] reviews scored
  neg: number[]; // per aspect, in `codes` order: scored negative reviews that mention it
  pos: number[];
};

export type AspectSeries = {
  codes: string[];
  days: AspectDay[];
};

export type GameData = Header & {
  appid: number;
  name: string;
  genres: string[];
  series: DayPoint[];
  patches: Patch[];
  aspects?: AspectSeries | null; // schema 2
};
