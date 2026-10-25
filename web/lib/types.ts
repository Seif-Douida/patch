// The site data (schema versions 3 and 4), as written by src/patchpulse/export/site.py.
// Schema 2 added the aspect fields (null on a night without a model); schema 3 the patch estimates
// (`causal` in the index, `effect` on each patch); schema 4 adds the reports (`reports` in the
// index, `report` on each patch, and reports/{bundle_id}.json), absent in schema 3, so they are
// optional.

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

// The model that detects aspects, with its macro-F1 on the gold test set and, when recorded,
// each aspect's F1 there.
export type AspectModel = {
  family: string;
  version: string;
  test_macro_f1: number | null;
  aspect_f1?: Record<string, number>;
};

export type CausalMethod = "its" | "synth"; // interrupted time series, synthetic control

// How the patch estimates were made: the default method (chosen by the benchmark) and its report.
export type CausalInfo = {
  default_method: CausalMethod;
  effects_version: string;
  benchmark_report: string | null; // its path in the repository; null until the benchmark ran
  validated: boolean; // a method met the benchmark's rule (ADR-007); false until it ran
};

export type SiteIndex = Header & {
  games: GameSummary[];
  aspect_model?: AspectModel | null; // schema 2
  causal?: CausalInfo | null; // schema 3
  reports?: number; // schema 4: how many report files there are
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

// Why a horizon has no estimate when no stored estimate says (otherwise `reason` is the
// estimator's own words, such as "too few comparison games without their own patch (3 of 5
// needed)").
export type EffectReasonCode =
  | "window_not_complete"
  | "another_patch_within_14_days" // 14 days only
  | "not_estimated_yet"; // the nightly effects stage hasn't reached it

// One horizon of a patch's estimated effect on the positive share.
export type HorizonEffect = {
  days: number; // 7 or 14
  ends: string; // the window's last day
  effect: number | null; // percentage points
  lo90: number | null; // the 90% range
  hi90: number | null;
  method: CausalMethod | null;
  flags: string[]; // "confounded", "discounts_unknown", "volume_spike", "binned"
  reason: string | null; // an EffectReasonCode, or the estimator's words; null with an effect
};

// A treatment patch's estimate is its bundle's: patches within 7 days of the first share one.
export type PatchEffect = {
  start: string; // the bundle's first day
  horizons: HorizonEffect[]; // 7 days, then 14
};

export type Patch = {
  date: string;
  title: string;
  url: string;
  type: PatchType;
  treatment: boolean; // a major update or a patch: the posts the analysis tests
  effect?: PatchEffect | null; // schema 3: null unless a treatment candidate
  report?: string | null; // schema 4: its bundle's report (the bundle id), null without one
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

// Schema 4: a patch bundle's report, as its public copy (src/patchpulse/agent/report.py
// `public_copy`): the verified claims with how many reviews support each, never the reviews
// themselves; withheld claims only as counts per verifier rule.
export type HypothesisCode = "H1" | "H2" | "H3" | "H4" | "H5";

export type Hypothesis = {
  code: HypothesisCode;
  status: "supported" | "rejected" | "inconclusive"; // set by fixed rules, not the model
  rule: string; // which branch of the rule set decided, "H1.placebo"
  evidence: Record<string, number | boolean | null>;
};

export type Finding = {
  title: string;
  hypothesis: HypothesisCode;
  claim_ids: string[];
};

export type ClaimType = "descriptive" | "comparative" | "causal";

export type ReportClaim = {
  claim_id: string;
  text: string;
  claim_type: ClaimType;
  value: number | null;
  unit: string | null;
  reworded: boolean; // a causal claim the verifier rewrote with a hedge
  supported_by: number; // reviews behind it
};

export type PublicReport = {
  report_id: string;
  appid: number;
  scope: "bundle" | "question";
  bundle_id: string | null;
  horizon: number | null; // 7 or 14 days
  question: string | null;
  window: { start: string; end: string };
  data_version: number;
  model_ids: Record<string, string>; // step -> model; empty for a basic report
  prompt_version: string;
  summary: string;
  confidence: "low" | "medium" | "high";
  basic: boolean; // written without a model: fixed claims and a template summary
  hypotheses: Hypothesis[];
  findings: Finding[];
  claims: ReportClaim[]; // verified only
  withheld: Record<string, number>; // verifier rule -> claims it withheld
  rule_pass_rates: Record<string, number>;
};

export type ReportFile = Header & {
  report: PublicReport;
};
