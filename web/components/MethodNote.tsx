import { METHOD_NAMES } from "@/lib/effects";
import type { AspectModel, CausalInfo } from "@/lib/types";

export const REPO_URL = "https://github.com/Seif-Douida/patch";
export const EVAL_REPORT_URL = `${REPO_URL}/blob/main/reports/aspects/phase-2-eval.md`;

/** Which model found the aspects and how well it did on the gold test set; nothing without one. */
export function MethodNote({ model }: { model: AspectModel | null | undefined }) {
  if (!model) return null;
  const f1 = model.test_macro_f1;
  const score =
    f1 === null ? "" : `, macro-F1 ${f1.toFixed(2)} on our test set of English and French reviews`;
  return (
    <p className="method-note">
      Aspects are detected by a model ({model.family.toUpperCase()} {model.version}
      {score}). <a href={EVAL_REPORT_URL}>How the model was evaluated</a>
    </p>
  );
}

/** What the patch estimates are, what their range means and can't rule out (schema 3). */
export function EstimateNote({ causal }: { causal: CausalInfo | null | undefined }) {
  if (!causal) return null;
  const method = METHOD_NAMES[causal.default_method];
  return (
    <p className="method-note">
      An estimated effect is the gap between what happened to a game&apos;s share of positive
      reviews after a patch and what was expected without it, from the game&apos;s own history
      (interrupted time series) or from similar games without a patch of their own (synthetic
      control). The 90% range should hold the true effect about 9 times in 10. It can&apos;t rule
      out something else that changed at the same time, such as an event, news or a big launch.{" "}
      {causal.benchmark_report ? (
        <>
          Estimates use {method} where it&apos;s possible, chosen by testing both methods on
          effects added to quiet stretches of real data.{" "}
          <a href={`${REPO_URL}/blob/main/${causal.benchmark_report}`}>
            How the methods were tested
          </a>
        </>
      ) : (
        <>
          Estimates use {method} for now; the comparison of the two methods isn&apos;t published
          yet.
        </>
      )}
    </p>
  );
}
