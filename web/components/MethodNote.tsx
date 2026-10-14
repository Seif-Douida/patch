import type { AspectModel } from "@/lib/types";

export const EVAL_REPORT_URL =
  "https://github.com/Seif-Douida/patch/blob/main/reports/aspects/phase-2-eval.md";

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
