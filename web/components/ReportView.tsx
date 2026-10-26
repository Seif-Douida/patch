import {
  CONFIDENCE_LABELS,
  EXPLANATIONS,
  STATUS_LABELS,
  fileClaims,
  formatRange,
  joinWords,
  modelNames,
  orderedHypotheses,
  supportLine,
  withheldLines,
  withheldTotal,
} from "@/lib/reports";
import type { Hypothesis, PublicReport, ReportClaim } from "@/lib/types";

/** "The patch: inconclusive" */
function sectionTitle(hypothesis: Hypothesis): string {
  return `${EXPLANATIONS[hypothesis.code].long}: ${STATUS_LABELS[hypothesis.status].toLowerCase()}`;
}

function Status({ status }: { status: Hypothesis["status"] }) {
  return <span className={`status status-${status}`}>{STATUS_LABELS[status]}</span>;
}

function Claim({ claim }: { claim: ReportClaim }) {
  const support = supportLine(claim.supported_by);
  return (
    <li className="claim">
      <span className="claim-tick" aria-hidden="true">
        ✓
      </span>
      <span className="visually-hidden">Verified: </span>
      <span className="claim-text">{claim.text}</span>
      {support && <span className="claim-support"> {support}</span>}
      {claim.reworded && (
        <span className="claim-reworded">
          Reworded by the verifier: the evidence doesn&apos;t support saying the patch caused this.
        </span>
      )}
    </li>
  );
}

function ClaimSection({ id, title, claims }: { id: string; title: string; claims: ReportClaim[] }) {
  return (
    <section className="report-claims" aria-labelledby={id}>
      <h2 id={id}>{title}</h2>
      <ul>
        {claims.map((claim) => (
          <Claim key={claim.claim_id} claim={claim} />
        ))}
      </ul>
    </section>
  );
}

function Provenance({ report }: { report: PublicReport }) {
  const models = modelNames(report);
  const written = !report.basic && models.length > 0;
  const by = written ? `by ${joinWords(models)}` : "without a language model";
  return (
    <p className="report-provenance">
      Written {by}, from data version {report.data_version} with prompt version{" "}
      {report.prompt_version}.
    </p>
  );
}

/**
 * A report in layout B (Phase 5 design D11): the five explanations first, then the summary, then
 * each verified claim under the explanation it backs, and what the verifier withheld, folded, as
 * kinds and counts only. Used for patch reports and, from the API, for Ask's answers.
 */
export function ReportView({ report }: { report: PublicReport }) {
  const { filed, unfiled } = fileClaims(report);
  const withheld = withheldLines(report.withheld);
  const days = report.horizon ? `The ${report.horizon} days from` : "From";
  return (
    <article className="report">
      <p className="report-meta">
        {days} {formatRange(report.window.start, report.window.end)}.{" "}
        {CONFIDENCE_LABELS[report.confidence]}.
      </p>

      {report.basic && (
        <p className="report-basic" role="note">
          Written without the language model: none answered when this report was due, so it gives
          the core numbers and a template summary. It is written again the next night.
        </p>
      )}

      <ul className="verdicts" aria-label="Explanations checked">
        {orderedHypotheses(report).map((h) => (
          <li key={h.code} className={`verdict verdict-${h.status}`}>
            <span className="verdict-name">{EXPLANATIONS[h.code].short}</span>
            <Status status={h.status} />
          </li>
        ))}
      </ul>

      <p className="report-summary">{report.summary}</p>

      {filed.map(({ hypothesis, claims }) => (
        <ClaimSection
          key={hypothesis.code}
          id={`claims-${hypothesis.code}`}
          title={sectionTitle(hypothesis)}
          claims={claims}
        />
      ))}
      {unfiled.length > 0 && <ClaimSection id="claims-other" title="Also found" claims={unfiled} />}

      {withheld.length > 0 && (
        <details className="report-withheld">
          <summary>Withheld by the verifier ({withheldTotal(report.withheld)})</summary>
          <ul>
            {withheld.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
          <p>
            The verifier checks every claim against the data before it is shown; these didn&apos;t
            pass, and their wording isn&apos;t shown.
          </p>
        </details>
      )}

      <Provenance report={report} />
    </article>
  );
}
