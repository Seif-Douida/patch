import type { Metadata } from "next";

import {
  AgreementChart,
  DriftChart,
  JudgeCharts,
  RunStrip,
  VerifierChart,
} from "@/components/MonitoringCharts";
import { loadAlerts, loadMonitoring } from "@/lib/data";
import {
  dayOf,
  firstCheckMessage,

  hasDriftPoints,
  ISSUES_URL,
  retrainSummary,
  trainedOn,
} from "@/lib/monitoring";
import { formatDate } from "@/lib/series";

export const metadata: Metadata = {
  title: "Monitoring",
  description: "How PatchPulse's nightly pipeline, aspect model and report agent are doing.",
};

/**
 * Monitoring (Phase 7, design D10): the nightly pipeline, the aspect model's drift and agreement
 * with its teacher, the report agent's verifier and judge, and the open alerts. Static: the data
 * is the night's monitoring.json and the alerts publish-site.yml lists at build time.
 */
export default function MonitoringPage() {
  const monitoring = loadMonitoring();
  const alerts = loadAlerts();
  const lastSuccess = dayOf(monitoring?.freshness.last_success ?? null);

  return (
    <>
      <header className="monitoring-header">
        <h1>Monitoring</h1>
        <p>
          How the nightly pipeline, the model that finds aspects in reviews and the agent that writes
          the patch reports are doing. Every number here is a count, a rate or a score: no review
          text.
        </p>
      </header>

      <section className="monitoring-section" aria-labelledby="alerts">
        <h2 id="alerts">Open alerts</h2>
        {alerts.length === 0 ? (
          <p>No alerts open</p>
        ) : (
          <ul className="alert-list" aria-label="Open alerts">
            {alerts.map((alert) => (
              <li key={alert.number}>
                <a href={`${ISSUES_URL}/${alert.number}`}>#{alert.number}</a> {alert.title}{" "}
                <span className="alert-meta">
                  {alert.label === "drift" ? "drift" : "alert"}, opened {formatDate(alert.opened_at.slice(0, 10))}
                </span>
              </li>
            ))}
          </ul>
        )}
        {monitoring && monitoring.rules.length > 0 && (
          <details className="table-view">
            <summary>Show the alert rules</summary>
            <div className="table-scroll">
              <table>
                <caption className="visually-hidden">Alert rules</caption>
                <thead>
                  <tr>
                    <th scope="col">Alert</th>
                    <th scope="col">Threshold</th>
                    <th scope="col">What it measures</th>
                  </tr>
                </thead>
                <tbody>
                  {monitoring.rules.map((rule) => (
                    <tr key={rule.id}>
                      <td>{rule.title}</td>
                      <td>{rule.threshold ?? "not set yet"}</td>
                      <td>{rule.meaning}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        )}
      </section>

      {monitoring === null ? (
        <p className="monitoring-empty">
          The monitoring data appears after the next night with this page&apos;s pipeline.
        </p>
      ) : (
        <>
          <section className="monitoring-section" aria-labelledby="pipeline">
            <h2 id="pipeline">Pipeline</h2>
            <p>
              {lastSuccess === null
                ? "No night has finished successfully yet."
                : `The last successful night finished on ${formatDate(lastSuccess)}.`}{" "}
              The pipeline runs every other night.
            </p>
            {monitoring.pipeline.length > 0 && <RunStrip runs={monitoring.pipeline} />}
            {monitoring.data_quality.length > 0 && (
              <table className="quality-table">
                <caption>Data-quality results over these nights</caption>
                <thead>
                  <tr>
                    <th scope="col">Check</th>
                    <th scope="col">Severity</th>
                    <th scope="col">Results</th>
                  </tr>
                </thead>
                <tbody>
                  {monitoring.data_quality.map((row) => (
                    <tr key={`${row.check}-${row.severity}`}>
                      <td>{row.check.replaceAll("_", " ")}</td>
                      <td>{row.severity}</td>
                      <td>{row.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>

          <section className="monitoring-section" aria-labelledby="model">
            <h2 id="model">Model</h2>
            {monitoring.champion && (
              <p>
                Aspects are found by aspect-classifier v{monitoring.champion.version} (
                {monitoring.champion.family.toUpperCase()}
                {trainedOn(monitoring.champion.model_version)
                  ? `, trained ${trainedOn(monitoring.champion.model_version)}`
                  : ""}
                ).
              </p>
            )}
            <h3>Drift, week by week</h3>
            <p className="section-note">
              Each week&apos;s reviews against the reviews the model was trained on. A value over its
              threshold opens a drift issue; hollow points are the acceptance drill&apos;s.
            </p>
            {!hasDriftPoints(monitoring.drift) && <p>{firstCheckMessage(monitoring.generated_at)}</p>}
            <div className="small-multiples">
              {monitoring.drift.map((section) => (
                <DriftChart key={section.rule} section={section} />
              ))}
            </div>
            <AgreementChart
              weeks={monitoring.agreement.weeks}
              reference={monitoring.agreement.reference}
              threshold={monitoring.agreement.threshold}
            />
            <h3>Retraining</h3>
            <p>{retrainSummary(monitoring.retrain)}</p>
          </section>

          <section className="monitoring-section" aria-labelledby="agent">
            <h2 id="agent">Agent</h2>
            {monitoring.online.nights.length === 0 ? (
              <p>No report has been checked yet.</p>
            ) : (
              <>
                <VerifierChart nights={monitoring.online.nights} />
                <h3>Judge (uncalibrated)</h3>
                <p className="section-note">
                  A model scores each report against a rubric. It hasn&apos;t been checked against
                  people&apos;s judgments yet, so read these as trends, not grades.
                </p>
                <JudgeCharts nights={monitoring.online.nights} />
              </>
            )}
          </section>
          <p className="section-note">
            Generated {formatDate(monitoring.generated_at.slice(0, 10))}.
          </p>
        </>
      )}
    </>
  );
}
