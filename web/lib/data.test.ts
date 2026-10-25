import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { SUPPORTED_SCHEMAS, loadGame, loadIndex, loadReport } from "./data";

// A site-data folder with an empty index, one game file and one report, all at the given schema
// version.
function siteData(schema: number): string {
  const dir = mkdtempSync(path.join(tmpdir(), "site-data-"));
  const header = {
    schema_version: schema,
    data_version: 7,
    generated_at: "2026-10-14T03:30:00Z",
    attribution: "Data from Steam",
  };
  writeFileSync(path.join(dir, "index.json"), JSON.stringify({ ...header, games: [] }));
  mkdirSync(path.join(dir, "games"));
  writeFileSync(
    path.join(dir, "games", "900001.json"),
    JSON.stringify({ ...header, appid: 900001, name: "Example", genres: [], series: [], patches: [] }),
  );
  mkdirSync(path.join(dir, "reports"));
  writeFileSync(
    path.join(dir, "reports", "5001.json"),
    JSON.stringify({ ...header, report: { report_id: "5001-14d", bundle_id: "5001", horizon: 14 } }),
  );
  return dir;
}

describe("site data", () => {
  it("site_reads_schema_3_and_4", () => {
    expect(SUPPORTED_SCHEMAS).toEqual([3, 4]);
    for (const schema of [3, 4]) {
      const dir = siteData(schema);
      expect(loadIndex(dir).schema_version).toBe(schema);
      expect(loadGame(900001, dir).schema_version).toBe(schema);
    }
    expect(loadReport("5001", siteData(4)).report.report_id).toBe("5001-14d");
    // Schema 2 predates the patch estimates; the live data has been at 3 since Phase 4.
    const old = siteData(2);
    expect(() => loadIndex(old)).toThrow("site data has schema 2; this site reads 3 or 4");
    expect(() => loadGame(900001, old)).toThrow("schema 2");
    // A newer exporter than this site must fail the build, not render half the data.
    const future = siteData(5);
    expect(() => loadIndex(future)).toThrow("site data has schema 5; this site reads 3 or 4");
    expect(() => loadGame(900001, future)).toThrow("schema 5");
    expect(() => loadReport("5001", future)).toThrow("schema 5");
  });
});
