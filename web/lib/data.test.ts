import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { SUPPORTED_SCHEMAS, loadGame, loadIndex } from "./data";

// A site-data folder with an empty index and one game file, both at the given schema version.
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
  return dir;
}

describe("site data", () => {
  it("site_reads_schema_2_and_3", () => {
    expect(SUPPORTED_SCHEMAS).toEqual([2, 3]);
    for (const schema of [2, 3]) {
      const dir = siteData(schema);
      expect(loadIndex(dir).schema_version).toBe(schema);
      expect(loadGame(900001, dir).schema_version).toBe(schema);
    }
    // Schema 1 predates the aspects; no site data has been at 1 since Phase 2's first night.
    const old = siteData(1);
    expect(() => loadIndex(old)).toThrow("site data has schema 1; this site reads 2 or 3");
    expect(() => loadGame(900001, old)).toThrow("schema 1");
    // A newer exporter than this site must fail the build, not render half the data.
    const future = siteData(4);
    expect(() => loadIndex(future)).toThrow("site data has schema 4; this site reads 2 or 3");
    expect(() => loadGame(900001, future)).toThrow("schema 4");
  });
});
