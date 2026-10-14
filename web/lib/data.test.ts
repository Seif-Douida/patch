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
  it("site_reads_schema_1_and_2", () => {
    expect(SUPPORTED_SCHEMAS).toEqual([1, 2]);
    for (const schema of [1, 2]) {
      const dir = siteData(schema);
      expect(loadIndex(dir).schema_version).toBe(schema);
      expect(loadGame(900001, dir).schema_version).toBe(schema);
    }
    // A newer exporter than this site must fail the build, not render half the data.
    const future = siteData(3);
    expect(() => loadIndex(future)).toThrow("site data has schema 3; this site reads 1 or 2");
    expect(() => loadGame(900001, future)).toThrow("schema 3");
  });
});
