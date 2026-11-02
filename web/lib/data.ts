// Reads the site data at build time. publish-site.yml checks out the site-data branch into
// public/data and sets SITE_DATA_DIR=public/data; everywhere else the build uses the invented
// sample in fixtures/, so local builds and CI never need live data.

import { readFileSync } from "node:fs";
import path from "node:path";

import type { GameData, Header, ReportFile, SiteIndex } from "./types";

// Schema 4 only adds fields (the reports), so this site reads 3 and 4: the live data may still be
// at 3 for a night after a deploy. Schema 2 (before the patch estimates) is no longer read.
export const SUPPORTED_SCHEMAS: readonly number[] = [3, 4];

const DATA_DIR = path.resolve(process.cwd(), process.env.SITE_DATA_DIR ?? "fixtures");

function read<T extends Header>(dataDir: string, relative: string): T {
  const document = JSON.parse(readFileSync(path.join(dataDir, relative), "utf8")) as T;
  if (!SUPPORTED_SCHEMAS.includes(document.schema_version)) {
    // A new exporter with an old site (or the reverse) must fail the build, not render garbage.
    throw new Error(
      `site data has schema ${document.schema_version}; this site reads ${SUPPORTED_SCHEMAS.join(" or ")}`,
    );
  }
  return document;
}

export function loadIndex(dataDir: string = DATA_DIR): SiteIndex {
  return read<SiteIndex>(dataDir, "index.json");
}

export function loadGame(appid: number, dataDir: string = DATA_DIR): GameData {
  return read<GameData>(dataDir, `games/${appid}.json`);
}

export function loadReport(bundleId: string, dataDir: string = DATA_DIR): ReportFile {
  return read<ReportFile>(dataDir, `reports/${bundleId}.json`);
}
