// Reads the site data at build time. publish-site.yml checks out the site-data branch into
// public/data and sets SITE_DATA_DIR=public/data; everywhere else the build uses the invented
// sample in fixtures/, so local builds and CI never need live data.

import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

import type {
  GameData,
  Header,
  MonitoringFile,
  OpenAlert,
  ReportFile,
  SiteIndex,
} from "./types";

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

// Phase 7 (design D10): the Monitoring page's files. Both may be missing (a night before Phase 7,
// or a local build without them): the page then shows its empty states instead of failing.
export const MONITORING_SCHEMAS: readonly number[] = [1];

export function loadMonitoring(dataDir: string = DATA_DIR): MonitoringFile | null {
  const file = path.join(dataDir, "monitoring.json");
  if (!existsSync(file)) return null;
  const document = JSON.parse(readFileSync(file, "utf8")) as MonitoringFile;
  if (!MONITORING_SCHEMAS.includes(document.schema_version)) {
    throw new Error(
      `monitoring.json has schema ${document.schema_version}; this site reads ${MONITORING_SCHEMAS.join(" or ")}`,
    );
  }
  return document;
}

export function loadAlerts(dataDir: string = DATA_DIR): OpenAlert[] {
  const file = path.join(dataDir, "alerts.json");
  return existsSync(file) ? (JSON.parse(readFileSync(file, "utf8")) as OpenAlert[]) : [];
}
