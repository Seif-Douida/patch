// Reads the site data at build time. publish-site.yml checks out the site-data branch into
// public/data and sets SITE_DATA_DIR=public/data; everywhere else the build uses the invented
// sample in fixtures/, so local builds and CI never need live data.

import { readFileSync } from "node:fs";
import path from "node:path";

import type { GameData, SiteIndex } from "./types";

export const SCHEMA_VERSION = 1;

const DATA_DIR = path.join(process.cwd(), process.env.SITE_DATA_DIR ?? "fixtures");

function read<T>(relative: string): T {
  return JSON.parse(readFileSync(path.join(DATA_DIR, relative), "utf8")) as T;
}

export function loadIndex(): SiteIndex {
  const index = read<SiteIndex>("index.json");
  if (index.schema_version !== SCHEMA_VERSION) {
    // A new exporter with an old site (or the reverse) must fail the build, not render garbage.
    throw new Error(
      `site data has schema ${index.schema_version}; this site reads ${SCHEMA_VERSION}`,
    );
  }
  return index;
}

export function loadGame(appid: number): GameData {
  return read<GameData>(`games/${appid}.json`);
}
