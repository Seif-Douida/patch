import type { NextConfig } from "next";

// A static site (Static Web Apps Free hosts files only). Trailing slashes make every page a
// folder with an index.html, which static hosting serves without rewrite rules.
const config: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  reactStrictMode: true,
};

export default config;
