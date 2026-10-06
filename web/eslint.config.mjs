import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";
import { defineConfig, globalIgnores } from "eslint/config";

export default defineConfig([
  ...nextVitals,
  ...nextTs,
  // eslint-plugin-react's "detect" reads the file name through an API ESLint 10 removed.
  { settings: { react: { version: "19.3" } } },
  globalIgnores([".next/**", "out/**", "next-env.d.ts", "public/data/**"]),
]);
