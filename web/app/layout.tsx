import type { Metadata } from "next";
import { Atkinson_Hyperlegible_Next } from "next/font/google";
import Link from "next/link";
import type { ReactNode } from "react";

import { Footer } from "@/components/Footer";
import { loadIndex } from "@/lib/data";

import "./globals.css";

// Next has no metrics to size a fallback for this face, so the system sans stands in while it loads.
const sans = Atkinson_Hyperlegible_Next({
  subsets: ["latin"],
  display: "swap",
  adjustFontFallback: false,
  fallback: ["system-ui", "Segoe UI", "sans-serif"],
});

export const metadata: Metadata = {
  title: { default: "PatchPulse", template: "%s | PatchPulse" },
  description: "How players' Steam reviews moved around each game's patches, updated every night.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  const index = loadIndex();
  return (
    <html lang="en-GB" className={sans.className}>
      <body>
        <header className="site-header">
          <Link href="/" className="wordmark">
            PatchPulse
          </Link>
        </header>
        <main>{children}</main>
        <Footer
          attribution={index.attribution}
          generatedAt={index.generated_at}
          dataVersion={index.data_version}
        />
      </body>
    </html>
  );
}
