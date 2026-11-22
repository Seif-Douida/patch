import type { Metadata } from "next";
import { Atkinson_Hyperlegible_Next } from "next/font/google";
import type { ReactNode } from "react";

import { Footer } from "@/components/Footer";
import { HeaderTitleProvider } from "@/components/HeaderTitle";
import { SiteHeader } from "@/components/SiteHeader";
import { loadIndex } from "@/lib/data";
import { SITE_DESCRIPTION } from "@/lib/site";

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
  description: SITE_DESCRIPTION,
};

export default function RootLayout({ children }: { children: ReactNode }) {
  const index = loadIndex();
  return (
    <html lang="en-GB" className={sans.className}>
      <body>
        <HeaderTitleProvider>
          <SiteHeader />
          <main className="site-main">{children}</main>
          <Footer
            attribution={index.attribution}
            generatedAt={index.generated_at}
            dataVersion={index.data_version}
          />
        </HeaderTitleProvider>
      </body>
    </html>
  );
}
