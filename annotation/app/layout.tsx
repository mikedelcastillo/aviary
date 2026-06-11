import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import "./globals.css";
import { SaveIndicator } from "@/components/SaveIndicator";

export const metadata: Metadata = {
  title: "Aviary Annotation",
  description: "Custom bird bounding-box annotation tool",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable}`}>
      <body>
        <SaveIndicator />
        {children}
      </body>
    </html>
  );
}
