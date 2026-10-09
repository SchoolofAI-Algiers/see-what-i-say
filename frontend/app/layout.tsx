import type { Metadata } from "next";
import { Orbitron, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const orbitron = Orbitron({
  subsets: ["latin"],
  variable: "--font-orbitron",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "STARK VISION // Mark LXXXV AR HUD",
  description: "Real-time multimodal speech-guided referring video segmentation powered by Grounding DINO and SAM 2.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark h-full overflow-hidden">
      <body
        className={`${orbitron.variable} ${jetbrainsMono.variable} font-mono bg-stark-obsidian text-slate-100 h-full w-full overflow-hidden select-none antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
