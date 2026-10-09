import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        stark: {
          cyan: "#00F5FF",
          gold: "#FFBF00",
          plasma: "#1E90FF",
          crimson: "#FF2E2E",
          obsidian: "#070B12",
          surface: "rgba(7, 11, 18, 0.85)",
          glass: "rgba(10, 18, 30, 0.70)",
          border: "rgba(0, 245, 255, 0.35)",
        },
      },
      fontFamily: {
        orbitron: ["var(--font-orbitron)", "sans-serif"],
        mono: ["var(--font-mono)", "monospace"],
      },
      boxShadow: {
        "cyan-glow": "0 0 20px rgba(0, 245, 255, 0.45), inset 0 0 15px rgba(0, 245, 255, 0.15)",
        "gold-glow": "0 0 20px rgba(255, 191, 0, 0.45), inset 0 0 15px rgba(255, 191, 0, 0.15)",
        "crimson-glow": "0 0 25px rgba(255, 46, 46, 0.55)",
      },
      animation: {
        "radar-spin": "spin 8s linear infinite",
        "radar-reverse": "spin 6s linear infinite reverse",
        "laser-sweep": "laserSweep 2.8s ease-in-out infinite",
        "pulse-glow": "pulseGlow 2.4s ease-in-out infinite",
      },
      keyframes: {
        laserSweep: {
          "0%, 100%": { top: "0%", opacity: "0.2" },
          "50%": { top: "100%", opacity: "0.9" },
        },
        pulseGlow: {
          "0%, 100%": { opacity: "0.4", transform: "scale(0.98)" },
          "50%": { opacity: "1", transform: "scale(1.02)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
