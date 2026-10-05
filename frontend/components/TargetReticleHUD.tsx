"use client";

import React from "react";
import { motion } from "framer-motion";
import { Lock, Shield, Compass } from "lucide-react";

export interface DetectionBox {
  id: string | number;
  label: string;
  confidence: number;
  box: [number, number, number, number]; // [x1, y1, x2, y2] normalized 0..1
  distance?: number;
  isAmbiguous?: boolean;
}

interface TargetReticleHUDProps {
  detection: DetectionBox;
  effectMode?: "highlight" | "recolor" | "blur" | "isolate";
  recolorTint?: string;
}

export default function TargetReticleHUD({
  detection,
  effectMode = "highlight",
  recolorTint = "rgba(0, 245, 255, 0.28)",
}: TargetReticleHUDProps) {
  const [x1, y1, x2, y2] = detection.box;
  const left = `${Math.max(0, Math.min(100, x1 * 100))}%`;
  const top = `${Math.max(0, Math.min(100, y1 * 100))}%`;
  const width = `${Math.max(4, Math.min(100, (x2 - x1) * 100))}%`;
  const height = `${Math.max(4, Math.min(100, (y2 - y1) * 100))}%`;

  const confPercent = (detection.confidence * 100).toFixed(1);
  const estDistance = detection.distance
    ? `${detection.distance.toFixed(2)}m`
    : "1.84m";

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.2 }}
      className="absolute pointer-events-none select-none z-10"
      style={{
        left,
        top,
        width,
        height,
      }}
    >
      {/* Recolor mode fill tint */}
      {effectMode === "recolor" && (
        <div
          className="absolute inset-0 rounded-lg pointer-events-none transition-all duration-300"
          style={{
            backgroundColor: recolorTint,
            boxShadow: "0 0 30px rgba(0, 245, 255, 0.5)",
          }}
        />
      )}

      {/* TOP TITLE TAG: #1 PERSON [94.2% TARGET ACQUISITION] */}
      <div className="absolute -top-10 left-1/2 -translate-x-1/2 flex flex-col items-center whitespace-nowrap">
        <span className="font-orbitron text-xs md:text-sm font-bold tracking-widest text-stark-cyan text-glow-cyan uppercase">
          #{detection.id} {detection.label}
        </span>
        <span className="font-orbitron text-[10px] md:text-[11px] font-bold tracking-wider text-stark-gold text-glow-gold uppercase">
          [{confPercent}% TARGET ACQUISITION]
        </span>
      </div>

      {/* 4 AUTHENTIC MARK 85 CHAMFERED BRACKETS */}
      {/* Top-Left Bracket */}
      <div className="absolute -top-2 -left-2 w-10 h-10 pointer-events-none">
        <svg viewBox="0 0 40 40" className="w-full h-full">
          <path
            d="M 2 38 L 2 12 Q 2 2 12 2 L 38 2"
            fill="none"
            stroke="#00F5FF"
            strokeWidth="3.5"
            strokeLinecap="round"
            className="filter drop-shadow-[0_0_8px_#00F5FF]"
          />
          <path
            d="M 6 36 L 6 14 Q 6 6 14 6 L 36 6"
            fill="none"
            stroke="#FFBF00"
            strokeWidth="2"
            strokeLinecap="round"
            opacity="0.9"
          />
        </svg>
      </div>

      {/* Top-Right Bracket */}
      <div className="absolute -top-2 -right-2 w-10 h-10 pointer-events-none">
        <svg viewBox="0 0 40 40" className="w-full h-full">
          <path
            d="M 38 38 L 38 12 Q 38 2 28 2 L 2 2"
            fill="none"
            stroke="#00F5FF"
            strokeWidth="3.5"
            strokeLinecap="round"
            className="filter drop-shadow-[0_0_8px_#00F5FF]"
          />
          <path
            d="M 34 36 L 34 14 Q 34 6 26 6 L 4 6"
            fill="none"
            stroke="#FFBF00"
            strokeWidth="2"
            strokeLinecap="round"
            opacity="0.9"
          />
        </svg>
      </div>

      {/* Bottom-Left Bracket */}
      <div className="absolute -bottom-2 -left-2 w-10 h-10 pointer-events-none">
        <svg viewBox="0 0 40 40" className="w-full h-full">
          <path
            d="M 2 2 L 2 28 Q 2 38 12 38 L 38 38"
            fill="none"
            stroke="#00F5FF"
            strokeWidth="3.5"
            strokeLinecap="round"
            className="filter drop-shadow-[0_0_8px_#00F5FF]"
          />
          <path
            d="M 6 4 L 6 26 Q 6 34 14 34 L 36 34"
            fill="none"
            stroke="#FFBF00"
            strokeWidth="2"
            strokeLinecap="round"
            opacity="0.9"
          />
        </svg>
      </div>

      {/* Bottom-Right Bracket */}
      <div className="absolute -bottom-2 -right-2 w-10 h-10 pointer-events-none">
        <svg viewBox="0 0 40 40" className="w-full h-full">
          <path
            d="M 38 2 L 38 28 Q 38 38 28 38 L 2 38"
            fill="none"
            stroke="#00F5FF"
            strokeWidth="3.5"
            strokeLinecap="round"
            className="filter drop-shadow-[0_0_8px_#00F5FF]"
          />
          <path
            d="M 34 4 L 34 26 Q 34 34 26 34 L 4 34"
            fill="none"
            stroke="#FFBF00"
            strokeWidth="2"
            strokeLinecap="round"
            opacity="0.9"
          />
        </svg>
      </div>

      {/* BLAZING BRIGHT CYAN HORIZONTAL LASER SCANNER BEAM */}
      <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-[3px] bg-gradient-to-r from-transparent via-[#00F5FF] to-transparent shadow-[0_0_20px_#00F5FF,0_0_35px_#00F5FF] animate-laser-sweep pointer-events-none">
        {/* Center intense flare */}
        <div className="absolute left-1/2 -translate-x-1/2 -top-1 w-6 h-3 bg-white rounded-full blur-[2px] opacity-90" />
      </div>

      {/* LEFT-SIDE TELEMETRY RAIL (ARMOR SYS, TGT LOCK, AMMO) */}
      <div className="absolute top-1/2 -translate-y-1/2 -left-32 hidden sm:flex flex-col gap-1.5 font-mono text-[9px] text-slate-300">
        <div className="flex items-center gap-1.5 text-stark-cyan font-bold">
          <Shield className="w-3.5 h-3.5 text-stark-cyan" />
          <span>ARMOR SYS</span>
        </div>
        <div className="flex items-center gap-1.5 text-stark-gold font-bold">
          <Lock className="w-3.5 h-3.5 text-stark-gold animate-pulse" />
          <span>TGT LOCK 🔒</span>
        </div>
        <div className="text-slate-400 text-[8px] leading-tight">
          AMMO [100%]<br />
          NANOTECH OPT<br />
          CAL [0x8F9B]
        </div>
      </div>

      {/* RIGHT-SIDE TELEMETRY RAIL (ENVIRONMENT, RANGE) */}
      <div className="absolute top-1/2 -translate-y-1/2 -right-32 hidden sm:flex flex-col gap-1.5 font-mono text-[9px] text-slate-300">
        <div className="flex items-center gap-1.5 text-stark-cyan font-bold">
          <Compass className="w-3.5 h-3.5 text-stark-cyan" />
          <span>ENVIRONMENT</span>
        </div>
        <div className="text-stark-gold font-bold">
          [OPTIMAL]
        </div>
        <div className="text-slate-400 text-[8px] leading-tight">
          DIST: <strong className="text-stark-cyan">{estDistance}</strong><br />
          ATM: 101.3 kPa<br />
          TEMP: 21.4°C
        </div>
      </div>
    </motion.div>
  );
}
