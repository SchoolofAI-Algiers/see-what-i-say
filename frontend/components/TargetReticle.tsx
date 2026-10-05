"use client";

import React from "react";
import { motion } from "framer-motion";

export interface DetectionBox {
  id: string | number;
  label: string;
  confidence: number;
  box: [number, number, number, number]; // [x1, y1, x2, y2] in 0..1 relative coordinates
  distance?: number; // estimated distance in meters
  isAmbiguous?: boolean;
}

interface TargetReticleProps {
  detection: DetectionBox;
  effectMode?: "highlight" | "recolor" | "blur" | "isolate";
  recolorTint?: string;
  isSelected?: boolean;
  onSelect?: () => void;
}

export default function TargetReticle({
  detection,
  effectMode = "highlight",
  recolorTint = "rgba(0, 245, 255, 0.25)",
  onSelect,
}: TargetReticleProps) {
  const [x1, y1, x2, y2] = detection.box;
  const left = `${Math.max(0, Math.min(100, x1 * 100))}%`;
  const top = `${Math.max(0, Math.min(100, y1 * 100))}%`;
  const width = `${Math.max(2, Math.min(100, (x2 - x1) * 100))}%`;
  const height = `${Math.max(2, Math.min(100, (y2 - y1) * 100))}%`;

  const confPercent = (detection.confidence * 100).toFixed(1);
  const distanceEst = detection.distance
    ? `${detection.distance.toFixed(2)}m`
    : `${(1.2 + (1 - (y2 - y1)) * 2.5).toFixed(2)}m`;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      onClick={onSelect}
      className="absolute cursor-pointer pointer-events-auto"
      style={{
        left,
        top,
        width,
        height,
      }}
    >
      {/* Recolor Mode Overlay */}
      {effectMode === "recolor" && (
        <div
          className="absolute inset-0 rounded-sm pointer-events-none transition-all duration-300"
          style={{
            backgroundColor: recolorTint,
            boxShadow: "0 0 20px rgba(0, 245, 255, 0.4)",
          }}
        />
      )}

      {/* Outer Cyan Chamfered Brackets */}
      <div className="absolute inset-0 border border-stark-cyan/40 pointer-events-none">
        {/* Top-Left Bracket */}
        <div className="absolute -top-1 -left-1 w-6 h-6 border-t-2 border-l-2 border-stark-cyan shadow-[0_0_10px_#00F5FF]" />
        <div className="absolute -top-0.5 -left-0.5 w-3 h-3 border-t border-l border-stark-gold" />

        {/* Top-Right Bracket */}
        <div className="absolute -top-1 -right-1 w-6 h-6 border-t-2 border-r-2 border-stark-cyan shadow-[0_0_10px_#00F5FF]" />
        <div className="absolute -top-0.5 -right-0.5 w-3 h-3 border-t border-r border-stark-gold" />

        {/* Bottom-Left Bracket */}
        <div className="absolute -bottom-1 -left-1 w-6 h-6 border-b-2 border-l-2 border-stark-cyan shadow-[0_0_10px_#00F5FF]" />
        <div className="absolute -bottom-0.5 -left-0.5 w-3 h-3 border-b border-l border-stark-gold" />

        {/* Bottom-Right Bracket */}
        <div className="absolute -bottom-1 -right-1 w-6 h-6 border-b-2 border-r-2 border-stark-cyan shadow-[0_0_10px_#00F5FF]" />
        <div className="absolute -bottom-0.5 -right-0.5 w-3 h-3 border-b border-r border-stark-gold" />
      </div>

      {/* Animated Horizontal Laser Sweep Line */}
      <div className="absolute inset-x-0 h-[2px] bg-gradient-to-r from-transparent via-stark-cyan to-transparent animate-laser-sweep pointer-events-none opacity-80 shadow-[0_0_8px_#00F5FF]" />

      {/* Center Micro Crosshair */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none flex items-center justify-center opacity-60">
        <div className="w-full h-[1px] bg-stark-gold/70" />
        <div className="h-full w-[1px] bg-stark-gold/70 absolute" />
      </div>

      {/* Floating Header Tag: Label & Confidence */}
      <div className="absolute -top-8 left-0 flex items-center gap-2 pointer-events-none whitespace-nowrap">
        <div className="hud-glass px-2.5 py-0.5 flex items-center gap-1.5 border-l border-l-stark-gold">
          <span className="w-1.5 h-1.5 rounded-full bg-stark-gold animate-ping" />
          <span className="font-orbitron text-[11px] font-bold tracking-wider text-stark-gold uppercase text-glow-gold">
            #{detection.id} {detection.label}
          </span>
          <span className="font-mono text-[10px] text-stark-cyan font-bold tracking-tight">
            [{confPercent}% TGT LOCK]
          </span>
        </div>

        {/* Ambiguity Alert Badge if triggered */}
        {detection.isAmbiguous && (
          <div className="hud-glass bg-stark-crimson/20 border-stark-crimson px-2 py-0.5 flex items-center gap-1 text-[9px] font-orbitron text-stark-crimson text-glow-crimson uppercase animate-bounce">
            <span>⚠️</span> AMBIGUOUS TARGET
          </div>
        )}
      </div>

      {/* Floating Footer Tag: Distance & Coordinates */}
      <div className="absolute -bottom-6 right-0 flex items-center gap-2 pointer-events-none whitespace-nowrap">
        <div className="hud-glass px-2 py-0.5 flex items-center gap-2 text-[9px] font-mono text-slate-300">
          <span>RANGE: <strong className="text-stark-cyan">{distanceEst}</strong></span>
          <span className="text-slate-600">|</span>
          <span className="text-slate-400">
            LOC [{Math.round(x1 * 640)}, {Math.round(y1 * 480)}]
          </span>
        </div>
      </div>
    </motion.div>
  );
}
