"use client";

import React from "react";
import { Battery, Heart } from "lucide-react";

interface ArmorDiagnosticsHUDProps {
  fps?: number;
  inferenceMs?: number;
  vramUsedMb?: number;
  wsConnected?: boolean;
  className?: string;
}

export default function ArmorDiagnosticsHUD({
  fps = 60,
  inferenceMs = 28,
  vramUsedMb = 3190,
  wsConnected = true,
  className = "",
}: ArmorDiagnosticsHUDProps) {
  const vramGb = (vramUsedMb / 1024).toFixed(2);

  return (
    <div className={`flex flex-col items-end gap-1.5 select-none ${className}`}>
      {/* Top Header Tag */}
      <div className="flex items-center gap-2">
        <span className="font-orbitron text-[10px] md:text-[11px] font-bold tracking-wider text-stark-gold text-glow-gold uppercase">
          MARK 85 ARMOR / SYSTEM DIAGNOSTICS{" "}
          <span className="text-stark-cyan text-glow-cyan">[LIVE]</span>
        </span>
        <div
          className={`w-1.5 h-1.5 rounded-full ${
            wsConnected ? "bg-stark-gold" : "bg-stark-crimson"
          } animate-ping`}
        />
      </div>

      {/* Main Chamfered Cyan/Gold Pod */}
      <div className="relative bg-[#070e1b]/85 border-2 border-stark-cyan rounded-tr-xl rounded-bl-xl p-3 pr-4 shadow-[0_0_25px_rgba(0,245,255,0.25)] flex items-center gap-4">
        {/* Outer Gold Trim Accent Lines */}
        <div className="absolute -top-1 -right-1 w-6 h-6 border-t-2 border-r-2 border-stark-gold" />
        <div className="absolute -bottom-1 -left-1 w-6 h-6 border-b-2 border-l-2 border-stark-gold" />

        {/* Battery Icon Indicator */}
        <div className="flex flex-col items-center justify-center p-1 bg-stark-cyan/10 border border-stark-cyan/40 rounded">
          <Battery className="w-5 h-5 text-stark-cyan animate-pulse" />
          <span className="text-[7px] font-mono text-stark-cyan font-bold">100%</span>
        </div>

        {/* System Stats Column */}
        <div className="flex flex-col gap-1 font-mono text-[10px] tracking-tight">
          <div className="flex justify-between items-center gap-4">
            <span className="text-slate-400">BATTERY</span>
            <span className="text-stark-cyan font-bold">[100%]</span>
          </div>
          <div className="flex justify-between items-center gap-4">
            <span className="text-slate-400">SHIELDS</span>
            <span className="text-stark-cyan font-bold">[OK]</span>
          </div>
          <div className="flex justify-between items-center gap-4">
            <span className="text-slate-400">HEALTH</span>
            <span className="text-stark-gold font-bold">[99%]</span>
          </div>
          <div className="flex justify-between items-center gap-4">
            <span className="text-slate-400">NANOTECH</span>
            <span className="text-stark-cyan font-bold">[OPTIMAL]</span>
          </div>
        </div>

        {/* Vertical Divider */}
        <div className="w-[1px] h-12 bg-gradient-to-b from-transparent via-stark-cyan/50 to-transparent" />

        {/* Pulsing EKG Heartbeat Badge */}
        <div className="flex flex-col items-center justify-center gap-1">
          <div className="relative w-10 h-10 rounded-full border border-stark-cyan/60 flex items-center justify-center bg-stark-cyan/10 shadow-[0_0_12px_#00F5FF]">
            <Heart className="w-5 h-5 text-stark-cyan animate-pulse" />
          </div>
          <span className="text-[8px] font-mono text-stark-cyan">72 BPM</span>
        </div>
      </div>

      {/* Real Hardware Sub-Telemetry Bar */}
      <div className="flex items-center gap-3 font-mono text-[9px] text-slate-400 tracking-wider">
        <span>
          GPU: <strong className="text-stark-cyan">RTX 4060</strong>
        </span>
        <span className="text-slate-600">|</span>
        <span>
          VRAM: <strong className="text-stark-gold">{vramGb}/8.0 GB</strong>
        </span>
        <span className="text-slate-600">|</span>
        <span>
          INFER: <strong className="text-stark-cyan">{inferenceMs}ms</strong>
        </span>
        <span className="text-slate-600">|</span>
        <span>
          FPS: <strong className="text-stark-gold">{fps.toFixed(1)}</strong>
        </span>
      </div>
    </div>
  );
}
