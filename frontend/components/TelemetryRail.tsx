"use client";

import React, { useState, useEffect } from "react";
import { Activity, Cpu, Wifi, ShieldCheck, Zap } from "lucide-react";

interface TelemetryRailProps {
  wsConnected: boolean;
  fps: number;
  inferenceMs: number;
  vramUsedMb: number;
  detectionCount: number;
  className?: string;
}

export default function TelemetryRail({
  wsConnected,
  fps = 60,
  inferenceMs = 28,
  vramUsedMb = 3190,
  detectionCount = 0,
  className = "",
}: TelemetryRailProps) {
  const [pulse, setPulse] = useState(72);

  // slight pulse jitter for ui liveliness
  useEffect(() => {
    const interval = setInterval(() => {
      setPulse(70 + Math.floor(Math.random() * 5));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const vramGb = (vramUsedMb / 1024).toFixed(2);

  return (
    <div
      className={`hud-glass clip-chamfer-tr p-3.5 flex flex-col gap-2.5 border-r-2 border-r-stark-cyan ${className}`}
    >
      {/* header */}
      <div className="flex items-center justify-between gap-4 border-b border-stark-cyan/20 pb-1.5">
        <div className="flex items-center gap-1.5">
          <ShieldCheck className="w-4 h-4 text-stark-gold" />
          <span className="font-orbitron text-[11px] font-bold tracking-wider text-stark-gold uppercase text-glow-gold">
            Mark 85 Armor Diagnostics
          </span>
        </div>
        <span className="font-mono text-[9px] px-1.5 py-0.5 rounded bg-stark-cyan/10 border border-stark-cyan/40 text-stark-cyan">
          TACTICAL HUD
        </span>
      </div>

      {/* metrics grid */}
      <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
        <div className="flex items-center gap-1.5">
          <Wifi
            className={`w-3.5 h-3.5 ${
              wsConnected ? "text-stark-cyan" : "text-stark-crimson animate-ping"
            }`}
          />
          <div className="flex flex-col">
            <span className="text-slate-400 text-[8px]">STREAM LINK</span>
            <span
              className={`font-bold ${
                wsConnected ? "text-stark-cyan" : "text-stark-crimson"
              }`}
            >
              {wsConnected ? "WS:8000 LIVE" : "OFFLINE"}
            </span>
          </div>
        </div>

        {/* fps counter */}
        <div className="flex items-center gap-1.5">
          <Activity className="w-3.5 h-3.5 text-stark-gold" />
          <div className="flex flex-col">
            <span className="text-slate-400 text-[8px]">VIEWPORT RENDER</span>
            <span className="font-bold text-stark-gold">
              {fps.toFixed(1)} FPS LOCKED
            </span>
          </div>
        </div>

        {/* detector latency */}
        <div className="flex items-center gap-1.5">
          <Zap className="w-3.5 h-3.5 text-stark-cyan" />
          <div className="flex flex-col">
            <span className="text-slate-400 text-[8px]">GROUNDING DINO</span>
            <span className="font-bold text-slate-100">
              <strong className="text-stark-cyan">{inferenceMs} ms</strong> GPU
            </span>
          </div>
        </div>

        {/* vram usage */}
        <div className="flex items-center gap-1.5">
          <Cpu className="w-3.5 h-3.5 text-stark-gold" />
          <div className="flex flex-col">
            <span className="text-slate-400 text-[8px]">RTX 4060 VRAM</span>
            <span className="font-bold text-slate-100">
              {vramGb} / 8.00 GB
            </span>
          </div>
        </div>
      </div>

      {/* memory bar */}
      <div className="w-full bg-slate-900 h-1.5 rounded-full overflow-hidden border border-slate-700">
        <div
          className="h-full bg-gradient-to-r from-stark-cyan to-stark-gold"
          style={{ width: `${(vramUsedMb / 8192) * 100}%` }}
        />
      </div>

      {/* footer status */}
      <div className="flex items-center justify-between text-[9px] font-mono text-slate-400 pt-1 border-t border-slate-800">
        <span className="flex items-center gap-1">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-stark-gold animate-pulse" />
          PULSE: <strong className="text-slate-200">{pulse} BPM</strong>
        </span>
        <span>
          LOCKED TARGETS:{" "}
          <strong className="text-stark-gold">{detectionCount}</strong>
        </span>
      </div>
    </div>
  );
}
