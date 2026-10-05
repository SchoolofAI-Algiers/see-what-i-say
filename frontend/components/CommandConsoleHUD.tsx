"use client";

import React, { useState, useEffect } from "react";
import {
  Mic,
  Crosshair,
  Sparkles,
  Layers,
  Eye,
  Search,
  Camera,
} from "lucide-react";
import { EffectMode } from "./CommandDock";

interface CommandConsoleHUDProps {
  query: string;
  onQueryChange: (q: string) => void;
  onSubmitQuery: (q: string) => void;
  isRecording: boolean;
  onToggleRecording: () => void;
  effectMode: EffectMode;
  onEffectModeChange: (mode: EffectMode) => void;
  boxThreshold: number;
  onBoxThresholdChange: (v: number) => void;
  textThreshold: number;
  onTextThresholdChange: (v: number) => void;
  cameraActive: boolean;
  onToggleCamera: () => void;
  className?: string;
}

export default function CommandConsoleHUD({
  query,
  onQueryChange,
  onSubmitQuery,
  isRecording,
  onToggleRecording,
  effectMode,
  onEffectModeChange,
  boxThreshold,
  onBoxThresholdChange,
  textThreshold,
  onTextThresholdChange,
  cameraActive,
  onToggleCamera,
  className = "",
}: CommandConsoleHUDProps) {
  const [activeTab, setActiveTab] = useState<"none" | "effects" | "maps" | "cfg">("none");

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        document.activeElement?.tagName === "INPUT" ||
        document.activeElement?.tagName === "TEXTAREA"
      ) {
        if (e.key === "Enter") onSubmitQuery(query);
        return;
      }

      if (e.code === "Space") {
        e.preventDefault();
        onToggleRecording();
      } else if (e.key === "1") {
        onEffectModeChange("highlight");
      } else if (e.key === "2") {
        onEffectModeChange("recolor");
      } else if (e.key === "3") {
        onEffectModeChange("blur");
      } else if (e.key === "4") {
        onEffectModeChange("isolate");
      } else if (e.key.toLowerCase() === "h") {
        setActiveTab((prev) => (prev === "cfg" ? "none" : "cfg"));
      } else if (e.key.toLowerCase() === "c") {
        onToggleCamera();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [query, onToggleRecording, onEffectModeChange, onSubmitQuery, onToggleCamera]);

  return (
    <div className={`flex flex-col items-center gap-2 select-none ${className}`}>
      {/* Expandable Panel for Prompts, Effects, or Config */}
      {activeTab !== "none" && (
        <div className="bg-[#070e1b]/90 border border-stark-gold/60 rounded-lg p-3 max-w-2xl w-full backdrop-blur-xl shadow-[0_0_25px_rgba(255,191,0,0.2)] animate-in fade-in duration-200">
          {activeTab === "cfg" && (
            <div className="flex flex-col gap-2.5">
              {/* Natural Language Query Bar */}
              <div className="flex items-center gap-2">
                <div className="relative flex-1">
                  <Search className="w-4 h-4 text-stark-cyan absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => onQueryChange(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") onSubmitQuery(query);
                    }}
                    placeholder="Speak or enter prompt (e.g. 'person in black shirt', 'water bottle')..."
                    className="w-full bg-slate-900/80 border border-stark-cyan/50 focus:border-stark-gold text-slate-100 placeholder:text-slate-500 pl-9 pr-20 py-2 rounded text-xs font-mono outline-none shadow-[0_0_10px_rgba(0,245,255,0.2)]"
                  />
                  <button
                    onClick={() => onSubmitQuery(query)}
                    className="absolute right-1 top-1/2 -translate-y-1/2 px-3 py-1 bg-stark-cyan/20 hover:bg-stark-cyan/40 border border-stark-cyan text-stark-cyan text-[10px] font-orbitron font-bold rounded uppercase"
                  >
                    LOCK ON
                  </button>
                </div>

                <button
                  onClick={onToggleCamera}
                  className={`px-3 py-1.5 rounded border text-[10px] font-orbitron uppercase flex items-center gap-1.5 transition-all ${
                    cameraActive
                      ? "border-stark-cyan text-stark-cyan bg-stark-cyan/10"
                      : "border-stark-gold text-stark-gold bg-stark-gold/10"
                  }`}
                >
                  <Camera className="w-3.5 h-3.5" />
                  {cameraActive ? "Webcam" : "Demo Room"}
                </button>
              </div>

              {/* Threshold Sliders */}
              <div className="flex items-center justify-between gap-4 flex-wrap pt-1 border-t border-slate-800 text-[10px] font-mono">
                <div className="flex items-center gap-2">
                  <span className="text-slate-400">Box Confidence:</span>
                  <input
                    type="range"
                    min="0.1"
                    max="0.8"
                    step="0.05"
                    value={boxThreshold}
                    onChange={(e) => onBoxThresholdChange(parseFloat(e.target.value))}
                    className="w-24 accent-stark-cyan cursor-pointer"
                  />
                  <span className="text-stark-cyan font-bold">{boxThreshold.toFixed(2)}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-slate-400">Text Alignment:</span>
                  <input
                    type="range"
                    min="0.1"
                    max="0.8"
                    step="0.05"
                    value={textThreshold}
                    onChange={(e) => onTextThresholdChange(parseFloat(e.target.value))}
                    className="w-24 accent-stark-gold cursor-pointer"
                  />
                  <span className="text-stark-gold font-bold">{textThreshold.toFixed(2)}</span>
                </div>
              </div>
            </div>
          )}

          {activeTab === "effects" && (
            <div className="flex items-center justify-center gap-2">
              <button
                onClick={() => onEffectModeChange("highlight")}
                className={`px-3 py-1.5 rounded text-xs font-orbitron uppercase flex items-center gap-1.5 ${
                  effectMode === "highlight"
                    ? "bg-stark-cyan text-black font-bold shadow-[0_0_12px_#00F5FF]"
                    : "text-slate-300 border border-slate-700 hover:text-stark-cyan"
                }`}
              >
                <Crosshair className="w-3.5 h-3.5" />
                [1] Reticle
              </button>
              <button
                onClick={() => onEffectModeChange("recolor")}
                className={`px-3 py-1.5 rounded text-xs font-orbitron uppercase flex items-center gap-1.5 ${
                  effectMode === "recolor"
                    ? "bg-stark-gold text-black font-bold shadow-[0_0_12px_#FFBF00]"
                    : "text-slate-300 border border-slate-700 hover:text-stark-gold"
                }`}
              >
                <Sparkles className="w-3.5 h-3.5" />
                [2] Recolor
              </button>
              <button
                onClick={() => onEffectModeChange("blur")}
                className={`px-3 py-1.5 rounded text-xs font-orbitron uppercase flex items-center gap-1.5 ${
                  effectMode === "blur"
                    ? "bg-stark-plasma text-white font-bold shadow-[0_0_12px_#1E90FF]"
                    : "text-slate-300 border border-slate-700 hover:text-stark-plasma"
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                [3] Blur
              </button>
              <button
                onClick={() => onEffectModeChange("isolate")}
                className={`px-3 py-1.5 rounded text-xs font-orbitron uppercase flex items-center gap-1.5 ${
                  effectMode === "isolate"
                    ? "bg-stark-crimson text-white font-bold shadow-[0_0_12px_#FF2E2E]"
                    : "text-slate-300 border border-slate-700 hover:text-stark-crimson"
                }`}
              >
                <Eye className="w-3.5 h-3.5" />
                [4] Isolate
              </button>
            </div>
          )}

          {activeTab === "maps" && (
            <div className="flex items-center justify-center gap-2 flex-wrap">
              {["person", "laptop", "cell phone", "bottle", "chair", "backpack"].map((tgt) => (
                <button
                  key={tgt}
                  onClick={() => {
                    onQueryChange(tgt);
                    onSubmitQuery(tgt);
                  }}
                  className="px-3 py-1 bg-slate-900 border border-stark-cyan/40 hover:border-stark-gold rounded font-mono text-xs text-slate-200 hover:text-stark-cyan transition-all"
                >
                  #{tgt}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Main Chiseled Gold Cockpit Chin Console (1:1 with Theme Mockup) */}
      <div className="relative bg-[#070e1b]/90 border-2 border-stark-gold rounded-t-xl px-6 py-2.5 flex flex-col items-center gap-2 shadow-[0_0_30px_rgba(255,191,0,0.25)]">
        {/* Inner Cyan Accent Line */}
        <div className="absolute inset-1 border border-stark-cyan/40 rounded-t-lg pointer-events-none" />

        {/* Top Console Title: COMM // [AUDIO-REC] */}
        <div className="font-orbitron text-[9px] md:text-[10px] font-bold tracking-[0.2em] text-stark-cyan text-glow-cyan uppercase">
          COMM // [AUDIO-REC]
        </div>

        {/* Action Button Row */}
        <div className="flex items-center gap-3">
          {/* Glowing Mic Push-to-Talk Button */}
          <button
            onClick={onToggleRecording}
            className={`p-2.5 rounded-lg border flex items-center justify-center transition-all ${
              isRecording
                ? "bg-stark-gold border-stark-gold text-black shadow-[0_0_20px_#FFBF00] scale-105"
                : "bg-stark-cyan/15 border-stark-cyan text-stark-cyan hover:bg-stark-cyan/30 shadow-[0_0_12px_rgba(0,245,255,0.4)]"
            }`}
            title="Push to Talk (SPACE)"
          >
            {isRecording ? (
              <Mic className="w-4 h-4 animate-pulse text-black" />
            ) : (
              <Mic className="w-4 h-4 text-stark-cyan" />
            )}
          </button>

          {/* [ SCAN ] */}
          <button
            onClick={() => onSubmitQuery(query)}
            className="px-4 py-2 bg-slate-900/80 border border-stark-cyan/60 hover:border-stark-cyan hover:bg-stark-cyan/20 text-stark-cyan font-orbitron text-[11px] font-bold tracking-wider rounded uppercase transition-all shadow-[0_0_10px_rgba(0,245,255,0.15)]"
          >
            SCAN
          </button>

          {/* [ WEAPONS / EFFECTS ] */}
          <button
            onClick={() => setActiveTab(activeTab === "effects" ? "none" : "effects")}
            className={`px-4 py-2 border font-orbitron text-[11px] font-bold tracking-wider rounded uppercase transition-all ${
              activeTab === "effects"
                ? "bg-stark-gold text-black border-stark-gold shadow-[0_0_15px_#FFBF00]"
                : "bg-slate-900/80 border-stark-cyan/60 hover:border-stark-gold text-stark-cyan hover:text-stark-gold"
            }`}
          >
            WEAPONS
          </button>

          {/* [ MAPS / PRESETS ] */}
          <button
            onClick={() => setActiveTab(activeTab === "maps" ? "none" : "maps")}
            className={`px-4 py-2 border font-orbitron text-[11px] font-bold tracking-wider rounded uppercase transition-all ${
              activeTab === "maps"
                ? "bg-stark-gold text-black border-stark-gold shadow-[0_0_15px_#FFBF00]"
                : "bg-slate-900/80 border-stark-cyan/60 hover:border-stark-gold text-stark-cyan hover:text-stark-gold"
            }`}
          >
            MAPS
          </button>

          {/* [ HUD CFG ] */}
          <button
            onClick={() => setActiveTab(activeTab === "cfg" ? "none" : "cfg")}
            className={`px-4 py-2 border font-orbitron text-[11px] font-bold tracking-wider rounded uppercase transition-all ${
              activeTab === "cfg"
                ? "bg-stark-gold text-black border-stark-gold shadow-[0_0_15px_#FFBF00]"
                : "bg-slate-900/80 border-stark-cyan/60 hover:border-stark-gold text-stark-cyan hover:text-stark-gold"
            }`}
          >
            HUD CFG
          </button>
        </div>
      </div>
    </div>
  );
}
